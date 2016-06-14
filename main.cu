#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <vector_types.h>
#include "cutil_math.h"

using namespace std;

#define WIDTH 1024
#define HEIGHT 1024
#define ASPECT WIDTH/HEIGHT
#define SAMPLES 1000
#define MAX_DEPTH 16
#define PPT 1
#define BLOCKS 512
#define THREADS 128

enum { NELEMS = WIDTH * HEIGHT };

typedef struct Ray {
	float3 direction;
	float3 origin;
}Ray;

typedef struct Camera {
	float3 lowleftcorner;
	float3 horizontal;
	float3 vertical;
	float3 origin;
}Camera;

typedef struct Material {
	float3 color;
	float reflectivity;
	float refractivity;
	float3 emissivity;
	float ior;
}Material;

typedef struct Path {
	Ray reverse;
	int depth;
}Path;

typedef struct Polygon {
	float3 dot[3];
	float3 normal;
	float D;
	Material material;
}Polygon;

typedef struct Hit_record {
	float3 point;
	float3 normal;
	float depth;
	Material material;
}Hit_record;

//**************RANDOM**************//

__device__ float rand_f(curandState* globalState) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = globalState[indx];
	float RANDOM = curand_uniform( &localState );
	globalState[indx] = localState;
	return RANDOM;
}

__device__ float rand_f2m1(curandState * gState) {
	return rand_f(gState) * 2.0f - 1.0f;
}

//**************SUPPORTS_FUNCTIONS**************//

__device__ void create_camera(Camera &cam, float3 pov, float3 target, float fov) {
	float alpha = fov * M_PI / 180.0f;
	float vfov = tanf(alpha / 2.0f);
	float hfov = ASPECT * vfov;
	float3 dz = normalize(pov - target);
	float3 dx = normalize(cross(make_float3(0,1,0), dz));
	float3 dy = cross(dx, dz);
	cam.lowleftcorner = pov - hfov * dx - vfov * dy - dz;
	cam.horizontal = 2 * hfov * dx;
	cam.vertical = 2 * vfov * dy;
	cam.origin = pov;
}

__device__ Ray shoot_ray(int stage, Camera &cam, curandState* gState) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x + stage * BLOCKS * THREADS;
	float w_off = (float)(indx % WIDTH + rand_f(gState)) / (float)(WIDTH);
	float h_off = (float)(indx / WIDTH + rand_f(gState)) / (float)(HEIGHT);
	Ray tmp;
	tmp.origin = cam.origin;
	tmp.direction = normalize(cam.lowleftcorner + w_off * cam.horizontal + h_off * cam.vertical - cam.origin);
	// tmp.direction = cam.lowleftcorner + w_off * cam.horizontal + h_off * cam.vertical - cam.origin;

	return tmp;
}

__device__ bool try_dot(Polygon p, float3 point) {
	if (dot(cross(p.dot[1] - p.dot[0], point - p.dot[0]), p.normal) < 0.001f) return false;
	if (dot(cross(point - p.dot[0], p.dot[2] - p.dot[0]), p.normal) < 0.001f) return false;
	if (dot(cross(p.dot[1] - point, p.dot[2] - point), p.normal) < 0.001f) return false;
	return true;
}

__device__ Ray shoot_ray(Polygon p, curandState *gState) {
	float3 dir = normalize(make_float3(rand_f2m1(gState), rand_f2m1(gState), rand_f2m1(gState)));
	float3 ori, a = p.dot[1] - p.dot[0], b = p.dot[2] - p.dot[0];
	if (dot(dir, p.normal) < 0) dir = -dir;
	float p1, p2;
	do {
		p1 = rand_f(gState);
		p2 = rand_f(gState);
		ori = p.dot[0] + p1 * a + p2 * b;
	} while (!try_dot(p, ori));
	return {dir, ori};
}

__device__ float3 get_ray(Ray r, float k) {
	return r.origin + k * r.direction;
}

__device__ float3 reflection(float3 v1, float3 v2) {
	return v1 - 2.0f * dot(v1, v2) * v2;
}

__device__ bool diffusion(curandState* gState, Ray r, Ray &scattered, Hit_record &rec, float3 &attenuated){
	if (rec.material.emissivity.x > 0 || rec.material.emissivity.y > 0 
		|| rec.material.emissivity.z > 0) return false;

	float ran = rand_f(gState);

	if (ran < rec.material.reflectivity && rec.material.refractivity == 0.0f) {
		// metal
		float3 refl = normalize(reflection(r.direction, rec.normal));
		scattered = {refl, rec.point};
		attenuated = rec.material.color;
		return (dot(scattered.direction, rec.normal) > 0);
	} else if (rec.material.refractivity == 0.0f) {
		// matte
		float3 v1 = {rand_f2m1(gState), rand_f2m1(gState), rand_f2m1(gState)};
		v1 = normalize(v1);
		if (dot(v1, rec.normal) < 0.0f) v1 = -v1;
		scattered = {v1, rec.point};
		attenuated = rec.material.color;
		return true;
	}
	return true;
}

__device__ bool get_intersection(Ray r, Hit_record &rec, float dist, Polygon p) {
	if (dot(p.normal, r.direction) == 0) return false;

	float A = p.normal.x, B = p.normal.y, C = p.normal.z, D = p.D;
	float3 point, o = r.origin, d = r.direction;

	point.y = o.y * ((A * d.x + C * d.z)/d.y) - D - A * o.x - C * o.z;
	point.y = point.y / ((A * d.x + C * d.z)/d.y + B);
	point.x = d.x * ((point.y - o.y)/d.y) + o.x;
	point.z = d.z * ((point.y - o.y)/d.y) + o.z;

	if (dot(r.origin - point, -p.normal) > 0.0f) return false;
	if (dot(r.origin - point, r.direction) > 0.0f) return false;
	float depth = length(r.origin - point);
	if (depth < dist && depth > 0.001f) {
		if (!try_dot(p, point)) return false;
		rec.point = point;
		rec.normal = p.normal;
		rec.depth = depth;
		rec.material = p.material;
		return true;	
	}
	return false;
}

__device__ bool nearest_intersection (Ray r, Polygon *scene, Hit_record &rec, int n_obj) {
	float dist = 1E+37f;
	bool hit = false;
	for (int i = 0; i < n_obj; ++i) {
		if (get_intersection(r, rec, dist, scene[i])) {
			hit = true;
			dist = rec.depth;
		}
	}
	return hit;

}

__device__ float3 inverse_raytrace(curandState* gState, Path *pMap, int path_num, Polygon *scene, Ray r, int n_obj) {
	Ray primary = r;
	Hit_record rec;
	float3 attenuated;
	float3 emitted;
	float3 composite = {0.0f, 0.0f, 0.0f};
	float3 counted = {1.0f, 1.0f, 1.0f};
	Ray scattered;
	for (int i = 0; i < MAX_DEPTH; ++i) {
		if (nearest_intersection(primary, scene, rec, n_obj)) {
			// return rec.material.color;
			emitted = rec.material.emissivity;
			if (diffusion(gState, primary, scattered, rec, attenuated)) {
				primary = scattered;
				composite += emitted * counted;
				counted = counted * attenuated;
			} else {
				return composite + emitted * counted;
			}
		} else {
			return {0, 0, 0};
		}
	}
	int it = (int) (rand_f(gState) * path_num);
	if (rec.material.reflectivity != 1) {
		float3 dir = pMap[it].reverse.origin - primary.origin;
		float dist = length(dir);
		dir = dir/dist;
		primary = {dir, primary.origin};
		if (nearest_intersection(primary, scene, rec, n_obj)) {
			if (length(rec.point - primary.origin) < dist) return {0, 0, 0};
			emitted = rec.material.emissivity;
			primary = { pMap[it].reverse.direction, pMap[it].reverse.origin};
			composite += emitted * counted;
			counted = counted * rec.material.color;
		}
		for (int i = 0; i < pMap[it].depth + 1; ++i) {
			if (nearest_intersection(primary, scene, rec, n_obj)) {
				// return rec.material.color;
				if (rec.material.reflectivity > 0) rec.material.reflectivity = 1;
				emitted = rec.material.emissivity;
				if (diffusion(gState, primary, scattered, rec, attenuated)) {
					primary = scattered;
					composite += emitted * counted;
					counted = counted * attenuated;
				} else {
					return composite + emitted * counted;
				}
			} else {
				return {0, 0, 0};
			}
		}
	}

	return {0, 0, 0};
	// return composite + emitted * counted;
}

__device__ Path direct_raytrace(curandState* gState, Polygon *scene, Ray r, int n_obj) {
	Path guide = {{{0, 0, 0}, {0, 0, 0}}, 0};
	Ray primary = r;
	Hit_record rec;
	// int j;
	for (int i = 0; i < MAX_DEPTH; ++i) {
		if (nearest_intersection(primary, scene, rec, n_obj)) {
			if (rec.material.emissivity.x > 0 || rec.material.emissivity.y > 0
				|| rec.material.emissivity.z > 0) {
				guide = {{{0, 0, 0}, {0, 0, 0}}, 0};
				return guide;
			} else if (rec.material.reflectivity == 0.0f) {
				guide.reverse.origin = rec.point;
				guide.reverse.direction = -r.direction;
				guide.depth = i;
				return guide;
			} else if (rec.material.reflectivity > 0.0f) {
				float3 refl = normalize(reflection(primary.direction, rec.normal));
				primary = {refl, rec.point};
			}
		} else {
			guide = {{{0, 0, 0}, {0, 0, 0}}, 0};
			return guide;
		}
	}
	return guide;
}

//**************KERNEL**************//

__global__ void direct(curandState* gState, unsigned long seed, Polygon *scene, Path *result, int n_obj, int *ligths, int light_obj) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init (seed, indx, 0, &gState[indx]);
	int j, shift = indx * PPT;
	// result[shift].reverse.direction = {0, 0, 0};
	Ray primary;
	for (int i = 0; i < PPT; ++i) {
		j = ligths[(int)(rand_f(gState) * light_obj)];
		primary = shoot_ray(scene[j], gState);
		result[shift + i] = direct_raytrace(gState, scene, primary, n_obj);
	}
}

__global__ void inverse(int stage, curandState* gState, Path *pMap, int path_num, Polygon *scene, float3 *result, int n, int n_obj) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	Camera cam;
	create_camera(cam, make_float3(-1800, 525.55f, 0), make_float3(0, 350, 0), 40.0f);
	
	if (indx < n) {
		Ray primary;
		result[indx + stage * BLOCKS * THREADS] = {0, 0, 0};
		float scaled = 1.0f / (float) SAMPLES;
		for (int i = 0; i < SAMPLES; ++i) {
			primary = shoot_ray(stage, cam, gState);
			result[indx + stage * BLOCKS * THREADS] += inverse_raytrace(gState, pMap, path_num, scene, primary, n_obj) * scaled;
		}
	}
}

//**************CPU**************//


// __host__ inline float gamma_correction(float val) {return val;}
// __host__ inline int gamma_correction(float val) {return val * 256;}
__host__ inline int gamma_correction(float val) { return int(255.0f * sqrt(fmaxf(0.0f, fminf(1.0f, val))));}

__host__ void find_normal(Polygon &p);
__host__ Polygon * init_polygons(int &n_obj);
__host__ void * alloc_mem_cpu(size_t size);
__host__ void mem_cpy_to_gpu(void *d, void *g, size_t size);
__host__ void * alloc_mem_gpu(size_t size);
__host__ void mem_cpy_to_cpu(void *g, void *d, size_t size);
__host__ void cudaErrors(cudaError_t error);


int main() {
	/* Allocate vectors on host */
	int n_obj;
	size_t sizer = sizeof(float3) * NELEMS;
	size_t sizec = sizeof(curandState) * NELEMS;
	size_t sizep = sizeof(Path) * BLOCKS * THREADS * PPT;
	Polygon *h_scene = NULL;
	Path *h_path = (Path *) alloc_mem_cpu(sizep);
	// float3 *h_result = (float3 *) alloc_mem_cpu(sizer);

	h_scene = init_polygons(n_obj);
	size_t sizes = sizeof(Polygon) * n_obj;

	int light_obj = 0;
	for (int i = 0; i < n_obj; ++i) {
		if (h_scene[i].material.emissivity.x > 0 || h_scene[i].material.emissivity.y > 0
			|| h_scene[i].material.emissivity.z > 0)
			++light_obj;
	}
	size_t sizel = sizeof(int *) * light_obj;
	int *h_lights = (int *) alloc_mem_cpu(sizel);
	for (int i = 0, j = 0; i < n_obj; ++i) {
		if (h_scene[i].material.emissivity.x > 0 || h_scene[i].material.emissivity.y > 0
			|| h_scene[i].material.emissivity.z > 0) {
			h_lights[j] = i; 
			++j;
		}
	}
	/* Allocate vectors on device */
	Polygon *d_scene = NULL;
	int *d_ligths = NULL;
	Path *d_path = NULL;
	// float3 *d_result = NULL;
	curandState* d_states = NULL;

	d_scene = (Polygon *) alloc_mem_gpu(sizes);
	d_ligths = (int *) alloc_mem_gpu(sizel);
	d_path = (Path *) alloc_mem_gpu(sizep);
	// d_result = (float3 *)alloc_mem_gpu(sizer);    
	d_states = (curandState*) alloc_mem_gpu(sizec);

	/* Copy the host vectors to device */

	mem_cpy_to_gpu(h_scene, d_scene, sizes);	
	mem_cpy_to_gpu(h_lights, d_ligths, sizel);

	int threadsPerBlock = THREADS;
	int blocksPerGrid = BLOCKS; //(NELEMS + threadsPerBlock - 1) / threadsPerBlock;
	int maxStage = NELEMS/(BLOCKS * THREADS) + (NELEMS%(BLOCKS * THREADS) != 0) * 1;
	

	direct<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL), d_scene, d_path, n_obj, d_ligths, light_obj);
	cudaError_t error;
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << "Failed to launch inverse!\n";
		cudaErrors(error);
		exit(EXIT_FAILURE);
	}
	mem_cpy_to_cpu(d_path, h_path, sizep);
	cout << "direct method is done!\n";
	int num_path = 0;
	for (int i = 0; i < BLOCKS * THREADS * PPT; ++i) {
		if (h_path[i].reverse.direction.x != 0 || h_path[i].reverse.direction.y != 0
			|| h_path[i].reverse.direction.z != 0) ++num_path;
	}
	cout << num_path << "\n";
	free(h_lights);
	cudaFree(d_ligths);
	mem_cpy_to_gpu(h_path, d_path, sizep);
	float3 *h_result = (float3 *) alloc_mem_cpu(sizer);
	float3 *d_result = NULL;
	d_result = (float3 *) alloc_mem_gpu(sizer);

	for (int i = 0; i < maxStage; ++i) {
		inverse<<<blocksPerGrid, threadsPerBlock>>>(i, d_states, d_path, num_path, d_scene, d_result, BLOCKS * THREADS, n_obj);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			cout << "Failed to launch inverse!\n";
			cudaErrors(error);
			exit(EXIT_FAILURE);
		}
	}
		
	mem_cpy_to_cpu(d_result, h_result, sizer);
	ofstream fout("img.ppm");
	fout << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
	for (int i = 0; i < NELEMS; ++i) {
		fout << gamma_correction(h_result[i].x) << "\t\t\t\t"
				<< gamma_correction(h_result[i].y) << "\t\t\t\t"
				<< gamma_correction(h_result[i].z) << "\n";
	}
	fout.close();
	cout << "We Fine!\n";
	cudaFree(d_path);
	cudaFree(d_scene);
	cudaFree(d_result);
	cudaFree(d_states);
	free(h_path);
	free(h_result);
	free(h_scene);
	cudaDeviceReset();
	return 0;
}

__host__ void find_normal(Polygon &p) {
	float3 a = p.dot[1] - p.dot[0], b = p.dot[2] - p.dot[1];
	p.normal = normalize(cross(a, b));
}

__host__ void find_factor(Polygon &p) {
	p.D = - p.normal.x * p.dot[0].x - p.normal.y * p.dot[0].y - p.normal.z * p.dot[0].z;
}

__host__ Material * init_materials(int &m) {
	ifstream inMat("inMat");
	inMat >> m;
	size_t sizeMat = sizeof(Material) * m;
	Material *materials = (Material *) alloc_mem_cpu(sizeMat);
    for (int i = 0; i < m; ++i) {
		inMat >> materials[i].color.x >> materials[i].color.y >> materials[i].color.z;
		inMat >> materials[i].reflectivity;
		inMat >> materials[i].refractivity;
		inMat >> materials[i].emissivity.x >> materials[i].emissivity.y >> materials[i].emissivity.z;
		inMat >> materials[i].ior;
	}
	inMat.close();
	return materials;
}

__host__ Polygon * init_polygons(int &n_obj) {
	int m, box;
	Material *materials = init_materials(m);
	ifstream infile("inPolygon");
	infile >> n_obj >> box;

	size_t sizeScene = sizeof(Polygon) * (n_obj + box * 12);
	Polygon *scene = (Polygon *)alloc_mem_cpu(sizeScene);

	for (int i = 0; i < n_obj; ++i) {
		for (int j = 0; j < 3; ++j)
	    	infile >> scene[i].dot[j].x >> scene[i].dot[j].y >> scene[i].dot[j].z;
	    int tmp;
	    infile >> tmp;
	    scene[i].material = materials[tmp];
	    find_normal(scene[i]);
	    find_factor(scene[i]);
	    // cout << scene[i].normal.x << ":" << scene[i].normal.y << ":" << scene[i].normal.z << "\n\n";
	}
	float3 dot[8];
	for (int i = 0; i < box; ++i) {
		for (int j = 0; j < 4; ++j) {
			infile >> dot[j].x >> dot[j].y >> dot[j].z;
			// cout << "[" << dot[j].x << ":" << dot[j].y << ":" << dot[j].z << "]";
		}
		// cout << "\n";

		float h;
		infile >> h;

		for (int j = 0; j < 4; ++j) {
			dot[4 + j] = dot[j];
			dot[4 + j].y = dot[j].y + h;
			// cout << "[" << dot[4 + j].x << ":" << dot[4 + j].y << ":" << dot[4 + j].z << "]";
		}
		// cout << "\n";
		
		int tmp;
		infile >> tmp;

		scene[n_obj + i * 12].dot[0] = dot[0];
		scene[n_obj + i * 12].dot[1] = dot[1];
		scene[n_obj + i * 12].dot[2] = dot[2];
		scene[n_obj + i * 12].material = materials[tmp];
		find_normal(scene[n_obj + i * 12]);
		find_factor(scene[n_obj + i * 12]);

		scene[n_obj + i * 12 + 1].dot[0] = dot[2];
		scene[n_obj + i * 12 + 1].dot[1] = dot[3];
		scene[n_obj + i * 12 + 1].dot[2] = dot[0];
		scene[n_obj + i * 12 + 1].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 1]);
		find_factor(scene[n_obj + i * 12 + 1]);

		scene[n_obj + i * 12 + 2].dot[0] = dot[4];
		scene[n_obj + i * 12 + 2].dot[1] = dot[6];
		scene[n_obj + i * 12 + 2].dot[2] = dot[5];
		scene[n_obj + i * 12 + 2].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 2]);
		find_factor(scene[n_obj + i * 12 + 2]);

		scene[n_obj + i * 12 + 3].dot[0] = dot[6];
		scene[n_obj + i * 12 + 3].dot[1] = dot[4];
		scene[n_obj + i * 12 + 3].dot[2] = dot[7];
		scene[n_obj + i * 12 + 3].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 3]);
		find_factor(scene[n_obj + i * 12 + 3]);

		scene[n_obj + i * 12 + 4].dot[0] = dot[0];
		scene[n_obj + i * 12 + 4].dot[1] = dot[3];
		scene[n_obj + i * 12 + 4].dot[2] = dot[4];
		scene[n_obj + i * 12 + 4].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 4]);
		find_factor(scene[n_obj + i * 12 + 4]);

		scene[n_obj + i * 12 + 5].dot[0] = dot[4];
		scene[n_obj + i * 12 + 5].dot[1] = dot[3];
		scene[n_obj + i * 12 + 5].dot[2] = dot[7];
		scene[n_obj + i * 12 + 5].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 5]);
		find_factor(scene[n_obj + i * 12 + 5]);

		scene[n_obj + i * 12 + 6].dot[0] = dot[3];
		scene[n_obj + i * 12 + 6].dot[1] = dot[2];
		scene[n_obj + i * 12 + 6].dot[2] = dot[7];
		scene[n_obj + i * 12 + 6].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 6]);
		find_factor(scene[n_obj + i * 12 + 6]);

		scene[n_obj + i * 12 + 7].dot[0] = dot[7];
		scene[n_obj + i * 12 + 7].dot[1] = dot[2];
		scene[n_obj + i * 12 + 7].dot[2] = dot[6];
		scene[n_obj + i * 12 + 7].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 7]);
		find_factor(scene[n_obj + i * 12 + 7]);

		scene[n_obj + i * 12 + 8].dot[0] = dot[2];
		scene[n_obj + i * 12 + 8].dot[1] = dot[1];
		scene[n_obj + i * 12 + 8].dot[2] = dot[6];
		scene[n_obj + i * 12 + 8].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 8]);
		find_factor(scene[n_obj + i * 12 + 8]);

		scene[n_obj + i * 12 + 9].dot[0] = dot[6];
		scene[n_obj + i * 12 + 9].dot[1] = dot[1];
		scene[n_obj + i * 12 + 9].dot[2] = dot[5];
		scene[n_obj + i * 12 + 9].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 9]);
		find_factor(scene[n_obj + i * 12 + 9]);

		scene[n_obj + i * 12 + 10].dot[0] = dot[1];
		scene[n_obj + i * 12 + 10].dot[1] = dot[0];
		scene[n_obj + i * 12 + 10].dot[2] = dot[5];
		scene[n_obj + i * 12 + 10].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 10]);
		find_factor(scene[n_obj + i * 12 + 10]);

		scene[n_obj + i * 12 + 11].dot[0] = dot[5];
		scene[n_obj + i * 12 + 11].dot[1] = dot[0];
		scene[n_obj + i * 12 + 11].dot[2] = dot[4];
		scene[n_obj + i * 12 + 11].material = materials[tmp];
		find_normal(scene[n_obj + i * 12 + 11]);
		find_factor(scene[n_obj + i * 12 + 11]);

	    // scene[i].material = materials[tmp];
		// find_normal(scene[i]);
		// find_factor(scene[i]);
	    // cout << scene[i].normal.x << ":" << scene[i].normal.y << ":" << scene[i].normal.z << "\n\n";
	}
	infile.close();
	n_obj = n_obj + box * 12;
	free(materials);
	return scene;
}

__host__ void * alloc_mem_cpu(size_t size) {
	void *ptr = malloc(size);
	if (ptr == NULL) {
		cout << "Allocation error.\n";
		exit(EXIT_FAILURE);
	}
	return ptr;
}

__host__ void mem_cpy_to_gpu(void *h, void *d, size_t size) {
	if (cudaMemcpy(d, h, size, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Host to device copying failed!\n";
		exit(EXIT_FAILURE);
	}
}

__host__ void * alloc_mem_gpu(size_t size) {
	void *ptr; 
	if (cudaMalloc ((void **)&ptr, size) != cudaSuccess) {
		cout << "Host to device copying failed\n";
		exit(EXIT_FAILURE);
	}
	return ptr;
}

__host__ void mem_cpy_to_cpu(void *d, void *h, size_t size) {
	if (cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Device to host copying failed\n";
		exit(EXIT_FAILURE);
	}
}

__host__ void cudaErrors(cudaError_t error) {
	switch(error) {
		case cudaSuccess: cout << "Success!\n"; break;
		case cudaErrorMissingConfiguration: cout << "cudaErrorMissingConfiguration!\n"; break;
		case cudaErrorMemoryAllocation: cout << "cudaErrorMemoryAllocation!\n"; break;
		case cudaErrorInitializationError: cout << "cudaErrorInitializationError!\n"; break;
		case cudaErrorLaunchFailure: cout << "cudaErrorLaunchFailure!\n"; break;
		case cudaErrorPriorLaunchFailure: cout << "cudaErrorPriorLaunchFailure!\n"; break;
		case cudaErrorLaunchTimeout: cout << "cudaErrorLaunchTimeout!\n"; break;
		case cudaErrorLaunchOutOfResources: cout << "cudaErrorLaunchOutOfResources!\n"; break;
		case cudaErrorInvalidDeviceFunction: cout << "cudaErrorInvalidDeviceFunction!\n"; break;
		case cudaErrorInvalidConfiguration: cout << "cudaErrorInvalidConfiguration!\n"; break;
		case cudaErrorInvalidDevice: cout << "cudaErrorInvalidDevice!\n"; break;
		case cudaErrorInvalidValue: cout << "cudaErrorInvalidValue!\n"; break;
		case cudaErrorInvalidPitchValue: cout << "cudaErrorInvalidPitchValue!\n"; break;
		case cudaErrorInvalidSymbol: cout << "cudaErrorInvalidSymbol!\n"; break;
		case cudaErrorMapBufferObjectFailed: cout << "cudaErrorMapBufferObjectFailed!\n"; break;
		case cudaErrorUnmapBufferObjectFailed: cout << "cudaErrorUnmapBufferObjectFailed!\n"; break;
		case cudaErrorInvalidHostPointer: cout << "cudaErrorInvalidHostPointer!\n"; break;
		case cudaErrorInvalidDevicePointer: cout << "cudaErrorInvalidDevicePointer!\n"; break;
		case cudaErrorInvalidTexture: cout << "cudaErrorInvalidTexture!\n"; break;
		case cudaErrorInvalidTextureBinding: cout << "cudaErrorInvalidTextureBinding!\n"; break;
		case cudaErrorInvalidChannelDescriptor: cout << "cudaErrorInvalidChannelDescriptor!\n"; break;
		case cudaErrorInvalidMemcpyDirection: cout << "cudaErrorInvalidMemcpyDirection!\n"; break;
		case cudaErrorAddressOfConstant: cout << "cudaErrorAddressOfConstant!\n"; break;
		case cudaErrorTextureFetchFailed: cout << "cudaErrorTextureFetchFailed!\n"; break;
		case cudaErrorTextureNotBound: cout << "cudaErrorTextureNotBound!\n"; break;
		case cudaErrorSynchronizationError: cout << "cudaErrorSynchronizationError!\n"; break;
		case cudaErrorInvalidFilterSetting: cout << "cudaErrorInvalidFilterSetting!\n"; break;
		case cudaErrorInvalidNormSetting: cout << "cudaErrorInvalidNormSetting!\n"; break;
		case cudaErrorMixedDeviceExecution: cout << "cudaErrorMixedDeviceExecution!\n"; break;
		// case cudaErrorcudartUnloading: cout << "cudaErrorcudartUnloading!\n"; break;
		case cudaErrorUnknown: cout << "cudaErrorUnknown!\n"; break;
		case cudaErrorNotYetImplemented: cout << "cudaErrorNotYetImplemented!\n"; break;
		case cudaErrorMemoryValueTooLarge: cout << "cudaErrorMemoryValueTooLarge!\n"; break;
		case cudaErrorInvalidResourceHandle: cout << "cudaErrorInvalidResourceHandle!\n"; break;
		case cudaErrorNotReady: cout << "cudaErrorNotReady!\n"; break;
		case cudaErrorInsufficientDriver: cout << "cudaErrorInsufficientDriver!\n"; break;
		case cudaErrorSetOnActiveProcess: cout << "cudaErrorSetOnActiveProcess!\n"; break;
		case cudaErrorInvalidSurface: cout << "cudaErrorInvalidSurface!\n"; break;
		case cudaErrorNoDevice: cout << "cudaErrorNoDevice!\n"; break;
		case cudaErrorECCUncorrectable: cout << "cudaErrorECCUncorrectable!\n"; break;
		case cudaErrorSharedObjectSymbolNotFound: cout << "cudaErrorSharedObjectSymbolNotFound!\n"; break;
		case cudaErrorSharedObjectInitFailed: cout << "cudaErrorSharedObjectInitFailed!\n"; break;
		case cudaErrorUnsupportedLimit: cout << "cudaErrorUnsupportedLimit!\n"; break;
		case cudaErrorDuplicateVariableName: cout << "cudaErrorDuplicateVariableName!\n"; break;
		case cudaErrorDuplicateTextureName: cout << "cudaErrorDuplicateTextureName!\n"; break;
		case cudaErrorDuplicateSurfaceName: cout << "cudaErrorDuplicateSurfaceName!\n"; break;
		case cudaErrorDevicesUnavailable: cout << "cudaErrorDevicesUnavailable!\n"; break;
		case cudaErrorInvalidKernelImage: cout << "cudaErrorInvalidKernelImage!\n"; break;
		case cudaErrorNoKernelImageForDevice: cout << "cudaErrorNoKernelImageForDevice!\n"; break;
		case cudaErrorIncompatibleDriverContext: cout << "cudaErrorIncompatibleDriverContext!\n"; break;
		case cudaErrorStartupFailure: cout << "cudaErrorStartupFailure!\n"; break;
		case cudaErrorApiFailureBase: cout << "cudaErrorApiFailureBase!\n"; break;
	}
}
		

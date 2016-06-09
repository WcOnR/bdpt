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

#define WIDTH 256
#define HEIGHT 256
#define ASPECT WIDTH/HEIGHT
#define SAMPLES 1000
#define MAX_DEPTH 128

enum { NELEMS = WIDTH * HEIGHT };

typedef struct Ray {
	float3 direction;
	float3 origin;
}Ray;

typedef struct ray {
	float3 origin;
	float3 direction;
	float3 radiance;
	float terminate;
} ray;

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

typedef struct Polygon {
	float3 dot[3];
	float3 normal;
	float D;
	Material material;
}Polygon;

typedef struct Sphere {
	float3 center;
	float radius;
	Material material;
}Sphere;

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

__device__ Ray shoot_ray(Camera &cam, curandState* gState) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	float w_off = (float)(indx % WIDTH + rand_f(gState)) / (float)(WIDTH);
	float h_off = (float)(indx / WIDTH + rand_f(gState)) / (float)(HEIGHT);
	Ray tmp;
	tmp.origin = cam.origin;
	tmp.direction = normalize(cam.lowleftcorner + w_off * cam.horizontal + h_off * cam.vertical - cam.origin);
	// tmp.direction = cam.lowleftcorner + w_off * cam.horizontal + h_off * cam.vertical - cam.origin;

	return tmp;
}

__device__ float3 get_ray(Ray r, float k) {
	return r.origin + k * r.direction;
}

__device__ float3 reflection(float3 v1, float3 v2) {
	return v1 - 2.0f * dot(v1, v2) * v2;
}

__device__ bool diffusion(curandState* gState, Ray r, Ray &scattered, Hit_record &rec, float3 &attenuated){
	if (length(rec.material.emissivity) > 0) return false;
	float ran = rand_f(gState);

	if (ran < rec.material.reflectivity && rec.material.refractivity == 0.0f) {
		// metal
		float3 refl = normalize(reflection(r.direction, rec.normal));
		scattered = {refl, rec.point};
		attenuated = rec.material.color;
		return (dot(scattered.direction, rec.normal) > 0);
	} else if (rec.material.refractivity == 0.0f) {
		// matte
		float3 v1 = {rand_f(gState), rand_f(gState), rand_f(gState)};
		v1 = normalize(v1);
		if (dot(v1, rec.normal) < 0.0f) v1 = -v1;
		scattered = {v1, rec.point};
		attenuated = rec.material.color;
		return true;
	}

	// if (rec.material.reflectivity == 0.0f && rec.material.refractivity == 0.0f) {
	// 	// matte
	// 	float3 v1 = {rand_f(gState), rand_f(gState), rand_f(gState)};
	// 	v1 = normalize(v1);
	// 	if (dot(v1, rec.normal) < 0.0f) v1 = -v1;
	// 	scattered = {v1, rec.point};
	// 	attenuated = rec.material.color;
	// 	return true;
	// } else if (rec.material.reflectivity > 0.0f && rec.material.refractivity == 0.0f) {
	// 	// metal
	// 	float3 refl = normalize(reflection(r.direction, rec.normal));
	// 	scattered = {refl, rec.point};
	// 	attenuated = rec.material.color;
	// 	return (dot(scattered.direction, rec.normal) > 0);
	// }
	return true;
}

__device__ bool get_intersection(Ray r, Hit_record &rec, float dist, Sphere s) {
	float3 oc = r.origin - s.center;

	float a = dot(r.direction, r.direction);
	float b = 2.0f * dot(oc, r.direction);
	float c = dot(oc, oc) - s.radius * s.radius;
	float D = b * b - 4.0f * a * c;

	if (D > 0) {
		float d1 = (-b - sqrt(D)) / (2.0f * a);
		float d2 = (-b + sqrt(D)) / (2.0f * a);
		float root = (d1 < d2) ? d1 : d2;

		if (root < dist && root > 0.001f) {	
			rec.point = get_ray(r, root);
			rec.normal = (rec.point - s.center) / s.radius;
			rec.depth = root;
			rec.material = s.material;
			
			return true;
		}
	}

	return false;
}

__device__ bool nearest_intersection (Ray r, Sphere *scene, Hit_record &rec, int n_obj) {
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

__device__ float3 raytrace(curandState* gState, Sphere *scene, Ray r, int n_obj) {
	Ray primary = r;
	Hit_record rec;
	float3 attenuated;
	float3 emitted;
	float3 composite = {0.0f, 0.0f, 0.0f};
	float3 counted = {1.0f, 1.0f, 1.0f};

	for (int i = 0; i < MAX_DEPTH; ++i) {
		if (nearest_intersection(primary, scene, rec, n_obj)) {
			// return rec.material.color;
			emitted = rec.material.emissivity;
			Ray scattered;
			if (diffusion(gState, primary, scattered, rec, attenuated)) {
				primary = scattered;
				composite += (emitted + attenuated) * counted;
				counted = counted * attenuated;
			} else {
				return composite + emitted * counted;
			}
		} else {
			return {0, 0, 0};
		}
	}
	return {0, 0, 0};
	// return composite + emitted * counted;
}



//**************KERNEL**************//

__global__ void kernel(curandState* gState, unsigned long seed, Sphere *scene, float3 *result, int n, int n_obj) {
	int indx = blockDim.x * blockIdx.x + threadIdx.x;
	Camera cam;
	create_camera(cam, make_float3(1, 100, 0), make_float3(0, 0, 0), 40.0f);
	
	if (indx < n) {
		curand_init (seed, indx, 0, &gState[indx]);
		Ray primary;
		result[indx] = {0, 0, 0};
		float scaled = 1.0f / (float) SAMPLES;
		for (int i = 0; i < SAMPLES; ++i) {
			primary = shoot_ray(cam, gState);
			result[indx] += raytrace(gState, scene, primary, n_obj) * scaled;
		}
	}
}

//**************CPU**************//


// __host__ inline float gamma_correction(float val) {return val;}
// __host__ inline int gamma_correction(float val) {return val * 256;}
__host__ inline int gamma_correction(float val) { return int(255.0f * sqrt(fmaxf(0.0f, fminf(1.0f, val))));}

__host__ void print_file_header(ofstream &file);
__host__ void find_normal(Polygon &p);
__host__ Sphere * init_spheres(int &n_obj);
// __host__ Polygon * init_polygons(int &n_obj);
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
	Sphere *h_scene = NULL;// = (Sphere *) alloc_mem_cpu(sizes);
	float3 *h_result = (float3 *) alloc_mem_cpu(sizer);

	h_scene = init_spheres(n_obj);
	size_t sizes = sizeof(Sphere) * n_obj;
	/* Allocate vectors on device */
	Sphere *d_scene = NULL;
	float3 *d_result = NULL;
	curandState* d_states = NULL;

	d_scene = (Sphere *)alloc_mem_gpu(sizes);
	d_result = (float3 *)alloc_mem_gpu(sizer);    
	d_states = (curandState*)alloc_mem_gpu(sizec);

	/* Copy the host vectors to device */

	mem_cpy_to_gpu(h_scene, d_scene, sizes);

	int threadsPerBlock = 128;
	int blocksPerGrid =(NELEMS + threadsPerBlock - 1) / threadsPerBlock;

	kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL), d_scene, d_result, NELEMS, n_obj);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << "Failed to launch kernel!\n";
		cudaErrors(error);
		exit(EXIT_FAILURE);
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
	cudaFree(d_scene);
	cudaFree(d_result);
	cudaFree(d_states);
	free(h_result);
	free(h_scene);
	cudaDeviceReset();
	return 0;
}

// __host__ void print_file_header(ofstream &file) {
// 	file << (uint8_t)'B' << (uint8_t)'M' << (uint32_t)WIDTH * HEIGHT * 3 + 54;
// 	file << (uint16_t)0 << (uint16_t)0 << (uint32_t)54 << (uint32_t)40;
// 	file << (uint32_t)WIDTH << (uint32_t)HEIGHT << (uint16_t)1;
// 	file << (uint16_t)24 << (uint32_t)0 << (uint32_t)0 << (uint32_t)0;
// 	file << (uint32_t)0 << (uint32_t)0 << (uint32_t)0;
// }

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

__host__ Sphere * init_spheres(int &n_obj) {
	int m;
	Material *materials = init_materials(m);
	ifstream infile("inSpher");
	infile >> n_obj;

	size_t sizeScene = sizeof(Sphere) * n_obj;
	Sphere *scene = (Sphere *)alloc_mem_cpu(sizeScene);

	for (int i = 0; i < n_obj; ++i) {
		infile >> scene[i].center.x >> scene[i].center.y >> scene[i].center.z;
		infile >> scene[i].radius;
		int tmp;
		infile >> tmp;
		scene[i].material = materials[tmp];
	}
	infile.close();
	free(materials);
	return scene;
}
// __host__ Polygon * init_polygons(int &n_obj) {
// 	int m;
// 	Material *materials = init_materials(m);
// 	ifstream infile("inPolygon");
// 	infile >> n_obj;

// 	size_t sizeScene = sizeof(Polygon) * n_obj;
// 	Polygon *scene = (Polygon *)alloc_mem_cpu(sizeScene);

// 	for (int i = 0; i < n_obj; ++i) {
// 		for (int j = 0; j < 3; ++j)
// 	    	infile >> scene[i].dot[j].x >> scene[i].dot[j].y >> scene[i].dot[j].z;
// 	    int tmp;
// 	    infile >> tmp;
// 	    scene[i].material = materials[tmp];
// 	    find_normal(scene[i]);
// 	    find_factor(scene[i]);
// 	    // cout << scene[i].normal.x << ":" << scene[i].normal.y << ":" << scene[i].normal.z << "\n\n";
// 	}
// 	infile.close();

// 	free(materials);
// 	return NULL;
// }

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
		

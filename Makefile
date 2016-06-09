all:
	/usr/local/cuda-7.5/bin/nvcc main.cu -lm --std=c++11 -o main.out

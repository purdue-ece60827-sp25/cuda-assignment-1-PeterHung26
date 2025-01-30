
#include "cudaLib.cuh"
#include "cpuLib.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < size)
		y[idx] = scale * x[idx] + y[idx];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";
	device_prop();
	int vectorByte = vectorSize * sizeof(float);
	// Set up the thread block
	dim3 DimGrid(ceil(vectorSize/128.0),1,1);
	dim3 DimBlock(128, 1, 1);
	// Memory Allocation and Variable on host (for the verification)
	float * x, * y, * veri,  scale;
	x = (float *) malloc(vectorByte);
	y = (float *) malloc(vectorByte);
	veri = (float *) malloc(vectorByte);
	float * x_d, * y_d;
	cudaMalloc((void **) &x_d, vectorByte);
	cudaMalloc((void **) &y_d, vectorByte);
	scale = 2.0f;
	vectorInit(x, vectorSize);
	vectorInit(y, vectorSize);
	cudaMemcpy(x_d, x, vectorByte, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, vectorByte, cudaMemcpyHostToDevice);
	// Kernel invocation code
	saxpy_gpu<<<DimGrid, DimBlock>>>(x_d, y_d, scale, vectorSize);
	// Check the result with CPU
	cudaMemcpy(veri, y_d, vectorByte, cudaMemcpyDeviceToHost);
	int errorCount = verifyVector(x, y, veri, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
	// Free the host and device memory
	free(x);
	free(y);
	free(veri);
	cudaFree(x_d);
	cudaFree(y_d);
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float x, y;
	if(idx < pSumSize){
		pSums[idx] = 0;
		// Setup RNG
		curandState_t rng;
		curand_init(clock64(), idx, 0, &rng);
		for (uint64_t i = 0; i < sampleSize; ++i) {
			// Generate random points
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);
			// Calculate whether the point is in the unit cycle
			if ( int(x * x + y * y) == 0 ) {
				++ pSums[idx];
			}
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	uint64_t reduceThreadCount = pSumSize/reduceSize;
	if(idx < reduceThreadCount){
		totals[idx] = 0;
		for(uint64_t i = 0; i < reduceSize; i++){
			if((idx*reduceSize + i) < pSumSize)
			totals[idx] = totals[idx] + pSums[idx*reduceSize + i];
		}
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	// Part 1: generatePoints
	// Set up the thread block
	dim3 DimGrid(ceil(generateThreadCount/128.0),1,1);
	dim3 DimBlock(128, 1, 1);
	// Allocate the memory and invoke the kernels
	uint64_t pSumSize = generateThreadCount;
	uint64_t * pSums;
	cudaMalloc((void **) &pSums, sizeof(uint64_t)*pSumSize);
	generatePoints<<<DimGrid, DimBlock>>>(pSums, pSumSize, sampleSize);
	// Part 2: reduceCounts
	// Set up the thread block
	dim3 DimGrid2(ceil(reduceThreadCount/128.0),1,1);
	dim3 DimBlock2(128, 1, 1);
	// Allocate the memory and invoke the kernels
	uint64_t * totals_d;
	uint64_t * totals;
	cudaMalloc((void **) &totals_d, sizeof(uint64_t)*reduceThreadCount);
	totals = (uint64_t *) malloc(sizeof(uint64_t)*reduceThreadCount);
	reduceCounts<<<DimGrid2, DimBlock2>>>(pSums, totals_d, pSumSize, reduceSize);
	// Sum up the total reduced partial sum and calculate the value of pi
	cudaMemcpy(totals, totals_d, sizeof(uint64_t)*reduceThreadCount, cudaMemcpyDeviceToHost);
	uint64_t totalHitCount = 0;
	for(int i = 0; i < reduceThreadCount; i++){
		totalHitCount = totalHitCount + totals[i];
	}
	approxPi = ((double)totalHitCount / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;
	// Free the memory
	free(totals);
	cudaFree(pSums);
	cudaFree(totals_d);
	return approxPi;
}

// Remeber to hide it before submission
int device_prop(){
	cudaDeviceProp prop;
    int device;
    cudaError_t err = cudaGetDevice(&device);
	if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaGetDeviceProperties(&prop, device);

	int maxActiveBlocks;
    int blockSize = 128;  // Example block size (you can vary this)
    size_t sharedMemoryPerBlock = 0; // Default shared memory usage

    // Get maximum active blocks per SM
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, saxpy_gpu, blockSize, sharedMemoryPerBlock
    );

    int totalThreadsPerSM = maxActiveBlocks * blockSize;
    float occupancy = (float)totalThreadsPerSM / prop.maxThreadsPerMultiProcessor * 100.0f;

    printf("Device Name: %s\n", prop.name);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Active Blocks per SM: %d\n", maxActiveBlocks);
    printf("Total Threads per SM: %d\n", totalThreadsPerSM);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Occupancy: %.2f%%\n", occupancy);

    return 0;
}
#include "utils.h"
#include <stdlib.h>

struct results
{
	float sum;
};

#include "summation_kernel.cu"

// CPU implementation
// increasing order
float log2_series(int n)
{
	int i; 
	float res=0.0;
	float tmp=1.0;
	for (i=0; i<n; i++){
		if ((i%2) == 1) {tmp = -1.0;}
		else tmp = 1.0;

		res += tmp / (i + 1);
		}
	return res;
}
// decreasing order
float log2_series_dec(int n)
{
	int i; 
	float res=0.0;
	float tmp=1.0;
	for (i=n; i>=0; i--){
		if ((i%2) == 1) {tmp = -1.0;}
		else tmp = 1.0;

		res += tmp / (i + 1);
		}
	return res;
}

int main(int argc, char ** argv)
{
    int data_size = 1024 * 1024 * 128;

    // Run CPU version
    double start_time = getclock();
    float log2 = log2_series(data_size);
    //float log2_dec = log2_series_dec(data_size);
    double end_time = getclock();
    
    printf("CPU result: %.15f\n", log2);
    //printf("CPU result: %.15f (dec)\n", log2_dec);
    printf(" log(2)=    %f\n", log(2.0));
    printf(" time=%fs\n", end_time - start_time);
    
    // Parameter definition
    int threads_per_block = 4 * 32;
    int blocks_in_grid = 8;
    
    int num_threads = threads_per_block * blocks_in_grid;

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    int results_size = num_threads;
    float * data_out_cpu;
	float resultat_final;

    // Allocating output data on CPU
	// Version calcul CPU if ((data_out_cpu = (float*) malloc(num_threads*sizeof(float))) == NULL) {printf("erreur allocation CPU"); exit(0);}
	if ((data_out_cpu = (float*) malloc(blocks_in_grid*sizeof(float))) == NULL) {printf("erreur allocation CPU"); exit(0);}
	
	// Allocating output data on GPU
	// int i;
	float* resGPU;
	float* resFinalGPU;
	if (cudaMalloc((void**)&resGPU, sizeof(float)*num_threads) != cudaSuccess){printf("erreur allocation GPU"); exit(0);}
	if (cudaMalloc((void**)&resFinalGPU, sizeof(float)) != cudaSuccess){printf("erreur allocation GPU"); exit(0);}

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Execute kernel
    summation_kernel<<<blocks_in_grid, threads_per_block, threads_per_block*sizeof(float)>>>(data_size/num_threads, resGPU);

	// Fusion finale
	second_summation_kernel<<<1, blocks_in_grid>>>(resGPU, resFinalGPU);

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get results back
	// Version calcul CPU if (cudaMemcpy(data_out_cpu, resGPU, sizeof(float)*num_threads, cudaMemcpyDeviceToHost) != cudaSuccess) {printf("erreur recopie resultat vers CPU"); exit(0);}
	//if (cudaMemcpy(data_out_cpu, resGPU, sizeof(float)*blocks_in_grid, cudaMemcpyDeviceToHost) != cudaSuccess) {printf("erreur recopie resultat vers CPU"); exit(0);}
	if (cudaMemcpy(&resultat_final, resFinalGPU, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {printf("erreur recopie resultat vers CPU"); exit(0);}
	
	// Finish reduction
    /*
	float sum = 0.;
	// Version calcul CPU
    for (i=num_threads-1; i>=0; i--){
		sum += data_out_cpu[i];
	}
	
	for (i=blocks_in_grid-1; i>=0; i--){
		sum += data_out_cpu[i];
	}
	*/

    // Cleanup
    cudaFree(resGPU);
	cudaFree(resFinalGPU);
	free(data_out_cpu);
    
    printf("GPU results:\n");
    //printf(" Sum: %f\n", sum);
    printf(" Sum: %f\n", resultat_final);

    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double)data_size;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);
  
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    return 0;
}


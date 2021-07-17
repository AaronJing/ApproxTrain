#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <error.cuh>
#include <gemm.cuh>


//Dimensions for matrix1. These should be a multiple of BLOCK
#define ROWS1 800
#define COLS1 1600

//DImensions for matrix2. These should be a multiple of BLOCK
#define ROWS2 1600
#define COLS2 800

#define BLOCK 16


int main(){

	//check whether dimensions are valid for matrix mutiplication
	if(COLS1!=ROWS2){
		printf("Matrix dimensions are invalid for matrix multiplication\n");
		exit(1);
	}
	//check whether the requirements for the current version of the program is met
	if(COLS1%BLOCK!=0 || COLS2%BLOCK!=0 || ROWS1%BLOCK!=0 || ROWS2%BLOCK!=0){
		fprintf(stderr, "This program need the COLS1 COLS2 ROWS1 and ROWS2 to be multiples of BLOCK\n");
		exit(1);
	}

	//Initialize arrays in RAM
	float *matA = (float *)malloc(sizeof(float)*ROWS1*COLS1);
	float *matB = (float *)malloc(sizeof(float)*ROWS2*COLS2);
	float *matC = (float *)malloc(sizeof(float)*ROWS1*COLS2);
	//check if out of memory.
	if(matA==NULL || matB==NULL || matC==NULL){
		perror("Memory out");
		exit(EXIT_FAILURE);
	}

	//generate some values for matrixA
	int i,j;
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			matA[i*COLS1+j]=i+j;
		}
	}

	// //print the matA
	// printf("Matrix A : \n");
	// for(i=0;i<ROWS1;i++){
		// for(j=0;j<COLS1;j++){
			// printf("%5f ",matA[i*COLS1+j]);
		// }
		// printf("\n");
	// }
	// printf("\n");


	//generate values for matrixB
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			matB[i*COLS2+j]=i-j;
		}
	}

	// //print the matB
	// printf("Matrix B : \n");
	// for(i=0;i<ROWS2;i++){
		// for(j=0;j<COLS2;j++){
			// printf("%5f ",matB[i*COLS2+j]);
		// }
		// printf("\n");
	// }
	// printf("\n");

	/********************************** CUDA stuff starts here *******************************/

	//start meauring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	//pointers for memory allocation in cudaa
	float *matA_cuda;
	float *matB_cuda;
	float *matC_cuda;

	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(float)*ROWS1*COLS1); checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(float)*ROWS2*COLS2); checkCudaError();
	cudaMalloc((void **)&matC_cuda,sizeof(float)*ROWS1*COLS2); checkCudaError();

	//copy memory from ram to cuda
	cudaMemcpy(matA_cuda,matA,sizeof(float)*ROWS1*COLS1,cudaMemcpyHostToDevice); checkCudaError();
	cudaMemcpy(matB_cuda,matB,sizeof(float)*ROWS2*COLS2,cudaMemcpyHostToDevice); checkCudaError();

	//multiply the matrices
	dim3 threadsPerBlock(BLOCK,BLOCK);
	dim3 numBlocks(ceil(COLS2/(float)BLOCK),ceil(ROWS1/(float)BLOCK));

	//start measuring time for cuda kernel only
	cudaEvent_t startkernel,stopkernel;
	float elapsedtimekernel;
	cudaEventCreate(&startkernel);
	cudaEventRecord(startkernel,0);
	// __global__ void gemm(size_t m, size_t n, size_t k,
    // const float *a, size_t lda, const float *b, size_t ldb,
    // float *c, size_t ldc)

// 	//Dimensions for matrix1. These should be a multiple of BLOCK
// #define ROWS1 800
// #define COLS1 1600

// //DImensions for matrix2. These should be a multiple of BLOCK
// #define ROWS2 1600
// #define COLS2 800

	gemm<<<numBlocks,threadsPerBlock>>>(ROWS1,COLS2,COLS1,matA_cuda, COLS1,matB_cuda, COLS2,matC_cuda,COLS2);
	cudaDeviceSynchronize(); checkCudaError();

	//end measuring time for cuda kernel
	cudaEventCreate(&stopkernel);
	cudaEventRecord(stopkernel,0);
	cudaEventSynchronize(stopkernel);
	cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);

	//copy the answer back from cuda ro ram
	cudaMemcpy(matC,matC_cuda,sizeof(float)*ROWS1*COLS2,cudaMemcpyDeviceToHost); checkCudaError();

	//free the cuda memory
	cudaFree(matA_cuda); checkCudaError();
	cudaFree(matB_cuda); checkCudaError();
	cudaFree(matC_cuda); checkCudaError();

	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);

	/********************** CUDA stuff ends here ********************************/

	// //print the answer
	printf("Answer : \n");
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS2;j++){
			printf("%5f ",matC[i*COLS2+j]);
		}
		printf("\n");
	}

	//print the time spent to stderr
	fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel/(float)1000);
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000);

	return 0;
}

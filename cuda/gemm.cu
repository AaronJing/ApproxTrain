#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#ifndef ERROR_CUH
#define ERROR_CUH

#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

static inline void gpuAssert(const char *file, int line){
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line);
        exit(1);
   }
}

#endif

//Dimensions for matrix1. These should be a multiple of BLOCK
#define ROWS1 800
#define COLS1 1600

//DImensions for matrix2. These should be a multiple of BLOCK
#define ROWS2 1600
#define COLS2 800

#define BLOCK 16


#define MANTISSA_MASK     8388607
#define EXPONENT_MASK     2139095040
#define MANTISSA_MASK_INV 4286578688  
#define SIGN_MASK         2147483648

// MANTISSA_MASK: 2^23 - 1    :     {9{0}, 23{1}}
// MANTISSA_MASK_INV: 511*2^23 :    {9{1}, 23{0}}
// EXPONENT_MASK:    255^2^23  : {0, 8{1}, 23{0}} 
// SIGN MASK             2^31 :  {1, 31{0}}


//********************************************************************************************************************************************
__device__ uint32_t fn_mul_mbmmant32(uint32_t Amnt, uint32_t Bmnt)
{
	// Approximate multiplication of mantissa 
	uint32_t Smnt;
	uint32_t Smnt_corr;
	uint32_t Smnt_corr_whbit;
	uint16_t carry;
	uint32_t CorrTem = 655360;    // constant:  (2^-4+2^-6 x 2^23): in fixed point 9_23 

	Smnt      = Amnt + Bmnt;
	carry     = (Smnt >= 8388608);     //2^23 = 8388608 <==>  x1+x2>1;
	Smnt_corr = (Smnt & MANTISSA_MASK) + (CorrTem >> carry);

	if (carry && ((Smnt_corr) >= 8388608))  // 1.0*2^23 (because one hidden bit is removed)  // cornercase: overflow: skip error correction term
	{	
		Smnt_corr = Smnt;
	}
	// Append hidden bit

	Smnt_corr_whbit = ((8388608 + Smnt_corr) << carry);  //  2^23 = 8388608: equivalent to adding  2^23 or 2^24???

	return Smnt_corr_whbit;	// unnormalized
}




//********************************************************************************************************************************************
__device__ float FPmultMBM_fast32(float Af, float Bf)
{
    // type casting
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;


	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);		
	uint32_t  Bmnt = (MANTISSA_MASK & bt);

    // Approximate Mantissa calculation
	uint32_t Mbm_mantmult = fn_mul_mbmmant32(Amnt, Bmnt);  // Passing without hidden bit

	// Accurate mantissa calculation
	uint64_t Acc_mantmul = ((uint64_t)(8388608 + Amnt)) * ((uint64_t)(8388608 + Bmnt));  // appending hidden bit : 2^23

	//--------------------------------------------------------------------------
	// Accurate FP multiplication
	float     Ofacc =  Af*Bf;
	uint32_t  Otacc = *(uint32_t *)&Ofacc;
	
	//Extracting sign, exponent, mantissa 
	uint32_t  Oaccsgn = (Otacc & SIGN_MASK);            // 2^31 :  {1, 31{0}}
	uint32_t  Oaccexp = (Otacc & EXPONENT_MASK) >> 23;

	//--------------------------------------------------------------------------
	float Oft;
	uint32_t Onewexp = Oaccexp;
	bool is_accurate_normalized;
	bool is_approx_normalized;
	
	is_approx_normalized   = (Mbm_mantmult >= 16777216);        // 2^24 {1, 24{0}} because this comes out to be 25-bit: This means approximate mantissa needs to be normalized. 

	is_accurate_normalized = (Acc_mantmul  >= 140737488355328); // 2^47 {1, 47{0}} because this comes out to be 48-bit: This means accurate mantissa was normalized.  Rounding effect is not handled yet
    if(is_accurate_normalized==0){                              // check if normalization happens due to rounding (note, I am assuming rounding-normalization will never happen if original normalization has happened)
        uint32_t Acc_mant_kept;             // 1.24 fxp
        uint32_t Acc_mant_drop;             
        uint32_t Acc_mant_rounded;          // 2.23 fxp
    
        bool rb, M0, R, S;
        
        Acc_mant_kept = Acc_mantmul >> 22;                 // Acc_mantmul has 47 bits (because normalization not needed). chop it down to 25 bits.
        Acc_mant_drop = Acc_mantmul % 4194304;             // 4194304== 2^22
        M0 = ((Acc_mant_kept & 2) != 0);	               // last remaining bit     __x.
		R =  ((Acc_mant_kept & 1) != 0);                   // first dropped bit      ___.x
		S = (Acc_mant_drop != 0);                                // OR of all remaining bit___._xxxxxxxx
		rb = (R & (M0 | S));

        Acc_mant_rounded = (Acc_mant_kept >> 1) + rb;

        is_accurate_normalized = (Acc_mant_rounded >= 16777216);  // 2^24 {1, 24{0}} because this comes out to be 25-bit(2.23)
    }

	// Adjustment of exponent
	if(Oaccexp == 255){ 		// case: inf or NAN
		Oft = Ofacc;  		//do nothing: pass out the special case
	}
	else if (Oaccexp == 0){ // case: subnormal or 0 
		Oft = 0;            // make the answer 0 because we do not handle subnormals
	}
	else{					// case: non-special case
		if(is_accurate_normalized && !is_approx_normalized){
			Onewexp = Oaccexp - 1;
		}

		if(!is_accurate_normalized && is_approx_normalized){
			Onewexp = Oaccexp + 1;
		}

		if(is_approx_normalized){
			Mbm_mantmult = Mbm_mantmult >> 1;  // normalization without rounding
		}

		Mbm_mantmult = Mbm_mantmult & MANTISSA_MASK; // getting rid of hidden bit

    	uint32_t Os = Oaccsgn;
	    uint32_t Oe = (Onewexp << 23);
	    uint32_t Om = Mbm_mantmult;

        uint32_t Oi = Os + Oe + Om;

	    Oft = *(float*)&Oi;
	}
	return Oft;
}


#define TILE_DIM 16
__global__ void gemm(size_t m, size_t n, size_t k,
    const float *a, size_t lda, const float *b, size_t ldb,
    float *c, size_t ldc)
{
    float value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

        if (i*TILE_DIM + threadIdx.x < k && Row < m){
            As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = float(0);
        }

        if (i*TILE_DIM + threadIdx.y < k && Col < n){
            Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = float(0);
        }

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n){
            //value += bitmasking(bitmasking(As[threadIdx.y][n])*bitmasking(Bs[n][threadIdx.x]));
            value += FPmultMBM_fast32(As[threadIdx.y][n],Bs[n][threadIdx.x]);
            //value += FPMult_SinglePrecision_Rnone_Mitchell(As[threadIdx.y][n],Bs[n][threadIdx.x],MT);
            //value += As[threadIdx.y][n]*Bs[n][threadIdx.x];
        }
            
        __syncthreads();
    }

    if (Row < m && Col < n) { 
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}

// int main(){
// 	const size_t m = 10;
// 	const size_t n = 12;
// 	const size_t k = 13;
// 	const lda = k;
// 	const ldb = n;
// 	const ldb = n;

// 	for(int i = 0; i < )	
// }
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

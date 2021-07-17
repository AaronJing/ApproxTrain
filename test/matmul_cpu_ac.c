/* Matrix multiplication program for CPU
This program generates a matrix of defined size and multiplies them
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//Dimensions for matrix1. These should be a multiple of BLOCK
#define ROWS1 800
#define COLS1 1600

//DImensions for matrix2. These should be a multiple of BLOCK
#define ROWS2 1600
#define COLS2 800

/* Function to do matrix multiplication */
void matMul(float matC[ROWS1][COLS2], float matA[ROWS1][COLS1], float matB[ROWS2][COLS2]){

	int row,col,k;
	for(row=0;row<ROWS1;row++){
		for(col=0;col<COLS2;col++){
			float prod=0;
			for(k=0;k<COLS1;k++){
				prod=prod+matA[row][k]*matB[k][col];
			}
			matC[row][col]=prod;
		}
	}

}

//Initialize arrays in RAM
float matA[ROWS1][COLS1];
float matB[ROWS2][COLS2];
float matC[ROWS1][COLS2];

int main(){

	//check whether dimensions are valid for a multiplication
	if(COLS1!=ROWS2){
		printf("Matrix dimensions are invalid for matrix multiplication\n");
		exit(1);
	}



	//generate some values for matrixA
	int i,j;
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			matA[i][j]=i+j;
		}
	}

	//print the matA
	// printf("Matrix A : \n");
	// for(i=0;i<ROWS1;i++){
		// for(j=0;j<COLS1;j++){
			// printf("%5d ",matA[i][j]);
		// }
		// printf("\n");
	// }
	// printf("\n");

	//generate values for matrixB
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			matB[i][j]=i-j;
		}
	}

	//print the matB
	// printf("Matrix B : \n");
	// for(i=0;i<ROWS2;i++){
		// for(j=0;j<COLS2;j++){
			// printf("%5d ",matB[i][j]);
		// }
		// printf("\n");
	// }
	// printf("\n");

	clock_t start = clock();
	//multiply the matrices
	matMul(matC,matA,matB);
	clock_t stop = clock();

	//print the answer
	printf("Answer : \n");
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS2;j++){
			printf("%5f ",matC[i][j]);
		}
		printf("\n");
	}

	//calculate the time taken and print to stderr
	double elapsedtime = (stop-start)/(double)CLOCKS_PER_SEC;
	fprintf(stderr,"Elapsed time for operation on CPU is %1.5f seconds \n",elapsedtime);

	return 0;

}

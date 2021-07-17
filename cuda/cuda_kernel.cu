#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include <fstream>
#include <assert.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <sys/time.h>
using namespace std;
#define THREADS_PER_BLOCK 1024
#define BLOCK_SIZE 1024
// #define MNT_MASK    0x7fffff
// #define EXP_MASK    0x7f800000
// #define SGN_MASK    0x80000000
// #define MAX_EXP     0xff
// #define BIAS        0x7f
#define PROFILE 1

#define T_SIZE 16
///************
#define MT 1

#define RD_NEAREST  0
#define RD_TOWZERO  1
#define RD_TOWPINF  2
#define RD_TOWNINF  3

__device__ unsigned long long int fn_MitchellMul_Optimized(unsigned int A, unsigned int B, unsigned int Sz)
{
	const unsigned int MNTSZ = 23;
	const unsigned int MNTSZ1 = 24;
	//-----------------------------------------------------------------
	unsigned long      int N = Sz;
	unsigned long long int one = 1;


	unsigned long long int charA = Sz-1;   // "8 -" is required because indexing in matlab is reverse than in the paper
	unsigned long long int charB = Sz-1;

	// May be different than depending on how I handle the fractional mantissa :: scaled or not scaled
	double mantA = (((double)A) / (one << charA) - 1);         // (10 Novemeber 2017) matissa is in fraction: I assume that the 52 bits in double precision are enough to retain the bits of the mantissa (which is integer in actual hardware implementation)
	double mantB = (((double)B) / (one << charB) - 1);


	// Get the log value in binary and in decimal respectively
	double lgAd = charA + mantA;
	double lgBd = charB + mantB;

	//--------------------------------------------------------------------------
	// Truncation
	unsigned long long int lgAv = (one << N) * (lgAd);  // (10 Novemeber 2017) Thefractional parts has given N Bits so that None of the bits from original number are lost
	unsigned long long int lgBv = (one << N) * (lgBd);  // (10 Novemeber 2017) Thefractional parts has given N Bits so that None of the bits from original number are lost

	//--------------------------------------------------------------------------

	// Adding the log of A and B
	unsigned long long int sumAB = lgAv + lgBv;
	// Converting it to binary
	double sumd = (double)sumAB / (one << N);

	// Extracting characteristic and mantissa part
	double charR = floor(sumd);
	double mantR = sumd - charR;

	double NormalizedAnswer = 1 + mantR;

	unsigned long long int Result;



	Result = (unsigned long long int) floor((NormalizedAnswer)* (one << (unsigned long long int)charR));


	if (A == 0 || B == 0)
		Result = 0;

	return Result;

}

__device__ unsigned long long int fn_mulN_Mitchell(unsigned int a, unsigned int b, unsigned int k)
{

	const unsigned int MNTSZ = 23;
	const unsigned int MNTSZ1 = 24;

//--------------------------------------------------------------------------
	unsigned long long int tA = MNTSZ;
	unsigned long long int tB = MNTSZ;

	unsigned long long int shamtA;
	unsigned long long int shamtB;

	unsigned long long int one = 1;

		shamtA = 24 - k;
		shamtB = 24 - k;

	unsigned long long int UA;
	unsigned long long int UB;
	unsigned long long int outL;

	UA = (a / (one << shamtA)) ;  // in integer form, UA plus the leading 1
	UB = (b / (one << shamtB)) ;

	outL = fn_MitchellMul_Optimized(UA , UB, k);

	return outL* (one << (shamtA  + shamtB ));

}




// Assumes inputs and outputs are in integer format
__device__ float FPMult_SinglePrecision_Rnone_Mitchell(float A, float B, unsigned long int k)
{
	unsigned long int round = RD_TOWZERO;
	// Reinterpretating the Value as an integer
	unsigned long int ai = *(unsigned int*)&A;
	unsigned long int bi = *(unsigned int*)&B;
	//------------------------------------------------------------------------------------------------------------------------
	// Given Parameters
	//------------------------------------------------------------------------------------------------------------------------
	const unsigned int EXPSZ = 8;
	const unsigned int MNTSZ = 23;
	//------------------------------------------------------------------------------------------------------------------------
	// Calcuated Parameters
	//------------------------------------------------------------------------------------------------------------------------
	const unsigned int P = MNTSZ + 1;   //% mantissa size with hidden bit

	const unsigned int MNT_MASK = (1 << MNTSZ) - 1;
	const unsigned int EXP_MASK = ((1 << EXPSZ) - 1) << MNTSZ;
	const unsigned int SGN_MASK = 1 << (EXPSZ + MNTSZ);

	const   signed int MAXEXP = (1 << EXPSZ) - 1;
	const   signed int BIAS = (1 << (EXPSZ - 1)) - 1;

	const		float REALMIN = pow(2, 1 - BIAS);

/*	//------------------------------------------------------------------------------------------------------------------------
	// Input Numbers and Accurate Output Calculation
	//------------------------------------------------------------------------------------------------------------------------
	float aF = -1e-7;
	float bF = 1-20;

	// logic for forcing subnormals to zero
	if (abs(aF) < REALMIN)
		aF = 0;
	if (abs(bF) < REALMIN)
		bF = 0;


	float oF = aF*bF;

	// Reinterpretating the Value as an integer
	unsigned long int ai = *(unsigned int*)&aF;
	unsigned long int bi = *(unsigned int*)&bF;


	*/
	//======================================================================================================================
	// FP Emulation STARTS HERE
	//======================================================================================================================
	

	// Separate the Sign bit, exponent and mantissa
	bool Asgn = (SGN_MASK & ai) != 0;
	bool Bsgn = (SGN_MASK & bi) != 0;

	signed long int Aexp = (EXP_MASK & ai) >> MNTSZ;
	signed long int Bexp = (EXP_MASK & bi) >> MNTSZ;

	unsigned long int Amnt = (MNT_MASK & ai);		// The hidden bit is not appended yet. In MATLAB or Verilog code, the hidden bit is appended at this point
	unsigned long int Bmnt = (MNT_MASK & bi);

	//--------------------------------------------------------------------------

	//----Exceptions Handling----
	// Flags for input and output flag decisions based on input
	bool	Azero = (Aexp == 0);
	bool	Bzero = (Bexp == 0);
	bool	Inzero = Azero | Bzero;

	bool	Ainf = (Aexp == MAXEXP);
	bool	Binf = (Bexp == MAXEXP);
	bool	Ininf = Ainf | Binf;          // will be high for both Nan or Inf

	// excp and checking value of msb of mantissa(if that is 1, it is a Nan otherwise Inf)
	bool	Anan = Ainf & (Amnt  != 0);      // Checking the MSB of the mantissa. If it is one, then its a NaN. I could also check here if Amnt> (2^(MNT-1))
	bool	Bnan = Binf & (Bmnt  != 0);

	bool	Innan = Anan | Bnan;

	//------------------------------
	// Initializing Output Flags
	bool	flag_zero0 = 0, flag_zero1 = 0, flag_zero3 = 0;
	bool	flag_nan0 = 0;
	bool	flag_inf0 = 0, flag_inf1 = 0, flag_inf2 = 0, flag_inf3 = 0;


	flag_zero0 = Inzero & !Ininf;            // it may be high even if the output should be nan.so nan flag should be given most prioriy
	flag_nan0 = (Inzero &  Ininf) || Innan;  // The " OR Innan" part will be covered in Ininf because;
	flag_inf0 = !Inzero &  Ininf;            // it may be high even if the output should be nan.so nan flag should be given most prioriy


	//--------------------------------------------------------------------------
	// Sign Calculation
	//--------------------------------------------------------------------------
	bool Osgn = Asgn^Bsgn;

	//--------------------------------------------------------------------------
	// Exponent Calculation
	// --------------------------------------------------------------------------
	signed long int	Oexp1 = (Aexp + Bexp - BIAS);

	flag_zero1 = (Oexp1 < 0);
	flag_inf1 = (Oexp1 >= MAXEXP);


	//**************************************************************************
	// Mantissa Calculation (Main computation of the Code)
	//**************************************************************************
	unsigned long long int	Mult = fn_mulN_Mitchell(((1 << MNTSZ) + Amnt), ((1 << MNTSZ) + Bmnt), k);  // not sure if it will get the right number of bits or it will get truncation
	unsigned long long int	Normd;
	signed	      long int	Oexp2;
	//-------------------------------------------
	//normalization after mantissa multiplication
	//-------------------------------------------
	if (Mult >> (2 * P - 1))
	{
		Normd = Mult;
		Oexp2 = Oexp1 + 1; //%% --IMPORTANT -- NEED to check Exception here %
	}
	else
	{
		Normd = Mult << 1;
		Oexp2 = Oexp1;
	}

	
	flag_inf2 = (Oexp2 == MAXEXP);

	//------------------------------
	// Rounding (NO rounding)
	// -----------------------------
	//rb = 0;
	//unsigned long long int Rounded = (Normd >> P) + rb;		// Truncate 24 lower bits from the multiplier result and add rounding bit
	unsigned long long int Rounded = (Normd >> P);		// Truncate 24 lower bits from the multiplier result and add rounding bit

	//-------------------------------------------
	// normalization after rounding (NOT NEEDED)
	//-------------------------------------------
	unsigned long long int Rounded2;
	signed	      long int	Oexp3;
	// Check if there is carry from rounding and adjust
	/*if (Rounded >> P)						// check if 25th bit is one
	{
	Rounded2 = Rounded >> 1;
	Oexp3 = Oexp2 + 1;					//--IMPORTANT -- NEED to check Exception here %
	}
	else
	{*/
	Rounded2 = Rounded;
	Oexp3 = Oexp2;
	//}

	flag_inf3 = (Oexp3 == MAXEXP);
	flag_zero3 = (Oexp3 == 0);
	//**************************************************************************
	//  Putting together the Output
	//**************************************************************************
	bool	flag_zero = flag_zero0 | flag_zero1 | flag_zero3;
	bool	flag_nan = flag_nan0;
	bool	flag_inf = flag_inf0 | flag_inf1 | flag_inf2 | flag_inf3;

	unsigned long int Omnt;
	signed long int Oexp4;

	unsigned int ResultCase = 4 * (flag_nan)+2 * (flag_inf)+(flag_zero);

	switch (ResultCase)
	{
	case 0:					// Normal case
		Omnt = Rounded2;
		Oexp4 = Oexp3;
		break;
	case 1:					// Zero
		Oexp4 = 0;
		Omnt = 0;
		break;
	case 2:					// Infinity
		Oexp4 = MAXEXP;
		Omnt = 0;
		break;
	case 4:					// NaN
		Oexp4 = MAXEXP;
		Omnt = 1 << (MNTSZ - 1);
		break;
	default:				// For every other case, set output as Nan(it depend on our choice how we want to handle the error)
		Oexp4 = MAXEXP;
		Omnt = 1 << (MNTSZ - 1);
		break;
	}


	unsigned long int Os = (((unsigned int)(Osgn & ~flag_nan)) << (EXPSZ + MNTSZ));
	unsigned long int Oe = (Oexp4 << MNTSZ);
	unsigned long int Om = Omnt & MNT_MASK;

	unsigned long int Oi = Os + Oe + Om;


	float O = *(float*)&Oi;


	return O;
}
///***************

static inline double realtime(void) {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return tp.tv_sec + tp.tv_usec * 1e-6;
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    //   if (abort) exit(code);
   }
}
__device__ float halfmul(const float a, const float b){


      half A = __float2half(a);
      half B = __float2half(b);
      half C;
  #if __CUDA_ARCH__ >= 530
      C = __hmul(A, B);
  #else
      C = __float2half(__half2float(A)*__half2float(B));
  #endif
      float c = __half2float(C);
      
    return c;
}
__device__ float bitmasking(float num){
	// int mask = 0xffff0000;
	// //convert to int
	// int b = *(int*)&num;
    // int masked = b&mask;
    // float ret  = *(float*)&masked;
	// return ret;
    return num;
}
__device__ unsigned long long  int fn_MitchellMul_Optimized_Unbiased_LowerBitsReduced_copy(unsigned int A, unsigned int B, unsigned int Sz)
{
	const unsigned int MNTSZ = 23;
	const unsigned int MNTSZ1 = 24;
	//-----------------------------------------------------------------
	unsigned long      int N = Sz;
	unsigned long long int one = 1;


	unsigned long long int charA = Sz-1;   // "8 -" is required because indexing in matlab is reverse than in the paper
	unsigned long long int charB = Sz-1;

	// May be different than depending on how I handle the fractional mantissa :: scaled or not scaled
	double mantA = (((double)A) / (one << charA) - 1);         // (10 Novemeber 2017) matissa is in fraction: I assume that the 52 bits in double precision are enough to retain the bits of the mantissa (which is integer in actual hardware implementation)
	double mantB = (((double)B) / (one << charB) - 1);


	// Get the log value in binary and in decimal respectively
	double lgAd = charA + mantA;
	double lgBd = charB + mantB;

	//--------------------------------------------------------------------------
	// Truncation
	unsigned long long int lgAv = (one << N) * (lgAd);  // (10 Novemeber 2017) Thefractional parts has given N Bits so that None of the bits from original number are lost
	unsigned long long int lgBv = (one << N) * (lgBd);  // (10 Novemeber 2017) Thefractional parts has given N Bits so that None of the bits from original number are lost


	//--------------------------------------------------------------------------

	// Adding the log of A and B
	unsigned  long long int sumAB = lgAv + lgBv;

	// Converting it to binary
	double sumd = (double)sumAB / (one << N);

	// Extracting characteristic and mantissa part
	unsigned  long long int charR = floor(sumd);
	double mantR = sumd - charR;



	double CorrTem = (1.0 / 16 + 1.0 / 64) / (one << (charR - (charA + charB))); //2^-4+2^-6; // ==0.0781;
	//------------------------
	//Edit for lower bit reduction
	unsigned int chopbits = (Sz - 1);
	CorrTem = floor((one << chopbits)*(CorrTem)) / (one << chopbits);
	//----------------------

	double NormalizedAnswer = 1.0 + mantR + CorrTem;  // Here, since the answers are in fraction, the Corr Terms precision wont be lost. (Which is important to keep the peak error low).
	// However, when charr is lessthan or equal to 6, these bits are lost in the
	// final shifting. Therefore a corner case needs to be added

	//-----------------------------------------------------------------
	if ((charR == (2 * N - 1)) && (NormalizedAnswer >= 2))
		// cornercase: overflow
		NormalizedAnswer = 1 + mantR;
	//-----------------------------------------------------------------


	unsigned  long long  int Result;

	Result = (unsigned long long  int)floor((NormalizedAnswer)*(one << charR));

	//-----------------------------------------------------------------
	if ((charR <= 6) && (charR - (charA + charB) == 1) && Sz >= 8)
		// cornercase: peak error 
		Result = Result + 1;
	//-----------------------------------------------------------------

	if (A == 0 || B == 0)
		Result = 0;


	return Result;

}



__device__ unsigned long long int fn_mulN_UREMrd(unsigned int a, unsigned int b, unsigned int k)
{

	const unsigned int MNTSZ = 23;
	const unsigned int MNTSZ1 = 24;

//--------------------------------------------------------------------------
	unsigned long long int tA = MNTSZ;
	unsigned long long int tB = MNTSZ;

	unsigned long long int shamtA;
	unsigned long long int shamtB;

	unsigned long int lA = 0;
	unsigned long int lB = 0;
	unsigned long long int one = 1;

	if (MNTSZ1 > k)
	{
		shamtA = MNTSZ - k + 2;
		lA = 1;
	}
	else
		shamtA = 0;

	if (MNTSZ1 > k)
	{
		shamtB = MNTSZ - k + 2;
		lB = 1;
	}
	else
		shamtB = (0);


	unsigned long long int UA;
	unsigned long long int UB;
	unsigned long long int outL;

	UA = ((a >> shamtA)  << lA) + lA;  // in integer form, UA plus the leading 1
	UB = ((b >> shamtB)  << lB) + lB;

	outL = fn_MitchellMul_Optimized_Unbiased_LowerBitsReduced_copy(UA,UB,k);

	return (outL  << ((shamtA - lA) + (shamtB - lB)) );

}




/* The gateway function */
__device__ float FPmultMBM_cppv2(float Af, float Bf, int t)
{
	//=============================================================================
    // 1ST FUNCTION BODY
    //=============================================================================
    unsigned long int ai = *(unsigned int*)&Af;
	unsigned long int bi = *(unsigned int*)&Bf;
	
	//unsigned long int bitmask =  ~((1 << t) - 1);   //==> 2^t-1;
	
	unsigned long int at = ai; // &bitmask; let the integer multiplier handle the truncation. In old work, I was truncating both input and output. but not now. So this is not needed
	unsigned long int bt = bi; // &bitmask;

	//========================================== MITCHELLs MULTIPLICATION FUNCTION (EQUIVALENT TO MATLAB IS HERE)======================================

	//------------------------------------------------------------------------------------------------------------------------
	// Given Parameters
	//------------------------------------------------------------------------------------------------------------------------
	const unsigned int EXPSZ = 8;
	const unsigned int MNTSZ = 23;
	//------------------------------------------------------------------------------------------------------------------------
	// Calcuated Parameters
	//------------------------------------------------------------------------------------------------------------------------
	const unsigned int P = MNTSZ + 1;   //% mantissa size with hidden bit
	int k = P - t;				// The input parameter t is comming as the number of bits truncated.
								// my new implementation of DRUM and MBM takes k as the parameter which means the number of bits retained


	const unsigned int MNT_POS = (1 << MNTSZ);
	const unsigned int MNT_MASK = MNT_POS - 1;

	const unsigned int EXP_MASK = ((1 << EXPSZ) - 1) << MNTSZ;
	const unsigned int SGN_MASK = 1 << (EXPSZ + MNTSZ);

	const   signed int MAXEXP = (1 << EXPSZ) - 1;
	const   signed int BIAS = (1 << (EXPSZ - 1)) - 1;

	//const	float REALMIN = pow(2, 1 - BIAS);

	////********************************************************************************************************************************************
	//======================================================================================================================
	// FP Emulation STARTS HERE
	//======================================================================================================================
	// Separate the Sign bit, exponent and mantissa
	bool Asgn = (SGN_MASK & at) != 0;
	bool Bsgn = (SGN_MASK & bt) != 0;

	signed long int Aexp = (EXP_MASK & at) >> MNTSZ;
	signed long int Bexp = (EXP_MASK & bt) >> MNTSZ;

	unsigned long int Amnt = (MNT_MASK & at);		// The hidden bit is not appended yet. In MATLAB or Verilog code, the hidden bit is appended at this point
	unsigned long int Bmnt = (MNT_MASK & bt);

	//--------------------------------------------------------------------------

	//----Exceptions Handling----
	// Flags for input and output flag decisions based on input
	bool	Azero = (Aexp == 0);
	bool	Bzero = (Bexp == 0);
	bool	Inzero = Azero | Bzero;

	bool	Ainf = (Aexp == MAXEXP);
	bool	Binf = (Bexp == MAXEXP);
	bool	Ininf = Ainf | Binf;          // will be high for both Nan or Inf

	// excp and checking value of msb of mantissa(if that is 1, it is a Nan otherwise Inf)
	bool	Anan = Ainf & (Amnt  != 0);      // Checking the MSB of the mantissa. If it is one, then its a NaN. I could also check here if Amnt> (2^(MNT-1))
	bool	Bnan = Binf & (Bmnt  != 0);

	bool	Innan = Anan | Bnan;

	//------------------------------
	// Initializing Output Flags
	bool	flag_zero0 = 0, flag_zero1 = 0, flag_zero3 = 0;
	bool	flag_nan0 = 0;
	bool	flag_inf0 = 0, flag_inf1 = 0, flag_inf2 = 0, flag_inf3 = 0;


	flag_zero0 = Inzero & !Ininf;            // it may be high even if the output should be nan.so nan flag should be given most prioriy
	flag_nan0 = (Inzero &  Ininf) || Innan;  // The " OR Innan" part will be covered in Ininf because;
	flag_inf0 = !Inzero &  Ininf;            // it may be high even if the output should be nan.so nan flag should be given most prioriy


	//--------------------------------------------------------------------------
	// Sign Calculation
	//--------------------------------------------------------------------------
	bool Osgn = Asgn^Bsgn;

	//--------------------------------------------------------------------------
	// Exponent Calculation
	// --------------------------------------------------------------------------
	signed long int	Oexp1 = (Aexp + Bexp - BIAS);

	flag_zero1 = (Oexp1 < 0);
	flag_inf1 = (Oexp1 >= MAXEXP);


	//**************************************************************************
	// Mantissa Calculation (Main computation of the Code)
	//**************************************************************************
	unsigned long long int	Mult = fn_mulN_UREMrd(((1 << MNTSZ) + Amnt), ((1 << MNTSZ) + Bmnt), k);  // not sure if it will get the right number of bits or it will get truncation
	//	unsigned long long int	Mult = MitchelOptimIntMult(Amnt, Bmnt);  // Passing without the hidden bit as it is always known

	//#############################################################################################################################

	unsigned long long int	Normd;
	signed	      long int	Oexp2;
	//-------------------------------------------
	//normalization after mantissa multiplication
	//-------------------------------------------
	if (Mult >> (2 * P - 1))
	{
		Normd = Mult;
		Oexp2 = Oexp1 + 1; //%% --IMPORTANT -- NEED to check Exception here %
	}
	else
	{
		Normd = Mult << 1;
		Oexp2 = Oexp1;
	}

	flag_inf2 = (Oexp2 == MAXEXP);

	//------------------------------
	// Rounding (NO rounding)
	// -----------------------------
	//rb = 0;
	//unsigned long long int Rounded = (Normd >> P) + rb;		// Truncate 24 lower bits from the multiplier result and add rounding bit
	unsigned long long int Rounded = (Normd >> P);		// Truncate 24 lower bits from the multiplier result and add rounding bit

	//-------------------------------------------
	// normalization after rounding (NOT NEEDED)
	//-------------------------------------------
	unsigned long long int Rounded2;
	signed	      long int	Oexp3;
	// Check if there is carry from rounding and adjust
	/*if (Rounded >> P)						// check if 25th bit is one
	{
	Rounded2 = Rounded >> 1;
	Oexp3 = Oexp2 + 1;					//--IMPORTANT -- NEED to check Exception here %
	}
	else
	{*/
	Rounded2 = Rounded;
	Oexp3 = Oexp2;
	//}

	flag_inf3 = (Oexp3 == MAXEXP);
	flag_zero3 = (Oexp3 == 0);

	//**************************************************************************
	//  Putting together the Output
	//**************************************************************************
	bool	flag_zero = flag_zero0 | flag_zero1 | flag_zero3;
	bool	flag_nan = flag_nan0;
	bool	flag_inf = flag_inf0 | flag_inf1 | flag_inf2 | flag_inf3;

	unsigned long int Omnt;
	signed long int Oexp4;

	unsigned int ResultCase = 4 * (flag_nan)+2 * (flag_inf)+(flag_zero);

	switch (ResultCase)
	{
	case 0:					// Normal case
		Omnt = Rounded2;
		Oexp4 = Oexp3;
		break;
	case 1:					// Zero
		Oexp4 = 0;
		Omnt = 0;
		break;
	case 2:					// Infinity
		Oexp4 = MAXEXP;
		Omnt = 0;
		break;
	case 4:					// NaN
		Oexp4 = MAXEXP;
		Omnt = 1 << (MNTSZ - 1);
		break;
	default:				// For every other case, set output as Nan(it depend on our choice how we want to handle the error)
		Oexp4 = MAXEXP;
		Omnt = 1 << (MNTSZ - 1);
		break;
	}


	unsigned long int Os = (((unsigned int)(Osgn & ~flag_nan)) << (EXPSZ + MNTSZ));
	unsigned long int Oe = (Oexp4 << MNTSZ);
	unsigned long int Om = Omnt & MNT_MASK;

	unsigned long int Oi = Os + Oe + Om;

    //unsigned long int Oi = MitchelFPMultiply(at, bt);

    /////*******************************************************************************************************************************************
	

	//==============================================************ ENDS HERE ********=================================================

	// Truncating Output Bits
	//	unsigned long int Oi = *(unsigned int*)&Of;
	unsigned long int Ot = Oi; //& bitmask;
	float Oft = *(float*)&Ot;
    //=============================================================================
    
	return Oft;

}
//=============================================================================
//===============================IM2COL KERNEL=================================
//=============================================================================
__global__ void im2col_improved(const float *in,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, float *out)
{
//unsigned pc = ow * oh;
unsigned pl = kw * kh * c;

for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x; tId < pc*pl; tId += blockDim.x * gridDim.x)
{
unsigned patchId = (tId + po*pl) / pl;
unsigned outB    = (patchId / ow) / oh;
unsigned outH    = (patchId / ow) % oh;
unsigned outW    = patchId % ow;

unsigned valueId = (tId + po*pl) % pl;
unsigned offsetH = valueId / (kw * c);
unsigned offsetW = (valueId / c) % kw;
unsigned offsetC = valueId % c;

unsigned inH = outH * sh - ph + offsetH * dh;
unsigned inW = outW * sw - pw + offsetW * dw;

if(inH >= 0 && inW >= 0 && inH < h && inW < w)
out[tId] = in[((outB * h + inH) * w + inW) * c + offsetC];
else
out[tId] = float(0);
}

}
//=============================================================================
//=============================================================================
//=============================================================================
void im2colLauncher_Improved(
    const float* im, 
    const int batch,
    const int in_row,
    const int in_col, 
    const int out_row,
    const int out_col,
    const int out_depth,
    const int in_depth,
    const int filter_row,
    const int filter_col,
    const int stride_row,
    const int stride_col,
    // Padding
    const int left_offset,
    const int top_offset,
    const int dw,
    const int dh,
    float* data_col)
{

    unsigned pl = filter_row * filter_col * in_depth; 
    unsigned blockSize = 256;
    unsigned gridSize  = (batch * pl + blockSize - 1) / blockSize;
    im2col_improved<<<gridSize,blockSize,0>>>(im, in_depth, in_col, in_row, out_col, out_row, filter_col, filter_row,  left_offset,top_offset, stride_col, stride_row,dw,dh,0,batch*out_row*out_col,data_col);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


//=============================================================================
//===============================GEMM KERNEL===================================
//=============================================================================
#define TILE_DIM 8
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

if (i*TILE_DIM + threadIdx.x < k && Row < m)
As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
else
As[threadIdx.y][threadIdx.x] = float(0);

if (i*TILE_DIM + threadIdx.y < k && Col < n)
Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
else
Bs[threadIdx.y][threadIdx.x] = float(0);

__syncthreads();

for (int n = 0; n < TILE_DIM; ++n)
//value += bitmasking(bitmasking(As[threadIdx.y][n])*bitmasking(Bs[n][threadIdx.x]));
//value += FPmultMBM_cppv2(As[threadIdx.y][n],Bs[n][threadIdx.x],T_SIZE);
//value += FPMult_SinglePrecision_Rnone_Mitchell(As[threadIdx.y][n],Bs[n][threadIdx.x],MT);
value += As[threadIdx.y][n]*Bs[n][threadIdx.x];
__syncthreads();
}

if (Row < m && Col < n)
c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) +
(blockIdx.x * blockDim.x) + threadIdx.x] = value;
}

//=============================================================================
//=============================================================================
//=============================================================================
__global__ void gemm(size_t m, size_t n,
    size_t k, const float* vectorized, size_t lda, const float* filter, size_t ldb,
    float* c, size_t ldc,int size){
        const size_t a_i_stride = lda; //4
        // const size_t a_l_stride = 1;
        // const size_t b_j_stride = 1;
        const size_t b_l_stride = ldb;//1
        const size_t c_i_stride = ldc; //1
        // const size_t c_j_stride = 1;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size){
            const int batch_idx = blockIdx.y;
            vectorized += batch_idx*k*m;
            c += batch_idx*m*n;
            int j = index / m;
            int i = index % m;
            float total(0);
            //loop filter_value_count
            for (int l = 0; l < k; l++) {
            const size_t a_index = ((i * a_i_stride) + l );
            const float a_value = vectorized[a_index];
            // filter
            const size_t b_index = (j  + (l * b_l_stride));

            const float b_value = filter[b_index];
            total += FPMult_SinglePrecision_Rnone_Mitchell(a_value,b_value,8);
            //total += FPmultMBM_cppv2(a_value , b_value,T_SIZE);
           // total += bitmasking(bitmasking(a_value)*bitmasking(b_value));
            // total += halfmul(a_value,b_value);
            }
            const size_t c_index = ((i * c_i_stride) + j );
            c[c_index] = total;
        }

}


void gemm_reference( size_t m, size_t n,
    size_t k, const float* a, size_t lda, const float* b, size_t ldb,
    float* c, size_t ldc){
    const size_t a_i_stride = lda; // filter_value_coun
    // const size_t a_l_stride = 1;
    // const size_t b_j_stride = 1;
    const size_t b_l_stride = ldb; // filter_count
    const size_t c_i_stride = ldc; // filter_count
    // const size_t c_j_stride = 1;
    size_t i, j, l;
    // loop output depth
    for (j = 0; j < n; j++) {
      //loop patch
      for (i = 0; i < m; i++) {
        float total(0);
        //loop filter_value_count
        for (l = 0; l < k; l++) {
          const size_t a_index = ((i * a_i_stride) + l );
          const float a_value = a[a_index];
          // filter
          const size_t b_index = (j  + (l * b_l_stride));

          const float b_value = b[b_index];

          total += (a_value * b_value);
          
        }
        const size_t c_index = ((i * c_i_stride) + j );
        c[c_index] = total;
      }
    }
  }

void gemmLauncher(){

}
// HWC -> (C*f_h*f_w)*(out_row*out_col)
__global__ void im2col(
    const int size,
    const float* im,
    const int in_row,
    const int in_col,
    const int filter_row,
    const int filter_col,
    const int left_offset,
    const int top_offest,
    const int stride_row,
    const int stride_col,
    const int in_channel,
    //output height width
    const int height_cols,
    const int width_cols,
    int im_stride,
    int vec_stride,
    float* data_vec)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // let's narrow down our sight into one batch (each batch will handle by blockIdx.y)
    // size is equal to output vectorized amount i.e. HWC (height * witdh * channel)
    if (index < size){
        const int batch_idx = blockIdx.y;
        // select current image batch HWC format
        im += batch_idx * im_stride;
        // select current vectorized batch (f_h * f_w * channel) * (output_row * output_col)
        data_vec += batch_idx * vec_stride;
        
        const int h_index = index / in_channel;
        // 1
        const int c_im = index % in_channel;
        // 0
        const int w_col = h_index % width_cols;
        // 0
        const int h_col = h_index / width_cols;
        // 1
        const int c_col = c_im;
        // 0
        const int h_offset = h_col * stride_row - top_offest;
        const int w_offset = w_col * stride_col - left_offset;


        //index = c + indepth*COL + indepth*col*ROW
        const float* im_ptr = im;
        // 1
        im_ptr +=(h_offset * in_col + w_offset) * in_channel + c_im;
        float* vec_ptr = data_vec;
        //HWC
        vec_ptr += (h_col * width_cols+ w_col) * in_channel * filter_col * filter_row + c_col;

        for(int i = 0; i < filter_row; i++){
            for(int j = 0; j < filter_col; j++){
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                // *vec_ptr = (h_im >= 0 && w_im >= 0 && h_im < in_row && w_im < in_col )? batch_idx * im_stride+(h_offset * in_col + w_offset) * in_channel + c_im+(i*in_col+j)*in_channel:0;
                *vec_ptr = (h_im >= 0 && w_im >= 0 && h_im < in_row && w_im < in_col )? im_ptr[(i*in_col+j)*in_channel]:0;

                vec_ptr += in_channel;
                
            }
        }

    }

}




void im2colLauncher(
    const float* im, 
    const int batch,
    const int in_row,
    const int in_col, 
    const int out_row,
    const int out_col,
    const int out_depth,
    const int in_depth,
    const int filter_row,
    const int filter_col,
    const int stride_row,
    const int stride_col,
    // Padding
    const int left_offset,
    const int top_offset,
    float* data_col)
{
    int height_col = out_row;
    int witdh_col = out_col;
    int size = in_depth * height_col * witdh_col;
    // number of elements in one batch of input
    int im_stride = in_depth * in_row * in_col;
    // number of element in one batch of vectorized output   
    int vec_stride = in_depth * filter_row * filter_col * out_row * out_col;
   // printf("size %d, im_stride %d, vec_stride %d, %d leftoffset, %d topoffset\n",size, im_stride, vec_stride,left_offset,top_offset);
    dim3 dim_grid(ceil((float)size/BLOCK_SIZE),batch);

    im2col<<<dim_grid,BLOCK_SIZE>>>(
        size, im, in_row, in_col, filter_row, filter_col, left_offset, top_offset,
        stride_row, stride_col, in_depth, height_col, witdh_col, im_stride, vec_stride, data_col
    );

    // printf("size %d, in_row %d, in_col %d, filter_row %d, filter_col %d, left_offset %d, top_offset %d, stride_row %d, stride_col %d, in_depth %d, height_col %d, width_col %d\n",size,in_row, in_col, filter_row, filter_col, left_offset, top_offset,
    // stride_row, stride_col, in_depth, height_col, witdh_col);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


void ConvamKernellLauncher(
    const float* inputs,
    const float* filter,
    float* im2col,
    const int batch,
    const int in_row,
    const int in_col,
    const int out_row,
    const int out_col,
    const int out_depth,
    const int in_depth,
    const int filter_row,
    const int filter_col,
    const int stride_row,
    const int stride_col,
    // Padding
    const int left_offset,
    const int top_offset,
    const int padding,
    float* output
  ){
//     printf("batch %d, in_row %d, in_col %d, out_row %d, out_col %d, out_depth %d, in_depth %d, filter_row %d, filter_col %d, stride_row %d, stride_col %d, left_off %d, top_off %d\n",
//     batch, in_row, in_col, out_row, out_col, out_depth, in_depth, filter_row, filter_col, stride_row, stride_col, left_offset, top_offset
// );

    if (filter_row == 1 && filter_col == 1 && stride_row == 1 &&
        stride_col == 1) {
      // The kernel is 1x1.
      const int m = batch * in_row * in_col;
      const int n = out_depth;
      const int k = in_depth;
      const int lda = k;
      const int ldb = n;
      const int ldc = n;
      const int size = m*n;
      dim3 blockSize(8, 8, 1);
      dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
      gemm<<<gridSize,blockSize,0>>>(m,n,k,inputs,lda,filter,ldb,output,ldc);
      gpuErrchk( cudaPeekAtLastError() );
gpuErrchk( cudaDeviceSynchronize() );
      return;
    } else if (filter_row == in_row && filter_col== in_col &&
               padding == 1) {
      // The input data and filter have the same height/width.
      const int m = batch;
      const int n = out_depth;
      const int k = in_depth*in_col*in_row;
      const int lda = k;
      const int ldb = out_depth;
      const int ldc = out_depth;
      const int size = m*n;
      dim3 blockSize(8, 8, 1);
      dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
      gemm<<<gridSize,blockSize,0>>>(m,n,k,inputs,lda,filter,ldb,output,ldc);
      gpuErrchk( cudaPeekAtLastError() );
gpuErrchk( cudaDeviceSynchronize() );
      return;
    }
    double begin = realtime();
   im2colLauncher_Improved(inputs, batch, in_row, in_col, out_row, out_col,out_depth, in_depth, filter_row, filter_col, stride_row, stride_col, left_offset,top_offset, 1,1 ,im2col);
   cudaDeviceSynchronize();
   double end = realtime();
#if PROFILE == 1
cout << "Forward Im2col time difference = " << end - begin << endl;
#endif
   const size_t m = batch*out_row*out_col; //4
   const size_t n = out_depth; //  1
   const size_t k = filter_col * filter_row * in_depth; //4
   const size_t lda = k; //4
   const size_t ldb = out_depth;
   const size_t ldc = out_depth;
   dim3 blockSize(8, 8, 1);
   dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    begin =realtime();
   gemm<<<gridSize,blockSize,0>>>(m,n,k,im2col,lda,filter,ldb,output,ldc);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
     end = realtime();
#if PROFILE == 1
    cout << "Forward gemm time difference = " << end - begin << endl;
#endif
    // const int64 kMaxChunkSize = (16 * 1024 * 1024) / sizeof(float);
    // int64 patchLength  = filter_col*filter_row*in_depth; 
    // int64 totalPatchesCount = batch * out_row * out_col;
    // const int64 patchesPerChunk = kMaxChunkSize / patchLength;
    // for(int64 i = 0; i < totalPatchesCount; i += patchesPerChunk)
    // {
    //     int64 temp_batch = i/(out_row*out_col);
    //     int patchOffset = int(i % (out_row * out_col));
    //     int patchesCount = int(min(patchesPerChunk, totalPatchesCount - i));
    //     const float *patchInputData = 
    // }
    

  }

void Im2col(const float* input_data, const int depth, const int height,
    const int width, const int filter_h, const int filter_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h, const int stride_w, float* col_data) {
    int height_col = 3;
    int width_col = 3;

    int h_pad = -pad_t;
    for (int h = 0; h < height_col; ++h) {
        int w_pad = -pad_l;
            for (int w = 0; w < width_col; ++w) {
                for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                    for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        memcpy(col_data, input_data + (ih * width + iw) * depth,
                            sizeof(float) * depth);
                    } else {
                        // This should be simply padded with zero.
                        memset(col_data, 0, sizeof(float) * depth);
                    }
                col_data += depth;
                }
            }
        w_pad += stride_w;
        }
        h_pad += stride_h;
    }
}
__global__ void gemm_filter(size_t m, size_t n,
    size_t filter_row, size_t filter_col,size_t in_depth,const int stride_row,const int stride_col,const float* vectorized, size_t lda, const float* filter, size_t ldb,
    float* c, size_t ldc,int size){
        const size_t a_i_stride = lda; //4
        // const size_t a_l_stride = 1;
        // const size_t b_j_stride = 1;
        const size_t b_l_stride = ldb;//1
        const size_t c_i_stride = ldc; //1
        // const size_t c_j_stride = 1;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size){
            const int batch_idx = blockIdx.y;
            vectorized += batch_idx*filter_col*filter_row*in_depth*m;
            c += batch_idx*m*n;
            int j = index / m;
            int i = index % m;

            float total(0);
            for (int f = 0; f < filter_row; ++f ){
                for(int o = 0; o < filter_col; ++o){
                    for(int p = 0; p < in_depth; ++p){
                        const size_t a_index = ((i * a_i_stride) + f*filter_col*in_depth + o*in_depth+ p );
                        const float a_value = vectorized[a_index];
                        // filter
                        const float i_row = f/(float)stride_row;
                        const float i_col = o/(float)stride_row;
                        const bool y = fmod(i_row,(float)1)==float(0);
                        const bool x = fmod(i_col,(float)1)==float(0);

                        const size_t b_index = (j  + ( ((int)i_row)*filter_col*in_depth + ((int)i_col)*in_depth+ p  * b_l_stride));
                        
                        const float b_value = x&y?filter[b_index]:0;

                        total += (a_value * b_value);
                    }
                }
            }


            const size_t c_index = ((i * c_i_stride) + j );
            c[c_index] = total;
        }

}

__global__ void filtergradkernel(
    const int size,
    const int OUT_CHANNEL,
    const int IN_CHANNEL,
    const int FILTER_COL,
    const int FILTER_ROW,
    const int BATCH,
    const int HOLE_GRAD_HEIGHT,
    const int HOLE_GRAD_WIDTH,
    const int GRAD_HEIGHT,
    const int GRAD_WIDTH,
    const int STRIDE_ROW,
    const int STRIDE_COL,
    const int INPUT_HEIGHT,
    const int INPUT_WIDTH,
    const int LEFT_OFFSET,
    const int TOP_OFFSET,
    const float* input,
    const float* grad,
    float* out
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        // index = out_channel + OUT_CHANNEL*(in_channel + IN_CHANNEL*(col + FILTER_COL*row))
        const int out_channel = index % OUT_CHANNEL;
        // idx_filter = in_channel + IN_CHANNEL*(col + FILTER_COL*row)
        const int idx_filter = index / OUT_CHANNEL;
        const int in_channel = idx_filter % IN_CHANNEL;
        // idx_filter_slice = col + FILTER_COL*row
        const int idx_filter_slice = idx_filter / IN_CHANNEL;
        const int col = idx_filter_slice % FILTER_COL;
        const int row = idx_filter_slice / FILTER_COL;

        float temp = 0;
        for(int i = 0; i < BATCH; ++i){
            for(int j = 0; j < HOLE_GRAD_HEIGHT; ++j){
                for(int k = 0; k < HOLE_GRAD_WIDTH; ++k){
                    
                    const float i_row = j/(float)STRIDE_ROW;
                    const float i_col = k/(float)STRIDE_COL;
                    const bool y = fmod(i_row,(float)1)==float(0);
                    const bool x = fmod(i_col,(float)1)==float(0);
                    float grad_val = (x&y)?grad[i*GRAD_WIDTH*GRAD_HEIGHT*OUT_CHANNEL + int(i_row)*GRAD_WIDTH*OUT_CHANNEL+ int(i_col)*OUT_CHANNEL + out_channel]:0;
                    // float input_val = input[i*INPUT_HEIGHT*INPUT_WIDTH*IN_CHANNEL++ in_channel];
                    const int input_row = row - TOP_OFFSET + j;
                    const int input_col = col - LEFT_OFFSET + k;
                    float input_val = 0;
                    if( input_row >= 0 && input_col >=0 && input_row < INPUT_HEIGHT && input_col < INPUT_WIDTH){
                        input_val = input[i*INPUT_WIDTH*INPUT_HEIGHT*IN_CHANNEL+ input_row*INPUT_WIDTH*IN_CHANNEL + input_col*IN_CHANNEL+ in_channel];
                    }
                   // temp += FPMult_SinglePrecision_Rnone_Mitchell(input_val,grad_val,MT);
                    //temp += FPmultMBM_cppv2(input_val , grad_val,T_SIZE);
                    //temp += bitmasking(bitmasking(input_val)*bitmasking(grad_val));
                    //temp+= halfmul(input_val,grad_val);
temp += input_val*grad_val;
                }
            }
        }
        out[index] = temp;
    }
}
void ConvamFilterGradKernelLauncher(
    const float* input,
    const float* grad,
    float* im2col,
    const int input_height,
    const int input_width,
    const int batch,
    const int in_depth,
    const int grad_width,
    const int grad_height,
    const int grad_channel,
    const int filter_left_offset,
    const int filter_top_offset,
    const int stride_row,
    const int stride_col,
    const int filter_width,
    const int filter_height,
    float* out
){

    const int total_size = filter_height*filter_width*in_depth*grad_channel;
    const int grid_size = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
double begin = realtime();
    filtergradkernel<<<grid_size,BLOCK_SIZE>>>( total_size, grad_channel, in_depth, filter_width, filter_height, batch, ((grad_height-1)*stride_row+1), ((grad_width-1)*stride_col+1),
    grad_height, grad_width, stride_row, stride_col, input_height, input_width, filter_left_offset, filter_top_offset, input, grad, out

    );
    double end = realtime();
#if PROFILE == 1
    cout << "Filter gradient kernel difference = " << end - begin << endl;
#endif
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

//     if (grad_height == 1 && grad_width == 1) {
//       // The kernel is 1x1.
//       const int m = batch * in_row * in_col;
//       const int n = out_depth;
//       const int k = in_depth;
//       const int lda = k;
//       const int ldb = n;
//       const int ldc = n;
//       const int size = m*n;
//       dim3 blockSize(8, 8, 1);
//       dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
//       gemm<<<gridSize,blockSize,0>>>(m,n,k,inputs,lda,filter,ldb,output,ldc);
//       gpuErrchk( cudaPeekAtLastError() );
// gpuErrchk( cudaDeviceSynchronize() );
//       return;
//     } else if (grad_height == in_row && grad_width== in_col &&
//                padding == 1) {
//       // The input data and filter have the same height/width.
//       const int m = batch;
//       const int n = out_depth;
//       const int k = in_depth*in_col*in_row;
//       const int lda = k;
//       const int ldb = out_depth;
//       const int ldc = out_depth;
//       const int size = m*n;
//       dim3 blockSize(8, 8, 1);
//       dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
//       gemm<<<gridSize,blockSize,0>>>(m,n,k,inputs,lda,filter,ldb,output,ldc);
//       gpuErrchk( cudaPeekAtLastError() );
// gpuErrchk( cudaDeviceSynchronize() );
//       return;
//     }

//    im2colLauncher_Improved(inputs, batch, in_row, in_col, out_row, out_col,out_depth, in_depth, filter_row, filter_col, stride_row, stride_col, left_offset,top_offset, 1,1 im2col);
//    cudaDeviceSynchronize();
// //    float col[81];
// //    cudaMemcpy(col,im2col, sizeof(float)*81, cudaMemcpyDeviceToHost);
// //         for(int i =0; i < 81; i++){
// //     printf("%f   ",col[i]);

// //     }
// //     printf("\n");
//    const size_t m = batch*out_row*out_col; //4
//    const size_t n = out_depth; //  1
//    const size_t k = filter_col * filter_row * in_depth; //4
//    const size_t lda = k; //4
//    const size_t ldb = out_depth;
//    const size_t ldc = out_depth;
//    dim3 blockSize(8, 8, 1);
//    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
//    gemm<<<gridSize,blockSize,0>>>(m,n,k,im2col,lda,filter,ldb,output,ldc);

//     gpuErrchk( cudaPeekAtLastError() );
//     gpuErrchk( cudaDeviceSynchronize() );
    
};

__global__ void inserthole(
    const float* grad,
    const int grad_height,
    const int grad_width,
    const int grad_channel,
    const int hole_grad_height,
    const int hole_grad_width,
    const int final_size,
    const int stride_row,
    const int stride_col,
    const int size,
    float* out
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < final_size){
        const int batch_idx = blockIdx.y;
        grad += batch_idx * size;
        out += batch_idx * final_size;
        const int g_channel = index % grad_channel;
        const int g_channel_level_index = index / grad_channel;
        const int g_channel_level_col = g_channel_level_index % hole_grad_width;
        const int g_channel_level_row = g_channel_level_index / hole_grad_width;
        out+=index;
        float i_x = (hole_grad_width==grad_width)?g_channel_level_col:(g_channel_level_col/(float)stride_col);
        float i_y = (hole_grad_height==grad_height)?g_channel_level_row:(g_channel_level_row/(float)stride_row);
        const bool x = fmod(i_x,(float)1)==float(0);
        const bool y = fmod(i_y,(float)1)==float(0);
        *out = (x&y)?grad[g_channel+((int)i_x)*grad_channel+((int)i_y)*grad_channel*grad_width]:0;
        //*out = grad[g_channel+((int)i_x)*grad_channel+((int)i_y)*grad_channel*grad_width];
    }
}
__global__ void gemm_inverse(size_t m, size_t n,
    size_t k, const float* vectorized, size_t lda, const float* filter, size_t ldb,
    float* c, size_t ldc,const int size,
    const int filter_col,
    const int filter_row,
    const int out_depth,
    const int in_depth
){
        const size_t a_i_stride = lda; //9
        // const size_t a_l_stride = 1;
        // const size_t b_j_stride = 1;
        const size_t b_l_stride = ldb;//1
        const size_t c_i_stride = ldc; //1
        // const size_t c_j_stride = 1;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size){
            const int batch_idx = blockIdx.y;
            vectorized += batch_idx*k*m;
            c += batch_idx*m*n;
            //input_depth
            // 0
            int j = index / m;
            //patch
            // 1
            int i = index % m;
            float total(0);
            //loop filter_value_count
            // for (int l = 0; l < k; l++) {
            for(int x = 0; x < filter_col; x++){
                for(int y = 0; y < filter_row; y++){
                    //ideally this is the in_depth
                    for(int z = 0; z < out_depth; z++){
                    const size_t a_index = ((i * a_i_stride) + y*out_depth*filter_col+ x*out_depth+z);
                    const float a_value = vectorized[a_index];
                    // filter
                    //HWIO
                    const size_t b_index = ((filter_row-1-y)*out_depth*in_depth*filter_col+ (filter_col-1-x)*out_depth*in_depth+j*out_depth+z);

                    const float b_value = filter[b_index];
        //            total += FPMult_SinglePrecision_Rnone_Mitchell(a_value,b_value,MT);
                    //total += FPmultMBM_cppv2(a_value ,b_value, T_SIZE);
                   // total += bitmasking(bitmasking(a_value)*bitmasking(b_value));
                    //total += halfmul(a_value,b_value);
total += a_value*b_value;
            }}}
            // }
            const size_t c_index = ((i * c_i_stride) + j );
            c[c_index] = total;
        }

}

void ConvamInputGradKernelLauncher(
    // grad needs pading and holes
    // im2col input
    const float* grad,
    float* holed_grad,
    float* im2col,
    const int real_grad_height,
    const int real_grad_width,
    const int hole_grad_width,
    const int hole_grad_height,
    const int back_pad_top,
    const int back_pad_left,
    const int back_pad_bottom,
    const int back_pad_right,
    // reverse
    const float* filter,
    const int filter_height,
    const int filter_width,
    const int output_channel,
    const int stride_rows,
    const int stride_cols,
    // input related
    const int input_batch,
    const int input_height,
    const int input_width,
    const int input_channel,
    float* output
){


    // __global__ void inserthole(
    //     const float* grad,
    //     const int grad_height,
    //     const int grad_width,
    //     const int grad_channel,
    //     const int hole_grad_height,
    //     const int hole_grad_width,
    //     const int final_size,
    //     const int stride_row,
    //     const int stride_col,
    //     const int size,
    //     float* out
    // )
double begin1 = realtime();
    if(hole_grad_height!=real_grad_height||hole_grad_width!=real_grad_width){
        // float holed[input_batch*hole_grad_width*hole_grad_height*output_channel];

        const int holed_size = hole_grad_width*hole_grad_height*output_channel;
        const int real_size = output_channel*real_grad_height*real_grad_width;
        dim3 dim_grid(ceil((float)holed_size/BLOCK_SIZE),input_batch);
        inserthole<<<dim_grid,BLOCK_SIZE>>>( grad, real_grad_height, real_grad_width, output_channel, hole_grad_height, hole_grad_width, holed_size,
            stride_rows, stride_cols, real_size, holed_grad);
            
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // cudaMemcpy(holed,holed_grad, sizeof(float)*hole_grad_width*hole_grad_height*output_channel*input_batch, cudaMemcpyDeviceToHost);
        // for(int i = 0; i < hole_grad_width*hole_grad_height*output_channel*input_batch; i++){
        //     printf("%d: %f\n",i,holed[i]);
        // }

    // void im2colLauncher(
    //     const float* im, 
    //     const int batch,
    //     const int in_row,
    //     const int in_col, 
    //     const int out_row,
    //     const int out_col,
    //     const int out_depth,
    //     const int in_depth,
    //     const int filter_row,
    //     const int filter_col,
    //     const int stride_row,
    //     const int stride_col,
    //     // Padding
    //     const int left_offset,
    //     const int top_offset,
    //     float* data_col)
    // cudaError_t cudaerr = cudaDeviceSynchronize();
    // if (cudaerr != cudaSuccess)
    // {
    //     printf("input kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    // }
    // float col[output_channel * filter_height * filter_width * input_height * input_width*input_batch];

        // begin = realtime();
   im2colLauncher(
       holed_grad, input_batch, hole_grad_height, hole_grad_width, input_height, input_width,0,output_channel,filter_height,
       filter_width,1,1,back_pad_left,back_pad_top,im2col);
    //    end = realtime();
    //    cout << "backprop input grad im2col Time difference = " << end - begin << endl;
//   cudaMemcpy(col,im2col, sizeof(float)*output_channel * filter_height * filter_width * input_height * input_width*input_batch, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < output_channel * filter_height * filter_width * input_height * input_width*input_batch; ++i){
//        printf("%f\n",col[i]);
//        if((i+1)%8==0&&i!=0){
           
//            printf("\n");
//        }
//    }
   } else {
    // double begin = realtime();
    im2colLauncher(
        grad, input_batch, hole_grad_height, hole_grad_width, input_height, input_width,0,output_channel,filter_height,
        filter_width,1,1,back_pad_left,back_pad_top,im2col);
        // double end = realtime();
        // cout << "backprop input grad im2col Time difference = " << end - begin << endl;

   }
    double end1 = realtime();

#if PROFILE == 1
    cout << "Error backpropagation: Im2Col time difference = " << end1 - begin1 << endl;
#endif
    const size_t m = input_height*input_width; //4
    const size_t n = input_channel; //  1
    const size_t k = filter_width * filter_height * output_channel; //4
    const size_t lda = k; //4
    const size_t ldb = input_channel;
    const size_t ldc = input_channel;
    const int size = m*n;
    dim3 dim_grid1(ceil((float)size/BLOCK_SIZE),input_batch);
double begin = realtime();
    gemm_inverse<<<dim_grid1,BLOCK_SIZE>>>(m,n,k,im2col,lda,filter,ldb,output,ldc,size,filter_width,filter_height,output_channel,input_channel);
    double end = realtime();
#if PROFILE == 1
    cout << "Error backpropagation: Gemm inverse time = " << end - begin << endl;
#endif

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // cudaError_t cudaerr = cudaDeviceSynchronize();
    // if (cudaerr != cudaSuccess)
    // {
    //     printf("input kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    // }
    // float out[input_batch*input_height*input_width*input_channel];
    // cudaMemcpy(out,output, sizeof(float)*input_batch*input_height*input_width*input_channel, cudaMemcpyDeviceToHost);
    // std::fstream fs;
    // fs.open("gpu_input_log", std::fstream::in | std::fstream::out | std::fstream::app);
    // // ----------test for forward propagation-----------
    // for(int i = 0; i < input_batch; i++){
    //   for(int j = 0; j < input_height; j++){
    //     for(int k = 0; k < input_width; k++){
    //       for(int l = 0; l < input_channel; l++){
    //           // printf("(%d, %d, %d, %d) %f\n",i,j,k,l,output_data[(i * dimensions.out_cols * dimensions.out_rows * dimensions.out_depth) +
    //           //           (j * dimensions.out_cols * dimensions.out_depth) +
    //           //           (k * dimensions.out_depth) + l]);
    //           fs << i <<", " << j << ", " <<k<<", " << l << ",  " << out[(i* input_height * input_width *input_channel) +
    //                     (j * input_width *input_channel) +
    //                     (k *input_channel) + l]<<"\n";
              
    //       }
    //     }
    //   }
    // }
    // // printf("From GPU Input\n");
    // fs << "From GPU\n";
    // fs.close();
};


























































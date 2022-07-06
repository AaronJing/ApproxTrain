#define MANTISSA_MASK     0x7fffff 
#define EXPONENT_MASK     2139095040
#define MANTISSA_MASK_INV 4286578688
#define SIGN_MASK         2147483648

#define MANTISSA_TRUNC_MASK       8323072
#define MANTISSA_TRUNC_SETBIT     65536
#define INPUT16_MASK 0xffff0000
#define CLEAR_NORMALIZED 0x7fffffff

#include <math.h>

// MANTISSA_MASK: 2^23 - 1          :     {9{0}, 23{1}}
// MANTISSA_MASK_INV: 511*2^23      :    {9{1}, 23{0}}
// EXPONENT_MASK:    255^2^23       : {0, 8{1}, 23{0}}
// SIGN MASK             2^31       :  {1, 31{0}}
// MANTISSA_TRUNC_MASK  (2^7-1)*2^16 :  {9{0}, 7{1}, 16{0}}
// MANTISSA_TRUNC_SETBIT   2^16     :   {15{0}, 1{1}, 16{0}}
//********************************************************************************************************************************************
__device__ uint32_t fn_mul_mbmmant16(uint32_t Amnt, uint32_t Bmnt)
{
	// Approximate multiplication of mantissa
	uint32_t Smnt;
	uint32_t Smnt_corr;
	uint32_t Smnt_corr_whbit;
	uint16_t carry;
	uint32_t CorrTem = 655360;    // constant:  (2^-4+2^-6 x 2^23): in fixed point 9_23

	Smnt      = ((Amnt&MANTISSA_TRUNC_MASK)|MANTISSA_TRUNC_SETBIT) + ((Bmnt&MANTISSA_TRUNC_MASK)|MANTISSA_TRUNC_SETBIT);
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




//******************************************************************************************************************************************
#if (OPTIMIZATION == 0) 
__device__ __inline__ float FPmultMBM_fast16(float Af, float Bf, cudaTextureObject_t lut, cudaTextureObject_t exp_lut, uint32_t* lut_mem)
{
    // type casting
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;


	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);
	uint32_t  Bmnt = (MANTISSA_MASK & bt);


	uint32_t Mbm_mantmult = fn_mul_mbmmant16(Amnt, Bmnt);  // Passing without hidden bit
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


    //=============================================================================

	return Oft;

}
#elif (OPTIMIZATION == 1)
__device__ __inline__ float FPmultMBM_fast16(float Af, float Bf, cudaTextureObject_t lut, cudaTextureObject_t exp_lut, uint32_t* lut_mem)
{
    // type casting
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;
    at = at & INPUT16_MASK;
    bt = bt & INPUT16_MASK;


	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);
	uint32_t  Bmnt = (MANTISSA_MASK & bt);

    // Approximate Mantissa calculation
    uint32_t Mbm_mantmult = uint32_t(tex1Dfetch<uint32_t>(lut, (int)((Amnt&MANTISSA_TRUNC_MASK)>>9) | ((Bmnt&MANTISSA_TRUNC_MASK)>>16)));
    // get is_accurate_normalized bit
    bool is_accurate_normalized = (bool)(Mbm_mantmult&1);
    // get is_approx_normalized bit
    bool is_approx_normalized = (bool)((Mbm_mantmult&2)>>1);
    // clear is_accurate_normalized bit, is_normalized_bit 
    Mbm_mantmult = (Mbm_mantmult>>2)<<2;

	// Accurate mantissa calculation
//	uint64_t Acc_mantmul = ((uint64_t)(8388608 + Amnt)) * ((uint64_t)(8388608 + Bmnt));  // appending hidden bit : 2^23

	//--------------------------------------------------------------------------
	// Accurate FP multiplication
    float    Ofacc =  (*(float*)&at)*(*(float*)&bt);
	uint32_t  Otacc = *(uint32_t *)&Ofacc;

	//Extracting sign, exponent, mantissa
	uint32_t  Oaccsgn = (Otacc & SIGN_MASK);            // 2^31 :  {1, 31{0}}
	uint32_t  Oaccexp = (Otacc & EXPONENT_MASK) >> 23;

	//--------------------------------------------------------------------------
	float Oft;
    uint32_t Onewexp = Oaccexp;
//bool is_accurate_normalized;
//bool is_approx_normalized;
//
//is_approx_normalized   = (Mbm_mantmult >= 16777216);        // 2^24 {1, 24{0}} because this comes out to be 25-bit: This means approximate mantissa needs to be normalized.
//
//is_accurate_normalized = (Acc_mantmul  >= 140737488355328); // 2^47 {1, 47{0}} because this comes out to be 48-bit: This means accurate mantissa was normalized.  Rounding effect is not handled yet
//if(is_accurate_normalized==0){                              // check if normalization happens due to rounding (note, I am assuming rounding-normalization will never happen if original normalization has happened)
//    uint32_t Acc_mant_kept;             // 1.24 fxp
//    uint32_t Acc_mant_drop;
//    uint32_t Acc_mant_rounded;          // 2.23 fxp
//
//    bool rb, M0, R, S;
//
//    Acc_mant_kept = Acc_mantmul >> 22;                 // Acc_mantmul has 47 bits (because normalization not needed). chop it down to 25 bits.
//    Acc_mant_drop = Acc_mantmul % 4194304;             // 4194304== 2^22
//    M0 = ((Acc_mant_kept & 2) != 0);	               // last remaining bit     __x.
//	R =  ((Acc_mant_kept & 1) != 0);                   // first dropped bit      ___.x
//	S = (Acc_mant_drop != 0);                                // OR of all remaining bit___._xxxxxxxx
//	rb = (R & (M0 | S));
//
//    Acc_mant_rounded = (Acc_mant_kept >> 1) + rb;
//
//    is_accurate_normalized = (Acc_mant_rounded >= 16777216);  // 2^24 {1, 24{0}} because this comes out to be 25-bit(2.23)
//}

	// Adjustment of exponent
     if(Oaccexp == 255){ 		// case: inf or NAN
         return Ofacc;
     }
     else if (Oaccexp == 0){ // case: subnormal or 0
         return float(0);
     }
     
     if(is_accurate_normalized && !is_approx_normalized){
     	Onewexp = Oaccexp - 1;
     }
     
     if(!is_accurate_normalized && is_approx_normalized){
     	Onewexp = Oaccexp + 1;
     }

//    uint32_t Onewexp = tex1Dfetch<uint32_t>(exp_lut, (int)((((uint32_t)is_accurate_normalized)<<9)|(((uint32_t)is_approx_normalized)<<8)|Oaccexp));

	Mbm_mantmult = Mbm_mantmult & MANTISSA_MASK; // getting rid of hidden bit

	uint32_t Os = Oaccsgn;
    uint32_t Oe = (Onewexp << 23);
    uint32_t Om = Mbm_mantmult;

    uint32_t Oi = Os + Oe + Om;
    Oi = Oi & INPUT16_MASK;

    Oft = *(float*)&Oi;


    //=============================================================================

	return Oft;

}

#elif (OPTIMIZATION == 2) 

__device__ __inline__ float FPmultMBM_fast16(float Af, float Bf, cudaTextureObject_t lut, cudaTextureObject_t exp_lut, uint32_t* lut_mem)
{
    // type casting
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;

    at = at & INPUT16_MASK;
    bt = bt & INPUT16_MASK;

	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);
	uint32_t  Bmnt = (MANTISSA_MASK & bt);

    // Approximate Mantissa calculation
    // 40 % of total computation time
    //uint32_t Mbm_mantmult = 0;
//    uint32_t Mbm_mantmult = fn_mul_mbmmant16(Amnt, Bmnt);  // Passing without hidden bit
    uint32_t Mbm_mantmult = uint32_t(tex1Dfetch<uint32_t>(lut, (int)((Amnt&MANTISSA_TRUNC_MASK)>>9) | ((Bmnt&MANTISSA_TRUNC_MASK)>>16)));
    // get is_accurate_normalized bit
    bool is_accurate_normalized = (bool)(Mbm_mantmult&1);
    // get is_approx_normalized bit
    bool is_approx_normalized = (bool)((Mbm_mantmult&2)>>1);
    // clear is_accurate_normalized bit, is_normalized_bit 
    Mbm_mantmult = (Mbm_mantmult>>2)<<2;
//   // Accurate mantissa calculation
   float Oft;
//   //--------------------------------------------------------------------------
//   // Accurate FP multiplication
   float    Ofacc =  (*(float*)&at)*(*(float*)&bt);
   //float Ofacc = 0;


   uint32_t  Otacc = *(uint32_t *)&(Ofacc);

   //Extracting sign, exponent, mantissa
   uint32_t  Oaccsgn = (at ^ bt) & SIGN_MASK;            // 2^31 :  {1, 31{0}}
   uint32_t  Oaccexp = (Otacc & EXPONENT_MASK) >> 23;
//  uint32_t  Oaccexp = (at & EXPONENT_MASK) + (bt & EXPONENT_MASK);
//   Oaccexp = (Oaccexp >> 23)-127;

	//--------------------------------------------------------------------------
//	bool is_approx_normalized = true;

//	bool is_approx_normalized   = (Mbm_mantmult >= 16777216);        // 2^24 {1, 24{0}} because this comes out to be 25-bit: This means approximate mantissa needs to be normalized.

    // start of second lookup table
//  uint64_t   Acc_mantmul = ((uint64_t)(8388608 | Amnt)) * ((uint64_t)(8388608 | Bmnt));  // appending hidden bit : 2^23
//bool is_accurate_normalized = 0; // 2^47 {1, 47{0}} because this comes out to be 48-bit: This means accurate mantissa was normalized.  Rounding effect is not handled yet
//  if(is_accurate_normalized==0){                              // check if normalization happens due to rounding (note, I am assuming rounding-normalization will never happen if original normalization has happened)
//    uint32_t Acc_mant_kept;             // 1.24 fxp
//    uint32_t Acc_mant_drop;
//    uint32_t Acc_mant_rounded;          // 2.23 fxp
//
//    bool rb, M0, R, S;
//
//    Acc_mant_kept = Acc_mantmul >> 22;                 // Acc_mantmul has 47 bits (because normalization not needed). chop it down to 25 bits.
//    Acc_mant_drop = Acc_mantmul % 4194304;             // 4194304== 2^22
//    M0 = ((Acc_mant_kept & 2) != 0);	               // last remaining bit     __x.
//	R =  ((Acc_mant_kept & 1) != 0);                   // first dropped bit      ___.x
//	S = (Acc_mant_drop != 0);                                // OR of all remaining bit___._xxxxxxxx
//	rb = (R & (M0 | S));
//
//    Acc_mant_rounded = (Acc_mant_kept >> 1) + rb;
//
//    is_accurate_normalized = (Acc_mant_rounded >= 16777216);  // 2^24 {1, 24{0}} because this comes out to be 25-bit(2.23)
// }
    // end of second lookup table
    

	// Adjustment of exponent
// if (is_accurate_normalized){
//     Oaccexp = Oaccexp + 1;
// }
//  uint32_t Onewexp = Oaccexp;
//  if(Oaccexp == 255){ 		// case: inf or NAN
//      uint32_t Os = Oaccsgn;
//      uint32_t Oe = Oaccexp << 23;
//      uint32_t Oi = Os + Oe;
//  	Oft = *(float*)&Oi ;  		//do nothing: pass out the special case
//      
//      return Oft;
//  }
//  else if (Oaccexp == 0){ // case: subnormal or 0
//  	            // make the answer 0 because we do not handle subnormals
//      return 0;
//  }
// if(is_accurate_normalized && !is_approx_normalized){
//     Onewexp = Oaccexp - 1;
// }
// if(!is_accurate_normalized && is_approx_normalized){
//     Onewexp = Oaccexp + 1;
// }
    
//   if(is_approx_normalized){
//   	Mbm_mantmult = Mbm_mantmult >> 1;  // normalization without rounding
//   }
    uint32_t new_Oaccexp = tex1Dfetch<uint32_t>(exp_lut, (int)((((uint32_t)is_accurate_normalized)<<9)|(((uint32_t)is_approx_normalized)<<8)|Oaccexp));
   uint32_t Onewexp = new_Oaccexp;
   
   Mbm_mantmult = Mbm_mantmult & MANTISSA_MASK; // getting rid of hidden bit

   uint32_t Os = Oaccsgn;
   uint32_t Oe = (Onewexp << 23);
   uint32_t Om = Mbm_mantmult;

   uint32_t Oi = Os +Oe + Om;
   Oi = Oi & INPUT16_MASK;

    Oft = *(float*)&Oi;

    //=============================================================================

	return Oft;

}
#elif ( OPTIMIZATION == 3 )
__device__ __inline__ float FPmultMBM_fast16(float Af, float Bf, cudaTextureObject_t lut, cudaTextureObject_t exp_lut, uint32_t* lut_mem)
{
    // type casting
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;

    at = at & INPUT16_MASK;
    bt = bt & INPUT16_MASK;

	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);
	uint32_t  Bmnt = (MANTISSA_MASK & bt);

    // Approximate Mantissa calculation
    // 40 % of total computation time
    //uint32_t Mbm_mantmult = 0;
//    uint32_t Mbm_mantmult = fn_mul_mbmmant16(Amnt, Bmnt);  // Passing without hidden bit
    uint32_t Mbm_mantmult = lut_mem[(int)((Amnt&MANTISSA_TRUNC_MASK)>>9) | ((Bmnt&MANTISSA_TRUNC_MASK)>>16)];
    // get is_accurate_normalized bit
    bool is_accurate_normalized = (bool)(Mbm_mantmult&1);
    // get is_approx_normalized bit
    bool is_approx_normalized = (bool)((Mbm_mantmult&2)>>1);
    // clear is_accurate_normalized bit, is_normalized_bit 
    Mbm_mantmult = (Mbm_mantmult>>2)<<2;
//   // Accurate mantissa calculation
   float Oft;
//   //--------------------------------------------------------------------------
//   // Accurate FP multiplication
   float    Ofacc =  (*(float*)&at)*(*(float*)&bt);
   //float Ofacc = 0;


   uint32_t  Otacc = *(uint32_t *)&(Ofacc);

   //Extracting sign, exponent, mantissa
   uint32_t  Oaccsgn = (at ^ bt) & SIGN_MASK;            // 2^31 :  {1, 31{0}}
   uint32_t  Oaccexp = (Otacc & EXPONENT_MASK) >> 23;
//  uint32_t  Oaccexp = (at & EXPONENT_MASK) + (bt & EXPONENT_MASK);
//   Oaccexp = (Oaccexp >> 23)-127;

	//--------------------------------------------------------------------------
//	bool is_approx_normalized = true;

//	bool is_approx_normalized   = (Mbm_mantmult >= 16777216);        // 2^24 {1, 24{0}} because this comes out to be 25-bit: This means approximate mantissa needs to be normalized.

    // start of second lookup table
//  uint64_t   Acc_mantmul = ((uint64_t)(8388608 | Amnt)) * ((uint64_t)(8388608 | Bmnt));  // appending hidden bit : 2^23
//bool is_accurate_normalized = 0; // 2^47 {1, 47{0}} because this comes out to be 48-bit: This means accurate mantissa was normalized.  Rounding effect is not handled yet
//  if(is_accurate_normalized==0){                              // check if normalization happens due to rounding (note, I am assuming rounding-normalization will never happen if original normalization has happened)
//    uint32_t Acc_mant_kept;             // 1.24 fxp
//    uint32_t Acc_mant_drop;
//    uint32_t Acc_mant_rounded;          // 2.23 fxp
//
//    bool rb, M0, R, S;
//
//    Acc_mant_kept = Acc_mantmul >> 22;                 // Acc_mantmul has 47 bits (because normalization not needed). chop it down to 25 bits.
//    Acc_mant_drop = Acc_mantmul % 4194304;             // 4194304== 2^22
//    M0 = ((Acc_mant_kept & 2) != 0);	               // last remaining bit     __x.
//	R =  ((Acc_mant_kept & 1) != 0);                   // first dropped bit      ___.x
//	S = (Acc_mant_drop != 0);                                // OR of all remaining bit___._xxxxxxxx
//	rb = (R & (M0 | S));
//
//    Acc_mant_rounded = (Acc_mant_kept >> 1) + rb;
//
//    is_accurate_normalized = (Acc_mant_rounded >= 16777216);  // 2^24 {1, 24{0}} because this comes out to be 25-bit(2.23)
// }
    // end of second lookup table
    
    uint32_t Onewexp = Oaccexp;
	// Adjustment of exponent
     if(Oaccexp == 255){ 		// case: inf or NAN
         return Ofacc;
     }
     else if (Oaccexp == 0){ // case: subnormal or 0
         return float(0);
     }
     
     if(is_accurate_normalized && !is_approx_normalized){
     	Onewexp = Oaccexp - 1;
     }
     
     if(!is_accurate_normalized && is_approx_normalized){
     	Onewexp = Oaccexp + 1;
     }

//    uint32_t Onewexp = tex1Dfetch<uint32_t>(exp_lut, (int)((((uint32_t)is_accurate_normalized)<<9)|(((uint32_t)is_approx_normalized)<<8)|Oaccexp));

	Mbm_mantmult = Mbm_mantmult & MANTISSA_MASK; // getting rid of hidden bit

	uint32_t Os = Oaccsgn;
    uint32_t Oe = (Onewexp << 23);
    uint32_t Om = Mbm_mantmult;

    uint32_t Oi = Os + Oe + Om;
    Oi = Oi & INPUT16_MASK;

    Oft = *(float*)&Oi;


    //=============================================================================

	return Oft;

}
#elif (OPTIMIZATION == 4)
__device__ __inline__ float FPmultMBM_fast16(float Af, float Bf, cudaTextureObject_t lut, cudaTextureObject_t exp_lut, uint32_t* lut_mem)
{
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;



	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);
	uint32_t  Bmnt = (MANTISSA_MASK & bt);

    // Approximate Mantissa calculation
    uint32_t Mbm_mantmult = uint32_t(tex1Dfetch<uint32_t>(lut, (int)((Amnt)>>9) | ((Bmnt)>>16)));
    // get is_accurate_normalized bit
    bool is_accurate_normalized = (bool)(Mbm_mantmult&1);
    // get is_approx_normalized bit
    bool is_approx_normalized = (bool)((Mbm_mantmult&2)>>1);
    // clear is_accurate_normalized bit, is_normalized_bit 
    Mbm_mantmult = (Mbm_mantmult>>2)<<2;

	// Accurate mantissa calculation
//	uint64_t Acc_mantmul = ((uint64_t)(8388608 + Amnt)) * ((uint64_t)(8388608 + Bmnt));  // appending hidden bit : 2^23

	//--------------------------------------------------------------------------
	// Accurate FP multiplication

	//Extracting sign, exponent, mantissa
	uint32_t  Oaccsgn = (at ^ bt) & SIGN_MASK;            // 2^31 :  {1, 31{0}}
	//uint32_t  Oaccexp = (Otacc & EXPONENT_MASK) >> 23;
    uint32_t Oaccexp = ((at & EXPONENT_MASK)>>23) + ((bt & EXPONENT_MASK)>>23);

    if(Oaccexp <= 127){ 		// case: inf or NAN
        return 0;
    }
     

    Oaccexp = Oaccexp - 127 - (uint32_t)(is_accurate_normalized & (!is_approx_normalized)) + (uint32_t)((!is_accurate_normalized) & is_approx_normalized);



    uint32_t Oi = Oaccsgn + (Oaccexp << 23) + Mbm_mantmult;




    //=============================================================================

	return *(float*)&Oi;

}
#else
__device__ __inline__ float FPmultMBM_fast16(float Af, float Bf, cudaTextureObject_t lut, cudaTextureObject_t exp_lut, uint32_t* lut_mem)
{
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;



	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);
	uint32_t  Bmnt = (MANTISSA_MASK & bt);

    // Approximate Mantissa calculation
    uint32_t Mbm_mantmult = uint32_t(tex1Dfetch<uint32_t>(lut, (int)((Amnt)>>9) | ((Bmnt)>>16)));
    uint32_t is_normalized = (Mbm_mantmult&SIGN_MASK)>>31;
    Mbm_mantmult = Mbm_mantmult&CLEAR_NORMALIZED;

	uint32_t Oaccsgn = (at ^ bt) & SIGN_MASK;            // 2^31 :  {1, 31{0}}
    uint32_t Oaccexp = ((at & EXPONENT_MASK)>>23) + ((bt & EXPONENT_MASK)>>23);

    if(Oaccexp <= 127){ 		// case:0
        return 0;
    }
    Oaccexp = Oaccexp - 127 + is_normalized;
    uint32_t Oi = Oaccsgn + (Oaccexp << 23) + Mbm_mantmult;
	return *(float*)&Oi;

}
#endif

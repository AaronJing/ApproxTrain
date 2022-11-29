#define MANTISSA_MASK     8388607
#define EXPONENT_MASK     2139095040
#define MANTISSA_MASK_INV 4286578688
#define SIGN_MASK         2147483648

// MANTISSA_MASK: 2^23 - 1    :     {9{0}, 23{1}}
// MANTISSA_MASK_INV: 511*2^23 :    {9{1}, 23{0}}
// EXPONENT_MASK:    255^2^23  : {0, 8{1}, 23{0}}
// SIGN MASK             2^31 :  {1, 31{0}}


//********************************************************************************************************************************************
__device__ __inline__ uint32_t fn_mul_mbmmant32(uint32_t Amnt, uint32_t Bmnt, bool& exp_adj)
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
    if (Smnt_corr_whbit >= 16777216) 
    {
        Smnt_corr_whbit = Smnt_corr_whbit >> 1;
        exp_adj = true;
    } else {
        exp_adj = false;
    }

	return Smnt_corr_whbit & MANTISSA_MASK;	// normalized
}


__device__ __inline__  float FPmultMBM_fast32(float Af, float Bf) {
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;
	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (MANTISSA_MASK & at);
	uint32_t  Bmnt = (MANTISSA_MASK & bt);
    bool exp_adj = 0;
    // Approximate Mantissa calculation
    uint32_t Mbm_mantmult = fn_mul_mbmmant32(Amnt, Bmnt, exp_adj); 

	uint32_t Oaccsgn = (at ^ bt) & SIGN_MASK;            // 2^31 :  {1, 31{0}}
    uint32_t Oaccexp = ((at & EXPONENT_MASK)>>23) + ((bt & EXPONENT_MASK)>>23);

    if(Oaccexp <= 127 or (at & EXPONENT_MASK == 0) or (bt & EXPONENT_MASK == 0))
    { 		// case:0
        return 0;
    }
    Oaccexp = Oaccexp - 127 + (uint32_t)exp_adj;
    uint32_t Oi = Oaccsgn + (Oaccexp << 23) + Mbm_mantmult;
	return *(float*)&Oi;
}



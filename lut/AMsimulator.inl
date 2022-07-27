
//#define MANTISSA_MASK     0x007f0000 
#define EXPONENT_MASK     2139095040
#define MANTISSA_MASK_INV 4286578688
#define SIGN_MASK         2147483648

#define INPUT16_MASK 0xffff0000
#define CLEAR_NORMALIZED 0x7fffffff

float AMsimulator(float Af, float Bf, std::vector<uint32_t> &lut, uint32_t mant_mask, int a_shift, int b_shift)
{
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;
	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (mant_mask & at);
	uint32_t  Bmnt = (mant_mask & bt);
    // Approximate Mantissa calculation
    uint32_t Mbm_mantmult = lut.data()[((Amnt)>>a_shift) | ((Bmnt)>>b_shift)];
    uint32_t is_normalized = (Mbm_mantmult&SIGN_MASK)>>31;
    Mbm_mantmult = Mbm_mantmult&CLEAR_NORMALIZED;

	uint32_t Oaccsgn = (at ^ bt) & SIGN_MASK;            // 2^31 :  {1, 31{0}}
    uint32_t Oaccexp = ((at & EXPONENT_MASK)>>23) + ((bt & EXPONENT_MASK)>>23);

    if(Oaccexp <= 127 | ((at&EXPONENT_MASK) == 0) | ((bt&EXPONENT_MASK) == 0)){ 		// case:0
        return 0;
    }
    if(Oaccexp >= 384){
        return INFINITY;
    }
    Oaccexp = Oaccexp - 127 + is_normalized;
    uint32_t Oi = Oaccsgn + (Oaccexp << 23) + Mbm_mantmult;
	return *(float*)&Oi;

}

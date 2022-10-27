
#define MANTISSA_MASK     0x7fffff 
#define EXPONENT_MASK     2139095040
#define MANTISSA_MASK_INV 4286578688
#define SIGN_MASK         2147483648

#define MANTISSA_TRUNC_MASK       8323072
#define MANTISSA_TRUNC_SETBIT     65536
#define INPUT16_MASK 0xffff0000
#define CARRY_MASK 0x80
#define CLEAR_CARRY_MASK 0x7f

__device__ __inline__ float AMsimulator(float Af, float Bf, cudaTextureObject_t lut, uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth)
{
    uint32_t  at = *(uint32_t *)&Af;
	uint32_t  bt = *(uint32_t *)&Bf;
	// Extracting mantissa bits: bitwise anding
	uint32_t  Amnt = (mant_mask & at);
	uint32_t  Bmnt = (mant_mask & bt);
    // Approximate Mantissa calculation
    uint32_t Mbm_mantmult = tex1Dfetch<uint8_t>(lut, (Amnt>>a_shift)|(Bmnt>>b_shift));
    uint32_t is_normalized = (Mbm_mantmult&CARRY_MASK)>>7;
    Mbm_mantmult = (Mbm_mantmult&CLEAR_CARRY_MASK) << (23-mant_bitwidth);

	uint32_t Oaccsgn = (at ^ bt) & SIGN_MASK;            // 2^31 :  {1, 31{0}}
    uint32_t Oaccexp = ((at & EXPONENT_MASK)>>23) + ((bt & EXPONENT_MASK)>>23);

    if(Oaccexp <= 127 or (at & EXPONENT_MASK == 0) or (bt & EXPONENT_MASK == 0)){ 		// case:0
        return 0;
    }
    Oaccexp = Oaccexp - 127 + is_normalized;
    uint32_t Oi = Oaccsgn + (Oaccexp << 23) + Mbm_mantmult;
	return *(float*)&Oi;

}

#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <bitset>
#include <string>
#include <cmath>
void floatToBinary(float f, std::string& str)
{

    union { float f; uint32_t i; } u;
    u.f = f;
    str.clear();

    for (int i = 0; i < 32; i++)
    {
        if (u.i % 2)  str.push_back('1');
        else str.push_back('0');
        u.i >>= 1;
    }

    // Reverse the string since now it's backwards
    std::string temp(str.rbegin(), str.rend());
    str = temp;
}
#ifdef FMBM16_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast16((a),(b));
    #include "FPmultMBM_fast16.inl"
    #define MANTISSA_BITWIDTH 7
    std::string lut_save = "MBM_7.bin";
#elif FMBM14_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast14((a),(b));
    #include "FPmultMBM_fast14.inl"
    #define MANTISSA_BITWIDTH 5
    std::string lut_save = "MBM_5.bin";
#elif FMBM12_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast12((a),(b));
    #include "FPmultMBM_fast12.inl"
    #define MANTISSA_BITWIDTH 3
    std::string lut_save = "MBM_3.bin";
#elif FMBM10_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMBM_fast10((a),(b));
    #include "FPmultMBM_fast10.inl"
    #define MANTISSA_BITWIDTH 1
    std::string lut_save = "MBM_1.bin";
#elif MITCHEL16_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast16((a),(b));
    #include "Mitchell_16.inl"
    #define MANTISSA_BITWIDTH 7
    std::string lut_save = "MIT_7.bin";
#elif MITCHEL14_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast14((a),(b));
    #include "Mitchell_14.inl"
    #define MANTISSA_BITWIDTH 5
    std::string lut_save = "MIT_5.bin";
#elif MITCHEL12_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast12((a),(b));
    #include "Mitchell_12.inl"
    #define MANTISSA_BITWIDTH 3
    std::string lut_save = "MIT_3.bin";
#elif MITCHEL10_MULTIPLIER
    #define MULTIPLY(a,b) FPmultMitch_fast10((a),(b));
    #include "Mitchell_10.inl"
    #define MANTISSA_BITWIDTH 1
    std::string lut_save = "MIT_1.bin";
#elif BFLOAT
    #define MULTIPLY(a,b) bfloat16mul((a),(b));
    #include "bfloat.inl"
    #define MANTISSA_BITWIDTH 7
    std::string lut_save = "ACC_7.bin";
#endif

#define EMPTYFP32 0x00000000
//#define SIGN_MASK_ 0x80000000
#define EXPONENT127 0x3f800000
#define EXPONENT_MASK_ 0x7f800000
#define MANTISSA_MASK_ ((uint32_t(pow(2,MANTISSA_BITWIDTH))-1) << (23-MANTISSA_BITWIDTH))
// implementation for approximate mantissa multiplications lookup table generation
int main(){
    // create a and b
    float a = 0;
    float b = 0;
    // cast to uint32_t
    uint32_t  at = *(uint32_t *)&a;
	uint32_t  bt = *(uint32_t *)&b;
    // FP32 with bits set to all zeros
    at = at & EMPTYFP32;
    bt = bt & EMPTYFP32;
    // set sign to 0 or 1
    // set exponents A B C (output of A*B) should be normal case
    // 0b0011 1111 1000 0000 0000 0000 0000 0000 Biased exponent = 127
    at = at | EXPONENT127;
    bt = bt | EXPONENT127;



    char *lut_save_name = &lut_save[0];
    FILE *f = fopen(lut_save_name, "wb");
    for(uint32_t i = 0; i < uint32_t(pow(2,MANTISSA_BITWIDTH)); ++i){
        for(uint32_t j = 0; j < uint32_t(pow(2,MANTISSA_BITWIDTH)); ++j){
            uint32_t newat = at | (i<<(23-MANTISSA_BITWIDTH));
            uint32_t newbt = bt | (j<<(23-MANTISSA_BITWIDTH));
            float newa = *(float*)&newat;
            float newb = *(float*)&newbt;
            float c = MULTIPLY(newa, newb);
            uint32_t ct = *(uint32_t *)&c;
            uint8_t MANTISSA = (ct & MANTISSA_MASK_) >> (23-MANTISSA_BITWIDTH);
            uint32_t c_exp = ct & EXPONENT_MASK_;
            uint32_t un_normalized_exp = ((EXPONENT127>>23) + (EXPONENT127>>23) - 127)<<23;
            uint8_t carry = 0;
            if(un_normalized_exp < c_exp)
                carry = 0x80;
            uint8_t result = carry | MANTISSA;
            fwrite(&result, sizeof(uint8_t), 1, f);
        }
    }

    fclose(f);
    return 0;
}

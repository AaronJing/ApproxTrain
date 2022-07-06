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
#include "mbm16_simulation.h"
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
#define EMPTYFP32 0x00000000
#define MANTISSA_BITWIDTH 7
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
    std::cout << MANTISSA_MASK_ << std::endl;
    FILE *f = fopen("mbmmantmul16.bin", "wb");
    for(uint32_t i = 0; i < uint32_t(pow(2,MANTISSA_BITWIDTH)); ++i){
        for(uint32_t j = 0; j < uint32_t(pow(2,MANTISSA_BITWIDTH)); ++j){
            uint32_t newat = at | (i<<(23-MANTISSA_BITWIDTH));
            uint32_t newbt = bt | (j<<(23-MANTISSA_BITWIDTH));
            float newa = *(float*)&newat;
            float newb = *(float*)&newbt;
            float c = FPmultMBM_fast16(newa, newb);
            uint32_t ct = *(uint32_t *)&c;
            uint32_t MANTISSA = ct & MANTISSA_MASK_;
            uint32_t c_exp = ct & EXPONENT_MASK_;
            uint32_t un_normalized_exp = ((EXPONENT127>>23) + (EXPONENT127>>23) - 127)<<23;
            uint32_t carry = 0;
            if(un_normalized_exp < c_exp)
                carry = 0x80000000;
            MANTISSA = carry | MANTISSA;
            fwrite(&MANTISSA, sizeof(uint32_t), 1, f);
            std::cout << MANTISSA << " " << newat << " " << newbt << " " << carry<< std::endl;
            
        }
    }
    fclose(f);
    return 0;
}

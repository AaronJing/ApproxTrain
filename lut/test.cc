
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
#define EMPTYFP32 0x00000000
//#define SIGN_MASK_ 0x80000000
#define EXPONENT127 0x3f800000
#define EXPONENT_MASK_ 0x7f800000
#define MANTISSA_MASK_ ((uint32_t(pow(2,MANTISSA_BITWIDTH))-1) << (23-MANTISSA_BITWIDTH))
// implementation for approximate mantissa multiplications lookup table generation
#define MULTIPLY_GOLD(a,b) FPmultMBM_fast14((a),(b));
#define SIGN_MASK_ 0x80000000
#define CLEAR_SIGN 0x7fffffff
#include "FPmultMBM_fast14.inl"
#include "AMsimulator.inl"
int main(int argc, char** argv){
    if (argc == 0) exit(1);

    uint32_t start = (uint32_t)std::stoi(argv[1]);
    uint32_t stop = (uint32_t)std::stoi(argv[2]);
    std::ifstream file("MBM_5.bin", std::ios::in | std::ios::binary);
    if(file.fail()) {
        std::cerr << "lut file read failed" << std::endl;
        exit(1);
    } 
    if(!file.is_open()) { 
        std::cerr << "lut file open failed" << std::endl;
        exit(1);
    }
    auto mant_mul_lut_ = std::vector<uint32_t>{};
    mant_mul_lut_.resize(2<<(5*2));
    file.read(
            reinterpret_cast<char *>(mant_mul_lut_.data()), 
            mant_mul_lut_.size() * sizeof(uint32_t)
        );
//  for(uint32_t i : mant_mul_lut_){
//      std::cout << "carry: " << ((i&SIGN_MASK_)>>31) << " mant: "<< (i&CLEAR_SIGN) << "\n";
//  }
//  return 0;
    uint32_t diff = 0;
    uint32_t exp_diff = 0;
    uint32_t mant_diff = 0;
    uint32_t sign_diff = 0;
    //for(uint32_t i = 0; i < uint32_t(pow(2,16)); ++i){
    for(uint32_t i = start; i < stop; ++i){
        for(uint32_t j = 0; j < uint32_t(pow(2,14)); ++j){
            uint32_t newa_uint32 = i<<18;
            uint32_t newb_uint32 = j<<18;
            float newa = *(float*)& newa_uint32;
            float newb = *(float*)& newb_uint32;
            float gold = MULTIPLY_GOLD(newa, newb);
            //uint32_t mant_mask = 0x007F0000;
            uint32_t mant_mask = ((1<<5) -1) << (23-5);
            float lut = AMsimulator(newa, newb, mant_mul_lut_,mant_mask,13,18);
            uint32_t newa_ = *(uint32_t*)&newa;
            uint32_t newb_ = *(uint32_t*)&newb;
            uint32_t gold_ = *(uint32_t*)&gold;
            uint32_t lut_ = *(uint32_t*)&lut;
            if(gold_ != lut_) {
                uint32_t gold_exp = (gold_&EXPONENT_MASK_) >> 23;
                uint32_t lut_exp = (lut_&EXPONENT_MASK_) >> 23;
                uint32_t new_a_exp = ((newa_&EXPONENT_MASK_)>>23);
                uint32_t new_b_exp = ((newb_&EXPONENT_MASK_)>>23);
                if(new_a_exp == 0 or new_b_exp == 0) continue;
                uint32_t gold_mant = (gold_&mant_mask);
                uint32_t lut_mant = (lut_&mant_mask);
                uint32_t sign_diff = (gold_&SIGN_MASK_) ^ (lut_&SIGN_MASK_);
                if(sign_diff)sign_diff++;
                if(gold_exp!=lut_exp){exp_diff++;}//std::cout << "new_a exp: "  << ((newa_&EXPONENT_MASK_)>>23) << "new_b exp: " << ((newb_&EXPONENT_MASK_)>>23) <<  "gold_exp: "<< gold_exp << " lut_exp: "<< lut_exp << "\n";}
                if(gold_mant!=lut_mant){mant_diff++;}//std::cout << "gold_mant: "<< gold_mant << " lut_mant: "<< lut_mant << "\n";}
                //std::cout << "\n";

                diff++;
            }
        }
    }
    std::cout << "difference ratio " << float(diff)/float(pow(2,28)) << std::endl;
    std::cout << "exp difference ratio " << float(exp_diff)/float(pow(2,28)) << std::endl;
    std::cout << "mant difference ratio " << float(mant_diff)/float(pow(2,28)) << std::endl;
    std::cout << "sign difference ratio " << float(sign_diff)/float(pow(2,28)) << std::endl;
    return 0;
}

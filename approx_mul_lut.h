#ifndef APPROX_MUL_LUT_H_
#define APPROX_MUL_LUT_H_
#include <tensorflow/core/framework/op_kernel.h>
#include <fstream>
typedef unsigned long long cudaTextureObject_t;
class approx_mul_lut_base {
    public:
        explicit approx_mul_lut_base(tensorflow::OpKernelConstruction* context) :
            mant_mul_lut_{0}, exp_mul_lut_{0} {
            load_lut_binary();
        }
        virtual ~approx_mul_lut_base() = default;
        // same for both CPU and GPU
        auto load_lut_binary() -> void {
            // open mant mul file
            std::ifstream file("mbmmantmul16.bin", std::ios::in | std::ios::binary);
            if(file.fail()) {
                std::cerr << "file mbmmantmul16.bin failed" << std::endl;
                exit(1);
            } 
            if(!file.is_open()) { 
                std::cerr << "file mbmmantmul16.bin open failed" << std::endl;
                exit(1);
            }
            mant_mul_lut_.resize(128*128);
            file.read(
                    reinterpret_cast<char *>(mant_mul_lut_.data()), 
                    mant_mul_lut_.size() * sizeof(uint32_t)
                );
            // open exponent file
            std::ifstream exp_file("exp.bin", std::ios::in|std::ios::binary);
            if(exp_file.fail()) {
                std::cerr << "file exp.bin failed" << std::endl;
                exit(1);
            } 
            if(!exp_file.is_open()) { 
                std::cerr << "file exp.bin open failed" << std::endl;
                exit(1);
            }
            exp_mul_lut_.resize(2*2*256);
            exp_file.read(
                    reinterpret_cast<char *>(exp_mul_lut_.data()),
                    exp_mul_lut_.size() * sizeof(uint32_t)
                    );
        }
        auto get_mant_mul_lut_text_() -> cudaTextureObject_t& {
            return mant_mul_lut_text_;
        }
        auto get_exp_mul_lut_text_() -> cudaTextureObject_t& {
            return exp_mul_lut_text_;
        }
    protected:
        std::vector<uint32_t> mant_mul_lut_;
        std::vector<uint32_t> exp_mul_lut_;
        uint32_t* mant_mul_lut_cuda_;
        uint32_t* exp_mul_lut_cuda_;
        cudaTextureObject_t mant_mul_lut_text_;
        cudaTextureObject_t exp_mul_lut_text_;
};
template <typename Device>
class approx_mul_lut : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(tensorflow::OpKernelConstruction *context);
        ~approx_mul_lut();
};
#endif

#ifndef APPROX_MUL_LUT_H_
#define APPROX_MUL_LUT_H_
#include <tensorflow/core/framework/op_kernel.h>
#include <fstream>
#include <string>
typedef unsigned long long cudaTextureObject_t;
class approx_mul_lut_base {
    public:
        explicit approx_mul_lut_base(tensorflow::OpKernelConstruction* context) :
            mant_mul_lut_{0} {
            load_lut_binary(context);
        }
        virtual ~approx_mul_lut_base() = default;
        // same for both CPU and GPU
        auto load_lut_binary(tensorflow::OpKernelConstruction* context) -> void {
            auto mant_lut_file_name = std::string{};
            OP_REQUIRES_OK(context, context->GetAttr("mant_mul_lut", &mant_lut_file_name));
            if(mant_lut_file_name.empty()){
                std::cerr << "no mant lut file name given" << std::endl;
                exit(1);
            }
            unsigned start_delimiter = mant_lut_file_name.find_last_of("_");
            unsigned stop_deliminter = mant_lut_file_name.find_last_of(".");
            auto mant_width_str = mant_lut_file_name.substr(start_delimiter+1, stop_deliminter - start_delimiter - 1);
            mant_width = std::stoi(mant_width_str);
            a_shift = 16 - mant_width;
            b_shift = 23 - mant_width;
            mant_mask = ((2 << mant_width) - 1) << (23 - mant_width);
            // open mant mul file
            std::ifstream file(mant_lut_file_name, std::ios::in | std::ios::binary);
            if(file.fail()) {
                std::cerr << "lut file read failed" << std::endl;
                exit(1);
            } 
            if(!file.is_open()) { 
                std::cerr << "lut file open failed" << std::endl;
                exit(1);
            }
            mant_mul_lut_.resize(2<<(mant_width*2));
            file.read(
                    reinterpret_cast<char *>(mant_mul_lut_.data()), 
                    mant_mul_lut_.size() * sizeof(uint32_t)
                );
        }
        auto get_mant_mul_lut_text_() -> cudaTextureObject_t& {
            return mant_mul_lut_text_;
        }
        auto get_mant_mul_lut_() -> uint32_t* {
            return mant_mul_lut_cuda_;
        }
        auto get_mant_mask_() -> int {
            return mant_mask;
        };
        auto get_a_shift_() -> int {
            return a_shift;
        };
        auto get_b_shift_() -> int {
            return b_shift;
        };
    protected:
        std::vector<uint32_t> mant_mul_lut_;
        uint32_t* mant_mul_lut_cuda_;
        cudaTextureObject_t mant_mul_lut_text_;
        std::string lut_file_name;
        int mant_width;
        uint32_t mant_mask;
        int a_shift;
        int b_shift;
};
template <typename Device>
class approx_mul_lut : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(tensorflow::OpKernelConstruction *context);
        ~approx_mul_lut();
};
#endif

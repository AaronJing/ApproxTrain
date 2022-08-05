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
            mant_width_ = std::stoi(mant_width_str);
            a_shift_ = 23 - mant_width_*2;
            b_shift_ = 23 - mant_width_;
            mant_mask_ = ((1 << mant_width_) - 1) << (23 - mant_width_);
            
            
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
            mant_mul_lut_.resize(uint32_t(pow(2,mant_width_*2)));
            file.read(
                    reinterpret_cast<char *>(mant_mul_lut_.data()), 
                    mant_mul_lut_.size() * sizeof(uint32_t)
                );
        }
        auto get_mant_mul_lut_text_() -> cudaTextureObject_t& {
            return mant_mul_lut_text_;
        }
        auto get_mant_mul_lut_() -> uint8_t* {
            return mant_mul_lut_cuda_;
        }
        auto get_mant_mask_() -> uint32_t {
            return mant_mask_;
        };
        auto get_a_shift_() -> uint8_t {
            return a_shift_;
        };
        auto get_b_shift_() -> uint8_t {
            return b_shift_;
        };
        auto get_mant_width_() ->uint8_t {
            return mant_width_;
        };
    protected:
        std::vector<uint8_t> mant_mul_lut_;
        uint8_t* mant_mul_lut_cuda_;
        cudaTextureObject_t mant_mul_lut_text_;
        std::string lut_file_name;
        uint8_t mant_width_;
        uint32_t mant_mask_;
        uint8_t a_shift_;
        uint8_t b_shift_;
};
template <typename Device>
class approx_mul_lut : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(tensorflow::OpKernelConstruction *context);
        ~approx_mul_lut();
};
#endif

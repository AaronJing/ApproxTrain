__device__ float bit16masking(float num){
    const uint32_t mask = 0xffff0000;
    uint32_t b = *(int*)&num;
    uint32_t masked = b&mask;
    float ret  = *(float*)&masked;
    return ret;
}

__device__ float bfloat16mul(float a, float b) {
    return bit16masking(bit16masking(a)*bit16masking(b));
}

float bit16masking(float num) {
    int mask = 0xffff0000;
    int b = *(int*)&num;
    int masked = b&mask;
    float ret  = *(float*)&masked;
    return ret;
}

float bfloat16mul(float a, float b) {
    return bit16masking(bit16masking(a)*bit16masking(b));
}

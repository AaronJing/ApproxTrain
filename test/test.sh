
#!/bin/bash

# terminate script
die() {
	echo "$1" >&2
	echo
	exit 1
}

rm -f test/data/*.txt
rm -f test/*.o
rm -f test/*.out


echo "Accurate multiply "
nvcc -O2 -I cuda/ test/test_gemm.cu cuda/gemm.cu  -o test/gemm_ac.out
test/gemm_ac.out > test/data/gemm_ac.txt && diff -q --strip-trailing-cr test/data/ac_ref_32bit.txt.exp test/data/gemm_ac.txt || die "Accurate multiply failed"

echo ""
echo "FMBM 32 multiply"
nvcc -O2 -I cuda/ -DFMBM32_MULTIPLIER=1 test/test_gemm.cu cuda/gemm.cu  -o test/gemm_am32.out
test/gemm_am32.out > test/data/gemm_am32.txt && diff -q --strip-trailing-cr test/data/am_ref_32bit.txt.exp test/data/gemm_am32.txt|| die "FMBM 32 multiply failed"

echo ""
echo "FMBM 16 multiply"
nvcc -O2 -I cuda/ -DFMBM16_MULTIPLIER=1 test/test_gemm.cu cuda/gemm.cu -o test/gemm_am16.out
test/gemm_am16.out > test/data/gemm_am16.txt && diff -q --strip-trailing-cr test/data/am_ref_16bit.txt.exp test/data/gemm_am16.txt|| die "FMBM 16 multiply failed"

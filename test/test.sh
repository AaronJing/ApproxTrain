
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
test/gemm_ac.out > test/data/gemm_ac.txt && diff -q -w test/data/ac_ref_32bit.txt.exp test/data/gemm_ac.txt || die "Accurate multiply failed"

echo ""
for i in 32 16 14 12 10
do
	echo "FMBM $i multiply"
	nvcc -O2 -I cuda/ -DFMBM${i}_MULTIPLIER=1 test/test_gemm.cu cuda/gemm.cu  -o test/gemm_am${i}.out
	test/gemm_am${i}.out > test/data/gemm_am${i}.txt && diff -q -w test/data/am_ref_${i}bit.txt.exp test/data/gemm_am${i}.txt|| die "FMBM $i multiply failed"
	echo ""
done

echo ""
for i in 32 16 14 12 10
do
	echo "Mitchel $i multiply"
	nvcc -O2 -I cuda/ -DMITCHEL${i}_MULTIPLIER=1 test/test_gemm.cu cuda/gemm.cu  -o test/gemm_am2${i}.out
	test/gemm_am2${i}.out > test/data/gemm_am2${i}.txt && diff -q -w test/data/am2_ref_${i}bit.txt.exp test/data/gemm_am2${i}.txt|| die "FMBM $i multiply failed"
	echo ""
done

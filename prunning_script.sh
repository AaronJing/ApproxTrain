
#for DATAWIDTH in 32 16 14 12 10 
#do
#    for MUL_NAME in FMBM MITCHELL
#    do
#        MUL="$MUL_NAME$DATAWIDTH""_MULTIPLIER"
#        echo $MUL
#        make clean && make MULTIPLIER=$MUL
#        MULTIPLIER=$MUL
#        ./compile.sh
#        PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
#        python3 mnist_prunning_example.py --approx True &> $PRUNNINGOUTPUT
#        echo $MUL "Done"
#    done
#done
start_time="$(date -u +%s)"
MUL="AFM16"
echo $MUL
PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
python3 mnist_prunning_example.py --mul "lut/MBM_7.bin" --approx True &> $PRUNNINGOUTPUT
echo $MUL "Done"
MUL="Bfloat16"
echo $MUL
PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
python3 mnist_prunning_example.py --mul "lut/ACC_7.bin" &> $PRUNNINGOUTPUT
echo $MUL "Done"
MUL="FP32"
echo $MUL
PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
python3 mnist_prunning_example.py &> $PRUNNINGOUTPUT
echo $MUL "Done"

grep -A 2 "sparsity" *_RAW > sparsity_acc_all
python3 prunning_plotting.py
rm *RAW
rm sparsity_acc_all
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for process"

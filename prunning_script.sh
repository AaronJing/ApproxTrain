
for DATAWIDTH in 32 16 14 12 10 
do
    for MUL_NAME in FMBM MITCHELL
    do
        MUL="$MUL_NAME$DATAWIDTH""_MULTIPLIER"
        echo $MUL
        make clean && make MULTIPLIER=$MUL
        MULTIPLIER=$MUL
        ./compile.sh
        PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
        python3 mnist_prunning_example.py --approx True &> $PRUNNINGOUTPUT
        echo $MUL "Done"
    done
done
#MUL="BFLOAT"
#echo $MUL
#make clean && make MULTIPLIER=$MUL
#MULTIPLIER=$MUL
#./compile.sh
#PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
#python3 mnist_prunning_example.py --MUL=BFLOAT --approx True &> $PRUNNINGOUTPUT
#echo $MUL "Done"
#
#MUL="TF"
#echo $MUL
#make clean && make MULTIPLIER=$MUL
#MULTIPLIER=$MUL
#./compile.sh
#PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
#python3 mnist_prunning_example.py --MUL=TF &> $PRUNNINGOUTPUT
#echo $MUL "Done"
#
#MUL="FMBM16_MULTIPLIER"
#echo $MUL
#make clean && make MULTIPLIER=$MUL
#MULTIPLIER=$MUL
#./compile.sh
#PRUNNINGOUTPUT=$MUL"_PRUNE_RAW"
#python3 mnist_prunning_example.py --MUL=MBM16 --approx True &> $PRUNNINGOUTPUT
#echo $MUL "Done"
#grep -A 2 "sparsity" *_RAW > sparsity_acc_all
#python3 prunning_plotting.py

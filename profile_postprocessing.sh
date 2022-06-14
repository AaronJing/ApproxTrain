source experiment_dataset.sh 
for j in "${dnndataset[@]}"; do
for i in "${experiment[@]}"; do
    BATCH=$(awk "/^$i$/,/^end $i$/" output_raw | grep "$j  elapsed" | wc -l)
    BATCH=$((BATCH-1))
    echo "$i $j: $(awk "/^$i$/,/^end $i$/" output_raw | grep "$j  elapsed" | tail -n $BATCH | awk -F ' ' '{print $4}' | awk '{s+=$1; n++} END {print int(s/n);}') ms"
    #awk "/$i/,/end $i/" gadi_output
done
echo ""
echo ""
done

experiment=(
	"NO OPT"
	"OPT 1lut texture"
	"OPT 2lut texture"
	"OPT 1lut global"
	"TF GPU"
	"TF CPU"
)
dnndataset=(
	"lenet31"
	"lenet5"
	"resnet18"
	"resnet34"
	"resnet50"
	"resnet50ImageNet"
)
for j in "${dnndataset[@]}"; do
for i in "${experiment[@]}"; do
    BATCH=$(awk "/$i/,/end $i/" output | grep "$j  elapsed" | wc -l)
    BATCH=$((BATCH-1))
    echo "$i $j: $(awk "/$i/,/end $i/" output | grep "$j  elapsed" | tail -n $BATCH | awk -F ' ' '{print $4}' | awk '{s+=$1; n++} END {print int(s/n);}') ms"
done
echo ""
echo ""
done

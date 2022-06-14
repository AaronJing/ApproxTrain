
source experiment_dataset.sh
for i in "${experiment[@]}"; do
    echo "$i"
    if [ "$i" != "ACCURATE" ]
    then
        MUL="FMBM16_MULTIPLIER"
    else
        MUL=""
    fi
    if [ "$i" = "OPT 1lut texture" ]
    then
        OPT="1"
    elif [ "$i" = "OPT 2lut texture" ]
    then
        OPT="2"
    elif [ "$i" = "OPT 1lut global" ]
    then
        OPT="3"
    elif [ "$i" = "OPT 1lut texture v2" ]
    then
        OPT="4"
    else
        OPT=""
    fi

    make clean && make convam MULTIPLIER=$MUL OPT=$OPT && make denseam_gpu.so MULTIPLIER=$MUL OPT=$OPT
for j in "${dnndataset[@]}"; do
    echo $j
    if [ "$i" = "TF GPU" ]
    then
        python3 profile.py --model=$j --batch-size=32 --batch-number=5 --gpu=1
    elif [ "$i" = "TF CPU" ]
    then
        python3 profile.py --model=$j --batch-size=32 --batch-number=5 --gpu=0 
    elif [ "$i" = "AM CPU" ]
    then
        python3 profile.py --model=$j --batch-size=1 --batch-number=32 --am=1 
    else
        python3 profile.py --model=$j --batch-size=32 --batch-number=5 --am=1 --gpu=1
    fi
done
    echo "end $i"
echo ""
echo ""
done

#echo "NO OPT"
#make clean && make convam MULTIPLIER=FMBM16_MULTIPLIER && make denseam_gpu.so MULTIPLIER=FMBM16_MULTIPLIER
#echo "lenet31"
#python3 profile.py --model=lenet31 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "lenet5"
#python3 profile.py --model=lenet5 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet18"
#python3 profile.py --model=resnet18 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet34"
#python3 profile.py --model=resnet34 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50"
#python3 profile.py --model=resnet50 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50ImageNet"
#python3 profile.py --model=resnet50ImageNet --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "end NO OPT"
#
#echo "OPT 1lut texture"
#make clean && make convam MULTIPLIER=FMBM16_MULTIPLIER OPT=1 && make denseam_gpu.so MULTIPLIER=FMBM16_MULTIPLIER OPT=1
#echo "lenet31"
#python3 profile.py --model=lenet31 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "lenet5"
#python3 profile.py --model=lenet5 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet18"
#python3 profile.py --model=resnet18 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet34"
#python3 profile.py --model=resnet34 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50"
#python3 profile.py --model=resnet50 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50ImageNet"
#python3 profile.py --model=resnet50ImageNet --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "end OPT 1lut texture"
#
#echo "OPT 2lut texture"
#make clean && make convam MULTIPLIER=FMBM16_MULTIPLIER OPT=2 && make denseam_gpu.so MULTIPLIER=FMBM16_MULTIPLIER OPT=2
#echo "lenet31"
#python3 profile.py --model=lenet31 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "lenet5"
#python3 profile.py --model=lenet5 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet18"
#python3 profile.py --model=resnet18 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet34"
#python3 profile.py --model=resnet34 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50"
#python3 profile.py --model=resnet50 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50ImageNet"
#python3 profile.py --model=resnet50ImageNet --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "end OPT 2lut texture"
#
#echo "OPT 1lut global"
#make clean && make convam MULTIPLIER=FMBM16_MULTIPLIER OPT=3 && make denseam_gpu.so MULTIPLIER=FMBM16_MULTIPLIER OPT=3
#echo "lenet31"
#python3 profile.py --model=lenet31 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "lenet5"
#python3 profile.py --model=lenet5 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet18"
#python3 profile.py --model=resnet18 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet34"
#python3 profile.py --model=resnet34 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50"
#python3 profile.py --model=resnet50 --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "resnet50ImageNet"
#python3 profile.py --model=resnet50ImageNet --batch-size=32 --batch-number=5 --am=1 --gpu=1
#echo "end OPT 1lut global"
#
##echo "NAVIE CPU ONLY"
##echo "lenet31"
##python3 profile.py --model=lenet31 --batch-size=32 --batch-number=5  --am=1 --gpu=0
##echo "lenet5"
##python3 profile.py --model=lenet5 --batch-size=32 --batch-number=5 --am=1 --gpu=0
##echo "resnet18"
##python3 profile.py --model=resnet18 --batch-size=32 --batch-number=5 --am=1 --gpu=0
##echo "resnet34"
##python3 profile.py --model=resnet34 --batch-size=32 --batch-number=5 --am=1 --gpu=0
##echo "resnet50"
##python3 profile.py --model=resnet50 --batch-size=32 --batch-number=5 --am=1 --gpu=0
##echo "resnet50ImageNet"
##python3 profile.py --model=resnet50ImageNet --batch-size=32 --batch-number=5 --am=1 --gpu=0 
#
#echo "TF GPU"
#echo "lenet31"
#python3 profile.py --model=lenet31 --batch-size=32 --batch-number=5  --gpu=1
#echo "lenet5"
#python3 profile.py --model=lenet5 --batch-size=32 --batch-number=5  --gpu=1
#echo "resnet18"
#python3 profile.py --model=resnet18 --batch-size=32 --batch-number=5  --gpu=1
#echo "resnet34"
#python3 profile.py --model=resnet34 --batch-size=32 --batch-number=5  --gpu=1
#echo "resnet50"
#python3 profile.py --model=resnet50 --batch-size=32 --batch-number=5  --gpu=1
#echo "resnet50ImageNet"
#python3 profile.py --model=resnet50ImageNet --batch-size=32 --batch-number=5  --gpu=1
#echo "end TF GPU"
#
#echo "TF CPU"
#echo "lenet31"
#python3 profile.py --model=lenet31 --batch-size=32 --batch-number=5  --gpu=0
#echo "lenet5"
#python3 profile.py --model=lenet5 --batch-size=32 --batch-number=5  --gpu=0
#echo "resnet18"
#python3 profile.py --model=resnet18 --batch-size=32 --batch-number=5  --gpu=0
#echo "resnet34"
#python3 profile.py --model=resnet34 --batch-size=32 --batch-number=5  --gpu=0
#echo "resnet50"
#python3 profile.py --model=resnet50 --batch-size=32 --batch-number=5  --gpu=0
#echo "resnet50ImageNet"
#python3 profile.py --model=resnet50ImageNet --batch-size=32 --batch-number=5  --gpu=0
#echo "end TF CPU"


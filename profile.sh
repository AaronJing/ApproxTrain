./gpu_compile.sh
cp convam_gpu.so test/ResNet-Tensorflow-Test/.
cd test/ResNet-Tensorflow-Test
python3 main.py --phase train --dataset cifar10 --epoch 200 --batch_size 128 --res_n 18 --acc TEST > output
grep "Im2col" output | grep -Eo '[+-]?([0-9][.][0-9]+)([e][+-][0-9]+)?' | awk '{ SUM += $1;} END { print "Forward Im2col " SUM }'
grep "Forward gemm" output | grep -Eo '[+-]?([0-9][.][0-9]+)([e][+-][0-9]+)?' | awk '{ SUM += $1;} END { print "Forward gemm " SUM }'
grep "Filter gradient" output | grep -Eo '[+-]?([0-9][.][0-9]+)([e][+-][0-9]+)?' | awk '{ SUM += $1;} END { print "Backpropagation filtergradient " SUM }'
grep "Im2Col" output | grep -Eo '[+-]?([0-9][.][0-9]+)([e][+-][0-9]+)?' | awk '{ SUM += $1;} END { print "Backpropagation error Im2Col " SUM }'
grep "Gemm inverse" output | grep -Eo '[+-]?([0-9][.][0-9]+)([e][+-][0-9]+)?' | awk '{ SUM += $1;} END { print "Backpropagation error gemm " SUM }'
cat output | grep -Eo '[+-]?([0-9][.][0-9]+)([e][+-][0-9]+)?' | awk '{ SUM += $1;} END { print "Total " SUM }'
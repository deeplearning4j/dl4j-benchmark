#!/usr/bin/env bash

dl4j_path="${dl4j_path:-"org.deeplearning4j.ModelCompare.dl4j.Dl4j_1Main"}"
caffe_path="${caffe_path:-"dl4j-core-benchmark/src/main/java/org/deeplearning4j/ModelCompare/caffe/"}"
tf_path="${tf_path:-"dl4j-core-benchmark/src/main/java/org/deeplearning4j/ModelCompare/tensorflow/tensorflow-1main.py"}"
torch_path="${torch_path:-"dl4j-core-benchmark/src/main/java/org/deeplearning4j/ModelCompare/torch/torch-1main.lua"}"
dl4j_warning="Make sure pom nd4j-backend set to cuda-7.5"

model_type='lenet'
model_type='mlp'

read -p "dl4j, caffe, tensorflow, torch? " platform
case $platform in
'dl4j')
    read -p "CPU, GPU, or MULTI? " dl
    case $dl in
        'CPU' ) time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar $dl4j_path --modelType $model_type;;
        'GPU' ) echo $dl4j_warning && time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar $dl4j_path --modelType $model_typet;;
        'MULTI' ) echo $dl4j_warning && time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar $dl4j_path --modelType $model_type -nGPU 12;;
         *) echo "Invalid response";;
    esac;;

'caffe')
    if [ model_type -eq "mlp" ]; then
        caffe_path=$caffe_path"caffe_mlp_solver.prototxt"
    else
        caffe_path=$caffe_path"caffe_lenet_solver.prototxt"
    fi
    read -p "CPU, GPU, or MULTI? " yn
    case $yn in
        'CPU' ) time $HOME/caffe/build/tools/caffe train --solver=$caffe_path;;
        'GPU' ) time $HOME/caffe/build/tools/caffe train --solver=$caffe_path -gpu 0;;
        'MULTI' ) time $HOME/caffe/build/tools/caffe train --solver=$caffe_path -gpu all;;
        *) echo "Invalid response";;
    esac;;

'tensorflow')
    read -p "CPU, GPU, or MULTI? " yn
    case $yn in
        'CPU' ) time python $tf_path --model_type $model_type;;
        'GPU' ) time python $tf_path --core_type "GPU"  --model_type $model_type --num_gpus 1;;
        'MULTI' ) time python $tf_path --core_type "MULTI" --model_type $model_type  --num_gpus 4;;
        *) echo "Invalid response";;
    esac;;

'torch')
    read -p "CPU, GPU, or MULTI? " model
    case $model in
        'CPU ' ) time th $torch_path -model_type $model_type;;
        'GPU' ) time th $torch_path -gpu -model_type $model_type;;
        'MULTI' ) time th $torch_path -gpu -multi -model_type $model_type;;
        *) echo "Invalid response";;
    esac;;
esac

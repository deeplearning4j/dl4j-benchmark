#!/usr/bin/env bash

# This script: launches the simple benchmark and captures the snapshots automatically

cd ..
rootDir=$(pwd)
#mvn clean package -DskipTests -Pv091,cudnn8
#mvn clean package -DskipTests -Pv100beta,cudnn8
#mvn clean package -DskipTests -Pv100beta,cudnn91
mvn clean package -DskipTests -Pv100snapshot,cudnn91
cd dl4j-core-benchmark
mkdir -p ../scripts/SimpleBenchmark

#declare -a versionBackend=("v091_cuda8-cudnn" "v100beta_cuda8-cudnn")
declare -a versionBackend=("v100beta_cuda91-cudnn")
declare -a batchSize=("32")

#model=ALEXNET
model=RESNET50PRE
xmx=16G
javacpp=16G

for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      export OUTPUT_DIR=../scripts/SimpleBenchmark/$model/opProfiler
      mkdir -p $OUTPUT_DIR
      echo "Running test: $i, batch size $j"
      echo java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.simple.SimpleBenchmark --nIter 100 --fit --minibatch $j --model $model --profile > $OUTPUT_DIR/output_"$i"_"$j".txt
      java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.simple.SimpleBenchmark --nIter 100 --fit --minibatch $j --model $model --profile >> $OUTPUT_DIR/output_"$i"_"$j".txt
   done
done



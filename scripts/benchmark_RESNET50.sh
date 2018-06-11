#!/usr/bin/env bash
cd ..
#mvn package -DskipTests -Pv091,cudnn8
#mvn package -DskipTests -Pv100alpha,cudnn8
#mvn package -DskipTests -Pv100beta,cudnn8
#mvn package -DskipTests -Pv100beta,cudnn91
cd dl4j-core-benchmark
#declare -a versionBackend=("v091_cuda8-cudnn" "v100alpha_cuda8-cudnn" "v100beta_cuda8-cudnn")
declare -a versionBackend=("v100beta_cuda91-cudnn")
declare -a batchSize=("32")
declare -a dataTypes=("FLOAT" "HALF")
modelType=RESNET50
xmx=16G
javacpp=16G
mkdir -p ../scripts/$modelType
## now loop through the above array
for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      for k in "${dataTypes[@]}"
      do
         echo "Running test: $i, batch size $j, data type $k"
         echo java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --datatype $k --cacheMode NONE > ../scripts/$modelType/output_"$i"_"$j"_"$k".txt
         java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --datatype $k --cacheMode NONE >> ../scripts/$modelType/output_"$i"_"$j"_"$k".txt
      done
   done
done

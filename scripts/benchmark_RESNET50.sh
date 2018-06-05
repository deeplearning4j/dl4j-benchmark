#!/usr/bin/env bash
cd ..
#mvn package -DskipTests -Pv091,cudnn8
#mvn package -DskipTests -Pv100alpha,cudnn8
#mvn package -DskipTests -Pv100beta,cudnn8
mvn package -DskipTests -Pv100beta,cudnn91
cd dl4j-core-benchmark
#declare -a versionBackend=("v091_cuda8-cudnn" "v100alpha_cuda8-cudnn" "v100beta_cuda8-cudnn")
declare -a versionBackend=("v100beta_cuda91-cudnn")
declare -a batchSize=("16" "32")
modelType=RESNET50
xmx=12G
javacpp=12G
mkdir -p ../scripts/$modelType
## now loop through the above array
for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      echo "Running test: $i, batch size $j"
      echo java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --cacheMode NONE > ../scripts/$modelType/output_"$i"_"$j".txt
      java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --cacheMode NONE >> ../scripts/$modelType/output_"$i"_"$j".txt
   done
done

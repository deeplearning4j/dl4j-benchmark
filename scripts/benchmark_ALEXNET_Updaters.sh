#!/usr/bin/env bash
cd ..
#mvn package -DskipTests -Pv091,cudnn8
#mvn package -DskipTests -Pv100alpha,cudnn8
mvn package -DskipTests -Pv100beta,cudnn8
cd dl4j-core-benchmark
declare -a versionBackend=("v100beta_cuda8-cudnn")
declare -a batchSize=("32")
declare -a updaters=("NONE", "SGD", "ADAM", "ADAMAX", "ADADELTA", "NESTEROVS", "ADADELTA", "RMSPROP",
modelType=ALEXNET
mkdir -p ../scripts/$modelType
## now loop through the above array
for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
       for k in "${updaters[@]}"
       do
          echo "Running test: $i, batch size $j, updater $k"
          echo java -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --cacheMode NONE --updater $k > ../scripts/$modelType/output_"$i"_"$j"_"$k".txt
          java -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j  --updater $k --cacheMode NONE >> ../scripts/$modelType/output_"$i"_"$j"_"$k".txt
       done
   done
done

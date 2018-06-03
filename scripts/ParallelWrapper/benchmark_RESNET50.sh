#!/usr/bin/env bash
cd ../..
mvn package -DskipTests -Pv100beta,cudnn91
cd dl4j-core-benchmark
declare -a versionBackend=("v100beta_cuda91-cudnn")
declare -a batchSize=("16" "32")
modelType=RESNET50
mkdir -p ../scripts/ParallelWrapper/$modelType
## now loop through the above array
for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      echo "Running test: $i, batch size $j"
      echo java -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --cacheMode NONE --usePW --pwNumThreads -1 > ../scripts/ParallelWrapper/$modelType/output_"$i"_"$j".txt
      java -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --cacheMode NONE --usePW --pwNumThreads -1 >> ../scripts/ParallelWrapper/$modelType/output_"$i"_"$j".txt
   done
done

#!/usr/bin/env bash

cd ..
mvn package -DskipTests -Pv091,cudnn8
mvn package -DskipTests -Pv100beta,cudnn8
cd dl4j-core-benchmark
mkdir -p ../scripts/SimpleBenchmark

cd dl4j-core-benchmark
declare -a versionBackend=("v091_cuda8-cudnn" "v100alpha_cuda8-cudnn" "v100beta_cuda8-cudnn")
declare -a batchSize=("32")

# Launching with profiling:
# https://www.yourkit.com/docs/java/help/agent.jsp
# https://www.yourkit.com/docs/java/help/startup_options.jsp
yourkitPath=/home/alex/Downloads/YourKit-JavaProfiler-2017.02/bin/linux-x86-64/libyjpagent.so

for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      echo "Running test: $i, batch size $j"
      echo java -agentpath:$yourkitPath=tracing -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.simple.SimpleBenchmark --forward false --minibatch $j > ../scripts/SimpleBenchmark/output_"$i"_"$j".txt
      java -agentpath:$yourkitPath=tracing -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.simple.SimpleBenchmark --forward false --minibatch $j >> ../scripts/SimpleBenchmark/output_"$i"_"$j".txt
   done
done



#!/usr/bin/env bash

# This script: launches the simple benchmark and captures the snapshots automatically

cd ..
mvn clean package -DskipTests -Pv091,cudnn8
mvn clean package -DskipTests -Pv100beta,cudnn8
cd dl4j-core-benchmark
mkdir -p ../scripts/SimpleBenchmark

declare -a versionBackend=("v091_cuda8-cudnn" "v100beta_cuda8-cudnn")
declare -a batchSize=("32")

# Launching with profiling:
# https://www.yourkit.com/docs/java/help/agent.jsp
# https://www.yourkit.com/docs/java/help/startup_options.jsp
yourkitPath=/home/alex/Downloads/YourKit-JavaProfiler-2017.02/bin/linux-x86-64/libyjpagent.so
yourkitJar=/home/alex/Downloads/YourKit-JavaProfiler-2017.02/lib/yjp-controller-api-redist.jar

for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      export SNAPSHOT_DIR=../scripts/SimpleBenchmark/snapshot-$i-$j/
      mkdir -p $SNAPSHOT_DIR
      echo "Running test: $i, batch size $j"
      echo java -agentpath:$yourkitPath=tracing,port=10001,dir=$SNAPSHOT_DIR -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.simple.SimpleBenchmark --nIter 100 --fit --minibatch $j > ../scripts/SimpleBenchmark/output_"$i"_"$j".txt
      java -agentpath:$yourkitPath=tracing,dir=$SNAPSHOT_DIR -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.simple.SimpleBenchmark --nIter 100 --fit --minibatch $j >> ../scripts/SimpleBenchmark/output_"$i"_"$j".txt & export APP_PID=$!

      #Note: snapshot recording will automatically occur when JVM shuts down


      #Manual recording
      ##java -jar $yourkitJar localhost 10001 start-cpu-tracing #Not required - already started
      #Can't simply wait for process to finish - can't manually do dump after this as the profiling agent is shut down :/
      #tail --pid=$APP_PID -f /dev/null     #wait
      #instead: wait 25 seconds (with minRuntimeSec = 30, we should have enough time to save the snapshot before the JVM shuts down)
      #sleep 25
      #java -jar $yourkitJar localhost 10001 stop-cpu-profiling
      #java -jar $yourkitJar localhost 10001 capture-performance-snapshot

   done
done



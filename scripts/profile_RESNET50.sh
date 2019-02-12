#!/usr/bin/env bash
cd ..
#mvn package -DskipTests -Pv091,cudnn8
#mvn package -DskipTests -Pv100alpha,cudnn8
#mvn package -DskipTests -Pv100beta,cudnn8
#mvn package -DskipTests -Pv100beta3,native
#mvn package -DskipTests -Pv100snapshot,native
#mvn package -DskipTests -Pv100beta3,cuda10
#mvn package -DskipTests -Pv100snapshot,cuda10
#mvn package -DskipTests -Pv100beta3,cudnn10
#mvn package -DskipTests -Pv100snapshot,cudnn10
cd dl4j-core-benchmark
#declare -a versionBackend=("v100beta3_native" "v100snapshot_native" "v100beta3_cuda10" "v100snapshot_cuda10" "v100beta3_cuda10-cudnn" "v100snapshot_cuda10-cudnn")
declare -a versionBackend=("v100snapshot_native" "v100beta3_native")
declare -a batchSize=("32")
declare -a dataTypes=("FLOAT")
modelType=RESNET50
xmx=16G
javacpp=16G
totalIterations=20
mkdir -p ../scripts/${modelType}_${totalIterations}_iter

# Launching with profiling:
# https://www.yourkit.com/docs/java/help/agent.jsp
# https://www.yourkit.com/docs/java/help/startup_options.jsp
yourkitPath=C:/PROGRA~1/YOURKI~2.02-/bin/win64/yjpagent.dll
yourkitJar=C:/PROGRA~1/YOURKI~2.02-/lib/yjp-controller-api-redist.jar

profilingSettingsPath=$rootDir/scripts/profiling/profiling_settings.txt

## now loop through the above array
for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      for k in "${dataTypes[@]}"
      do
         echo "Running test: $i, batch size $j, data type $k"
		 export SNAPSHOT_DIR=../scripts/${modelType}_${totalIterations}_iter/profileOutput_"$i"_"$j"_"$k"
         mkdir -p $SNAPSHOT_DIR
         echo java -agentpath:"$yourkitPath"=tracing,port=10001,dir=$SNAPSHOT_DIR,tracing_settings_path=$profilingSettingsPath -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --datatype $k --cacheMode NONE --totalIterations $totalIterations > ../scripts/${modelType}_${totalIterations}_iter/output_"$i"_"$j"_"$k".txt
         java -agentpath:"$yourkitPath"=tracing,port=10001,dir=$SNAPSHOT_DIR,tracing_settings_path=$profilingSettingsPath -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --datatype $k --cacheMode NONE --totalIterations $totalIterations >> ../scripts/${modelType}_${totalIterations}_iter/output_"$i"_"$j"_"$k".txt
      done
   done
done

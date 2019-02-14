#!/usr/bin/env bash
cd ..
#mvn package -DskipTests -Pv100beta3,native
#mvn package -DskipTests -Pv100snapshot,native
cd dl4j-core-benchmark
#declare -a versionBackend=("v100snapshot_native" "v100beta3_native")
declare -a versionBackend=("v100snapshot_native")

# Launching with profiling:
# https://www.yourkit.com/docs/java/help/agent.jsp
# https://www.yourkit.com/docs/java/help/startup_options.jsp
yourkitPath=C:/PROGRA~1/YOURKI~1.02-/bin/win64/yjpagent.dll
yourkitJar=C:/PROGRA~1/YOURKI~1.02-/lib/yjp-controller-api-redist.jar
profilingSettingsPath=$rootDir/scripts/profiling/profiling_settings.txt

xmx=16G
javacpp=16G

## now loop through the above array
for i in "${versionBackend[@]}"
do
     echo "Running test: $i"
	 export SNAPSHOT_DIR=../scripts/nd4j_profile_${i}/
         mkdir -p $SNAPSHOT_DIR
     echo java -agentpath:"$yourkitPath"=tracing,port=10001,dir=$SNAPSHOT_DIR,tracing_settings_path=$profilingSettingsPath -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.nd4j.SmallArrayBenchmark > ../scripts/nd4j_smallarray_profile_${i}.txt
     java -agentpath:"$yourkitPath"=tracing,port=10001,dir=$SNAPSHOT_DIR,tracing_settings_path=$profilingSettingsPath -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.nd4j.SmallArrayBenchmark > ../scripts/nd4j_smallarray_profile_${i}.txt
done

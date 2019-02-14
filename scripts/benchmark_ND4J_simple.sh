#!/usr/bin/env bash
cd ..
mvn package -DskipTests -Pv100beta3,native
mvn package -DskipTests -Pv100snapshot,native
cd dl4j-core-benchmark
declare -a versionBackend=("v100snapshot_native" "v100beta3_native")
#declare -a versionBackend=("v100snapshot_native")
xmx=16G
javacpp=16G
## now loop through the above array
for i in "${versionBackend[@]}"
do
     echo "Running test: $i"
     echo java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.nd4j.SimpleOpBenchmarks > ../scripts/nd4j_${i}.txt
     java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.nd4j.SimpleOpBenchmarks > ../scripts/nd4j_${i}.txt
done

FROM dl4j-base:1.0.0 as builder

COPY / /app/

RUN cd /app && mvn install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true -P v100snapshot,native -pl "!samediff-benchmark,!dl4j-core-benchmark"
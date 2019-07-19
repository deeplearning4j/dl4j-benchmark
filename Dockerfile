FROM dl4j-base:1.0.0 as builder

COPY / /app/

RUN cd /app && mvn -T 4 install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
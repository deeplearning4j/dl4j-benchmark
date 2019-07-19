FROM ubuntu:bionic

RUN apt-get update && \
    apt-get -y --no-install-recommends install \
            wget vim inetutils-ping git scala openjdk-11-jre openjdk-11-jdk-headless maven gcc g++ automake cmake build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev libgomp1 libprotobuf-dev && \
    apt-get clean

# eventually change this to eclipse repo
RUN cd / && git clone --depth 1 https://github.com/SkymindIO/deeplearning4j.git
RUN cd /deeplearning4j/libnd4j && ./buildnativeoperations.sh -a native -j 8
RUN cd /deeplearning4j/nd4j && mvn install -T 2 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd /deeplearning4j/datavec && mvn install -T 4 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd /deeplearning4j/deeplearning4j && mvn install -T 4 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd /deeplearning4j/arbiter && mvn install -T 4 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
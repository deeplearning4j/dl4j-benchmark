FROM ubuntu:bionic

RUN apt-get update && \
    apt-get -y --no-install-recommends install \
            wget vim inetutils-ping git unzip scala openjdk-11-jre openjdk-11-jdk-headless maven gcc g++ automake cmake build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev libgomp1 libprotobuf-dev && \
    apt-get clean


# download datasets
RUN wget https://deeplearning4jblob.blob.core.windows.net/datasets/tinyimagenet_200_dl4j.v1.zip && mkdir -p ~/.deeplearning4j/data/TINYIMAGENET_200 && unzip tinyimagenet_200_dl4j.v1.zip -d ~/.deeplearning4j/data/TINYIMAGENET_200
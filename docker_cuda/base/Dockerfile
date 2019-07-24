FROM nvidia/cuda:10.1-devel-ubuntu16.04

RUN apt-get update && \
    apt-get -y --no-install-recommends install \
            wget vim inetutils-ping git unzip scala default-jre default-jdk-headless maven gcc g++ automake cmake build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev libgomp1 libprotobuf-dev && \
    apt-get clean


# download datasets
RUN wget https://deeplearning4jblob.blob.core.windows.net/datasets/tinyimagenet_200_dl4j.v1.zip && mkdir -p ~/.deeplearning4j/data/TINYIMAGENET_200 && unzip tinyimagenet_200_dl4j.v1.zip -d ~/.deeplearning4j/data/TINYIMAGENET_200

# upgrade cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0.tar.gz && tar -xzf cmake-3.15.0.tar.gz && cd cmake-3.15.0 && ./configure --prefix=/usr && make all -j 8 && make install -j 8


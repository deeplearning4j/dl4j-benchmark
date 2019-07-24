#!/usr/bin/env bash

set -e

cp Dockerfile ../

# base image: java, cpp and full DL4j stack
docker build -t dl4j-cuda-base:1.0.0 ./base

# base DL4j image: stack
docker build --no-cache -t dl4j-base:1.0.0 ./deeplearning4j-base

# benchmark image: contains base dl4j image + dl4j-benchmarking jars
docker build -t dl4j-benchmark-base:1.0.0 ../

# dl4j memory benchmark containers
docker build -t dl4j-benchmark-memory-conv2d:1.0.0 ./benchmark-memory-dl4j-conv2d

# dl4j pw memory benchmark containers
#docker build -t dl4j-benchmark-memory-pw:1.0.0 ./benchmark-memory-dl4j-pw

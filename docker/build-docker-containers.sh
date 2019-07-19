#!/usr/bin/env bash

set -e

# base image: java, cpp and full DL4j stack
docker build -t dl4j-linux-base:1.0.0 ./base

# base DL4j image: stack
docker build -t dl4j-base:1.0.0 ./deeplearning4j-base

# benchmark image: contains base dl4j image + dl4j-benchmarking jars
docker build -t dl4j-benchmark-base:1.0.0 ../

# dl4j memory benchmark containers
docker build -t dl4j-benchmark-memory-conv2d:1.0.0 ./benchmark-memory-conv2d
docker build -t dl4j-benchmark-memory-rnn:1.0.0 ./benchmark-memory-rnn

# samediff memory benchmark container
#docker build -t dl4j-benchmark-memory-sd:1.0.0 ./benchmark-memory-sd
#!/usr/bin/env bash

set -e

# base image: java, cpp and full DL4j stack
docker build --no-cache -t dl4j-base:1.0.0 ./base

# benchmark image: contains base dl4j image + dl4j-benchmarking jars
docker build -t dl4j-benchmark-base:1.0.0 ../

# dl4j memory benchmark container
docker build -t dl4j-benchmark-memory-dl4j:1.0.0 ./benchmark-memory-dl4j

# samediff memory benchmark container
#docker build -t dl4j-benchmark-memory-sd:1.0.0 ./benchmark-memory-sd
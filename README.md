# Benchmarks
Repository to track Dl4j benchmarks in relation to well known architectures on CPUs and GPUs.

#### Purpose
These benchmarks are designed to show the comparison performance between CPUs and GPUs.
We tested various sort of neural networks including CNNs, RNNs, MLP and also tested MultiLayerNetwork and ComputationGraph as well. for more details, you can refer to Benchmark Details below.

#### How to run the benchmarks
```sh
# build with cudnn8.0 (for cpus, -P native, for gpus -P cuda8)
$ mvn clean package -DskipTests -P cudnn8

# run VGG16 benchmark for 16x3x224x224 input
$ java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.benchmarks.BenchmarkTinyImageNet --modelType VGG16 -w 224 -h 224 -c 3 -b 16
```

#### Benchmark Environment

| Device | DGX-1 |
| ---------- |:-----:|
| Operating System  | GNU/Linux Ubuntu 14.04.4 LTS |
| CPU  | Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz |
| CPU Cores | 80 |
| BLAS Vendor for CPU | OPENBLAS |
| GPU (#)  | Tesla P100-SXM2-16GB (8) |
| BLAS Vendor for GPU  | CUBLAS, CUDA 8.0 |
| CUDNN | v5.1|
| DL4J Version | 0.8.0 |

#### Benchmark Details

[**MLP**](#mlp)
- Input : 64x784
- Total Params : 1796010
- Total Layers : 3

|                       | CPU       | GPU       | Multi   |
| --------------------- |:---------:| ---------:| -------:|
| Avg Feedforward (ms)  | 14.32     | 1.95      |
| Avg Backprop (ms)     | 24.8      | 2.48      |
| Avg Iteration (ms)    | 50.51     | 15.21     |
| Avg Samples/sec       | 1429.2    | 11491.08  | 23389.24|
| Avg Batches/sec       | 22.33     | 179.55    | 366.79  |

[**LeNet**](#lenet)
- Input : 64x1x28x28
- Total Params : 431080
- Total Layers : 6

|                       | CPU       | GPU       | Multi   |
| --------------------- |:---------:| ---------:| -------:|
| Avg Feedforward (ms)  | 44.21     | 3.29      |
| Avg Backprop (ms)     | 94.32     | 6.21      |
| Avg Iteration (ms)    | 170.35    | 16.72     |
| Avg Samples/sec       | 420.51    | 10280.08  | 20531.8 |
| Avg Batches/sec       | 6.57      | 160.63    | 321.14  |

[**LSTM**](#lstm)
- Input : 64x300x256
- Total Params : 571650
- Total Layers : 2

|                       | CPU       | GPU       | Multi   |
| --------------------- |:---------:| ---------:| -------:|
| Avg Feedforward (ms)  | 825.28    | 233.66    |         |
| Avg Backprop (ms)     | 2820.08   | 792.96    |         |
| Avg Iteration (ms)    | 5905.69   | 1285      |         |
| Avg Samples/sec       | 11.46     | 49.34     | 189.96  |
| Avg Batches/sec       | 0.18      | 0.77      | 2.97    |


[**AlexNet**](#alexnet)
- Input : 32x3x224x224
- Total Params : 59100744
- Total Layers : 13

|                       | CPU       | GPU       | Multi   |
| --------------------- |:---------:| ---------:| -------:|
| Avg Feedforward (ms)  | 812.67    | 307.46    |
| Avg Backprop (ms)     | 2083.62   | 1105.46   |
| Avg Iteration (ms)    | 3710.5    | 2335.57   |
| Avg Samples/sec       | 8.59      | 13.52     | 52.39   |
| Avg Batches/sec       | 0.27      | 0.42      | 1.64    |


[**INCEPTIONRESNETV1**](#inceptionv1)
- Input : 32x3x160x160
- Total Params : 16003768
- Total Layers : 301

|                       | CPU       | GPU       | Multi   |
| --------------------- |:---------:| ---------:| -------:|
| Avg Feedforward (ms)  | 2806.36   | 62.98     |         |
| Avg Backprop (ms)     | 10426.49  | 205.49    |         |
| Avg Iteration (ms)    | 17373.15  | 582.21    |         |
| Avg Samples/sec       | 1.85      | 56.07     | 148.72  |
| Avg Batches/sec       | 0.06      | 1.75      | 4.63    |

[**VGG16**](#vgg16)
- Input : 32x3x224x224
- Total Params : 135079944
- Total Layers : 21

|                       | CPU       | GPU       | Multi   |
| --------------------- |:---------:| ---------:| -------:|
| Avg Feedforward (ms)  | 14452.92  | 1245.66   |         |
| Avg Backprop (ms)     | 40445.12  | 2834.26   |         |
| Avg Iteration (ms)    | 52013.42  | 6299.24   |         |
| Avg Samples/sec       | 0.52      | 5.03      | 13.1    |
| Avg Batches/sec       | 0.02      | 0.16      | 0.4     |




#### APPENDIX

<a name="mlp">**MLP**</a>
- Implementation Code : [here](https://github.com/deeplearning4j/dl4j-benchmark/blob/dh_nvidiakr/dl4j-core-benchmark/src/main/java/org/deeplearning4j/models/mlp/SimpleMLP.java)
- Network Summary : [here](https://gist.github.com/kepricon/622fc5f6131b2f6fdbf02e755bcb0d7b)

<a name="lenet">**LeNet**</a>
- Implementation Code : [here](https://github.com/deeplearning4j/dl4j-benchmark/blob/dh_nvidiakr/dl4j-core-benchmark/src/main/java/org/deeplearning4j/models/cnn/LeNet.java)
- Network Summary : [here](https://gist.github.com/kepricon/86f76610dbf6c8f629c53a6d1cbccc8e)

<a name="lstm">**LSTM**</a>
- Implementation Code : [here](https://github.com/deeplearning4j/dl4j-benchmark/blob/dh_nvidiakr/dl4j-core-benchmark/src/main/java/org/deeplearning4j/models/rnn/W2VSentiment.java)
- Network Summary : [here](https://gist.github.com/kepricon/8637248febfa41350f89643695ba6a1b)

<a name="alexnet">**AlexNet**</a>
- Implementation Code : [here](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/models/cnn/AlexNet.java)
- Reference: https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf
- Network Summary : [here](https://gist.github.com/kepricon/f3184026890024bc44f73da22d1fed27)

<a name="vgg16">**VGG16**</a>
- Implementation Code : [here](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/models/cnn/VGG16.java)
- Reference: https://arxiv.org/pdf/1409.1556.pdf
- Network Summary : [here](https://gist.github.com/kepricon/3ae1776656432382cf02c1c8f110c98d)

<a name="inceptionv1">**INCEPTIONRESNETV1**</a>
- Implementation Code : [here](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/models/cnn/InceptionResNetV1.java)
- Reference: https://arxiv.org/abs/1503.03832
- Network Summary : [here](https://gist.github.com/kepricon/1dad994a6bfaa79dcbf5903870f1d187)

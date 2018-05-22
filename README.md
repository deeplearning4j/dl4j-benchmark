# Deeplearning4j Benchmarks

Benchmarks popular models and configurations on Deeplearning4j, and output performance and versioning statistics.

#### Core Benchmarks

Available benchmarks:
* Performance benchmarks:
    * CNN benchmarks: [Link](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/benchmarks/BenchmarkCnn.java) - benchmarks for a
    number of CNN models on random data.
    * MLP/RNN Benchmarks [Link](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/benchmarks/BenchmarkMlpRnn.java) - benchmarks
    for some simple MLP and RNN models on random data.
    * BenchmarkCustom: [Link](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/benchmarks/BenchmarkCustom.java) - benchmarks
    for CNN models with custom image data
* Memory benchmarks:
    * CNN memory benchmarks: [Link](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/memory/BenchmarkCnnMemory.java) - used
    to measure the memory requirements of CNN inference and training.



## Running Benchmarks

Multiple version of DL4J can be benchmarked in this repo using Maven profiles:

* 0.9.1 (profile name: v091)
* 1.0.0-alpha (profile name: v100alpha)
* Master/snapshots (profile name: v100snapshot)

Furthermore, multiple backends can be configured:
* Native (profile name: native)
* CUDA 8 (profile name: cuda8)
* CUDA 9.1 (profile name: cuda91 - can only be used with 1.0.0-alpha and master/snapshots)
* CUDA 8 with CUDNN (profile name: cudnn8)
* CUDA 9.1 with CUDNN (profile name: cuda91 - can only be used with 1.0.0-alpha and master/snapshots)

These Maven profiles allow any supported combinations of backends and DL4J versions to be run. These are specified
at build time. You must build the repository before running benchmarks.

For example, to build the benchmark repo with support for ND4J-native backend for v0.9.1, use:

```mvn package -Pnative,v091 -DskipTests```

Similarly, to build for v1.0.0-alpha with CUDA 9.1 + CUDNN, use:

```mvn package -Pcudnn91,v100alpha -DskipTests```


Finally, to run the benchmarks, use the following:
```
mvn package -Pcudnn91,v100alpha -DskipTests
cd dl4j-core-benchmark
java -cp dl4j-core-benchmark-v100beta_native.jar org.deeplearning4j.benchmarks.BenchmarkCnn --modelType ALEXNET --batchSize 32
```
*** NOTE: The JAR file name encodes which profiles (version + backend) were used when building ***

For the full list of configuration options, see the configuration section below. 

*** NOTE: There is also a benchmark script to compare backends: see scripts/benchmark.sh ***


### Running Benchmarks in IntelliJ

In the same was as building/running through Maven, running  the benchmark repos through Intellij requires the selection
of two Maven profiles (one for the backend, one for the version). Link: [Setting Maven Profiles](https://www.jetbrains.com/help/idea/13.0/activating-and-deactivating-maven-profiles.pdf)

Additionally, IntelliJ does not properly handle the version-specific code configured using the Maven build  helper plugin.
Consequently, you will need to exclude the irrelevant directories.

For example, when running with profile ```v091``` you should exclude the ```v100alpha``` and ```v100snapshot``` directories.
You can do this by finding the directory in the project window -> right click -> Mark Directory as -> Excluded.
To switch between versions (after previously marking as excluded), switch the Maven profiles as before, then cancel the
 exclusion on the source directory, and mark that same directory as a sources root (both using the same right click menu).


## Configuring Benchmarks

Benchmarks have a number of configuration options, with defaults for most values.

Performance benchmarks:
* [modelType](https://github.com/deeplearning4j/dl4j-benchmark/blob/master/dl4j-core-benchmark/src/main/java/org/deeplearning4j/models/ModelType.java):
    - ALL
    - CNN
    - SIMPLECNN
    - ALEXNET
    - LENET
    - GOOGLELENET
    - VGG16
    - INCEPTIONRESNETV1
    - FACENETNN4
    - RNN
    - MLP_SMALL
    - RNN_SMALL
* numLabels: output size for the network
* totalIterations: Number of iterations to perform
* batchSize: Minibatch size (number of examples) for benchmarks
* gcWindow: Garbage collection frequency/window
* profile: If true, run ND4J op profiler and report results. Has considerable performance overhead, but provides a performance information on a per-operation basis
* cacheMode: DL4J CacheMode to use
* workspaceMode: DL4J WorkspaceMode to use
* updater: Updater to use (for example, NONE, ADAM, NESTEROVS, SGD, etc)

Memory benchmarks:
* modelType: As per performance benchmarks
* memoryTest: Type of test to run: ```TRAINING``` or ```INFERENCE```
* numLabels: output size for the network
* batchSizes: Minibatch sizes (note: multiple are possible) to benchmark. For multiple, use space separated: ```--batchSizes 8 16 32```
* gcWindow: Garbage collection frequency/window
* cacheMode: DL4J CacheMode to use
* workspaceMode: DL4J WorkspaceMode to use
* updater: Updater to use (for example, NONE, ADAM, NESTEROVS, SGD, etc)



## Top Benchmarks

The following benchmarks have been run using the SNAPSHOT version of DL4J 0.9.1.
This version utilizes workspace concepts and is significantly faster for inference
than 0.8.0. The number of labels used for benchmarks was 1000. Note that for full
training iteration timings, the number of labels and batch size impacts updater timing.
CUDA_VISIBLE_DEVICES has been set to 1.

### AlexNet 16x3x224x224

The AlexNet batch 16 benchmark below was developed as a comparison 
to: https://github.com/jcjohnson/cnn-benchmarks. Note that the linked benchmarks do not provide 
values for training iterations.

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  2 | 5.01  | 7.01  | 14.33  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                            ALEXNET 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      24400680
                Total Layers                                            11
        Avg Feedforward (ms)                                             2
           Avg Backprop (ms)                                          5.01
          Avg Iteration (ms)                                         14.33
             Avg Samples/sec                                       1075.93
             Avg Batches/sec                                         67.25
```

### AlexNet 128x3x224x224

The AlexNet batch 128 benchmark is a comparison to benchmarks on popular
CNNs: https://github.com/soumith/convnet-benchmarks. Note that the linked benchmarks do
not provide values for training iterations.

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  10 | 33.32  | 43.32 | 58.58  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                           ALEXNET 128x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      24400680
                Total Layers                                            11
        Avg Feedforward (ms)                                            10
           Avg Backprop (ms)                                         33.32
          Avg Iteration (ms)                                         58.58
             Avg Samples/sec                                        2098.4
             Avg Batches/sec                                         16.39
```

## LeNet 16x3x224x224

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  5 | 18.02  | 23.02 | 35.28  |

Full versioning and statistics:

```
                        Name                                         LENET
                 Description                            LENET 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      70753070
                Total Layers                                             6
        Avg Feedforward (ms)                                             5
           Avg Backprop (ms)                                         18.02
          Avg Iteration (ms)                                         35.28
             Avg Samples/sec                                        435.66
             Avg Batches/sec                                         27.23
```

## LeNet 128x3x224x224

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  28.13 | 130.4  | 158.17 | 164.24  |

Full versioning and statistics:

```
                        Name                                         LENET
                 Description                           LENET 128x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12782075904
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      70753070
                Total Layers                                             6
        Avg Feedforward (ms)                                         28.13
           Avg Backprop (ms)                                         130.4
          Avg Iteration (ms)                                        164.24
             Avg Samples/sec                                        758.82
             Avg Batches/sec                                          5.93
```

## VGG-16

DL4J summary (milliseconds):

This benchmark is analogous to VGG-16 Torch which is [available here](https://github.com/jcjohnson/cnn-benchmarks#vgg-16). The
model uses 1,000 classes/outputs. All available optimizations have been applied.

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  44.24 | 129.04  | 173.28 | 178.39  |

Full versioning and statistics:

```
                        Name                                         VGG16
                 Description                            VGG16 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      39803688
                Total Layers                                            19
        Avg Feedforward (ms)                                         44.24
           Avg Backprop (ms)                                        129.04
          Avg Iteration (ms)                                        178.39
             Avg Samples/sec                                         86.15
             Avg Batches/sec                                          5.38
```


## Contributing

Contributions are welcome. Please see https://deeplearning4j.org/devguide.

FROM dl4j-benchmark-base:1.0.0 as builder

# add actual run params here
CMD ["/usr/bin/java", "-Xmx8G", "-Dorg.bytedeco.javacpp.maxbytes=8G", "-Dorg.bytedeco.javacpp.maxphysicalbytes=8G", "-cp", "/app/memory-testing/target/memory-testing-2.0-SNAPSHOT-bin.jar", "ai.skymind.TestRun", "--dataClass", "ai.skymind.pipeline.RnnRandomPipeline", "ai.skymind.pipeline.ImgRRPipeline", "--modelClass", "ai.skymind.models.dl4j.RNNModelMLN", "ai.skymind.models.dl4j.CNN2DModelMLN", "--maxIters", "10", "--runtimeSec", "10", "--debugMode", "--asyncShield"]
FROM dl4j-cuda-base:1.0.0 as builder

# eventually change this to eclipse repo
RUN cd / && git clone --depth 1 https://github.com/SkymindIO/deeplearning4j.git
RUN cd /deeplearning4j/libnd4j && ./buildnativeoperations.sh -a native -j 8 && ./buildnativeoperations.sh -c cuda -cc 61 -j 8
RUN cd /deeplearning4j/nd4j && mvn install -T 2 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true -Dlibnd4j.cuda
RUN cd /deeplearning4j/datavec && mvn install -T 4 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd /deeplearning4j/deeplearning4j && mvn install -T 4 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd /deeplearning4j/arbiter && mvn install -T 4 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
#RUN cd /deeplearning4j/nd4s && mvn install -T 4 -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
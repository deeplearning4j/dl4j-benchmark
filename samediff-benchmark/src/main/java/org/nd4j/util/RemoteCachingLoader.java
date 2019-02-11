package org.nd4j.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.function.BiFunction;
import org.nd4j.resources.Downloader;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;

@Slf4j
public class RemoteCachingLoader implements BiFunction<File,String,SameDiff> {

    public static RemoteCachingLoader LOADER = new RemoteCachingLoader();
    public static final File BASE_MODEL_DL_DIR = new File(getBaseModelDir(), ".nd4jtests");
    public static File currentTestDir;

    public static String getBaseModelDir(){
        String s = System.getProperty("org.nd4j.tests.modeldir");
        if(s != null && !s.isEmpty()){
            return s;
        }
        return System.getProperty("user.home");
    }

    @Override
    public SameDiff apply(File file, String name) {
        try {
            String s = FileUtils.readFileToString(file, StandardCharsets.UTF_8).replaceAll("\r\n","\n");
            String[] split = s.split("\n");
            if(split.length != 2 && split.length != 3){
                throw new IllegalStateException("Invalid file: expected 2 lines with URL and MD5 hash, or 3 lines with " +
                        "URL, MD5 hash and file name. Got " + split.length + " lines");
            }
            String url = split[0];
            String md5 = split[1];

            File localDir = new File(BASE_MODEL_DL_DIR, name);
            if(!localDir.exists())
                localDir.mkdirs();

            String filename = FilenameUtils.getName(url);
            File localFile = new File(localDir, filename);

            if(localFile.exists() && !Downloader.checkMD5OfFile(md5, localFile)) {
                log.info("Deleting local file: does not match MD5. {}", localFile.getAbsolutePath());
                localFile.delete();
            }

            if (!localFile.exists()) {
                log.info("Starting resource download from: {} to {}", url, localFile.getAbsolutePath());
                Downloader.download(name, new URL(url), localFile, md5, 3);
            }

            File modelFile;

            if(filename.endsWith(".pb")) {
                modelFile = localFile;
            } else if(filename.endsWith(".tar.gz") || filename.endsWith(".tgz")){
                List<String> files = ArchiveUtils.tarGzListFiles(localFile);
                String toExtract = null;
                if(split.length == 3){
                    //Extract specific file
                    toExtract = split[2];
                } else {
                    for (String f : files) {
                        if (f.endsWith(".pb")) {
                            if (toExtract != null) {
                                throw new IllegalStateException("Found multiple .pb files in archive: " + toExtract + " and " + f);
                            }
                            toExtract = f;
                        }
                    }
                }
                Preconditions.checkState(toExtract != null, "Found to .pb files in archive: %s", localFile.getAbsolutePath());

                Preconditions.checkNotNull(currentTestDir, "currentTestDir has not been set (is null)");
                modelFile = new File(currentTestDir, "tf_model.pb");
                ArchiveUtils.tarGzExtractSingleFile(localFile, modelFile, toExtract);
            } else if(filename.endsWith(".zip")){
                throw new IllegalStateException("ZIP support - not yet implemented");
            } else {
                throw new IllegalStateException("Unknown format: " + filename);
            }

            try(InputStream is = new BufferedInputStream(new FileInputStream(modelFile))){
                SameDiff sd = TFGraphMapper.getInstance().importGraph(is);
                return sd;
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }
}

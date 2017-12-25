package org.rainforc.deepleaining4j.common;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.apache.commons.io.filefilter.SuffixFileFilter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 将图片数据序列化为dataset方便上传到以训练模型
 * Created by qhp
 * 2017/12/04 17:12.
 */
public class SerializeData {
  public static void main(String [] args){
    File dir = new File("E:\\test\\mnist_png\\training");
    String trainingPath = "E:\\test\\mnist_png\\mnistTraining.dat";
    final List<String> lstLabelNames = Arrays.asList("0","1","2","3","4","5","6","7","8","9");  //Chinese Label
    final ImageLoader imageLoader = new ImageLoader(28, 28, 1);             //Load Image
    final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);    //Normalize
    //构建spark net
    SparkConf sparkConf = new SparkConf();
    sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
    sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
    sparkConf.set("spark.locality.wait", "0");
    sparkConf.setMaster("local[*]");

    //--conf spark.locality.wait=0
    sparkConf.setAppName("DL4J Spark MLP Example");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);
    Function imageFunc = new Function<String, DataSet>() {
      @Override
      public DataSet call(String imagePath) throws Exception {
        FileSystem fs = FileSystem.get(new Configuration());
        DataInputStream in = fs.open(new Path(imagePath));
        INDArray features = imageLoader.asRowVector(in);            //features tensor
        String[] tokens = imagePath.split("\\\\");
        String label = tokens[tokens.length-2];
        int intLabel = Integer.parseInt(label);
        //一维数组设置标签值
        INDArray labels = Nd4j.zeros(10);                           //labels tensor
        labels.putScalar(0, intLabel, 1.0);
        DataSet trainData = new DataSet(features, labels);          //DataSet, wrapper of features and labels
        trainData.setLabelNames(lstLabelNames);
        scaler.preProcess(trainData);                               //normalize
        fs.close();
        return trainData;
      }
    };
    List<String> fileNames = new ArrayList<>();
    listFiles(fileNames, dir.toPath(),NativeImageLoader.ALLOWED_FORMATS,true);
    System.out.print("size:"+fileNames.size());
    JavaRDD<String> javaRDDImagePathTrain = sc.parallelize(fileNames);
    JavaRDD<DataSet> javaRDDImageTrain = javaRDDImagePathTrain.map(imageFunc);
    //save training data
    javaRDDImageTrain.saveAsObjectFile(trainingPath);
  }

  private static Collection<String> listFiles(Collection<String> fileNames, java.nio.file.Path dir, String[] allowedFormats,
      boolean recursive) {
    IOFileFilter filter;
    if (allowedFormats == null) {
      filter = new RegexFileFilter(".*");
    } else {
      filter = new SuffixFileFilter(allowedFormats);
    }

    try (DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(dir)) {
      for (java.nio.file.Path path : stream) {
        if (Files.isDirectory(path) && recursive) {
          listFiles(fileNames, path, allowedFormats, recursive);
        } else {
          if (allowedFormats == null) {
            fileNames.add(path.toString());
          } else {
            if (filter.accept(path.toFile())) {
              fileNames.add(path.toString());
            }
          }
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    return fileNames;
  }
}

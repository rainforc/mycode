package org.rainforc.deepleaining4j.common;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.util.Random;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 分别序列化训练集和测试集到本地磁盘
 * Created by qhp
 * 2017/12/11 11:23.
 */
public class InitialData {
  private static final Logger LOGGER = LoggerFactory.getLogger(FeaturizedPreSave.class);
  private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
  private static final Random rng  = new Random(13);
  private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
  private static final int height = 224;
  private static final int width = 224;
  private static final int channels = 3;
  private static final int trainPerc = 80;
  private static final int batchSize = 20;
  private static final int numClasses = 257;

  public static final String featurizeExtractionLayer = "fc2";
    public static void main(String []args)  throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
      //import org.deeplearning4j.transferlearning.vgg16 and print summary
      LOGGER.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
      InputStream modelStream = new ClassPathResource("vgg16/vgg16_dl4j_inference.zip").getInputStream();

      ComputationGraph vgg16 =  ModelSerializer.restoreComputationGraph(modelStream);
      LOGGER.info(vgg16.summary());

      //use the TransferLearningHelper to freeze the specified vertices and below
      //NOTE: This is done in place! Pass in a cloned version of the model if you would prefer to not do this in place
      TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16, featurizeExtractionLayer);
      LOGGER.info(vgg16.summary());
      String picDir = new ClassPathResource("vgg16\\257class-pic\\257_ObjectCategories").getFile().getPath();
      setup(picDir, transferLearningHelper);
      LOGGER.info("Finished pre saving featurized test and train data");
    }



  private static void setup(String picDir,  TransferLearningHelper transferLearningHelper) throws IOException {
      File parentDir = new File(picDir);
      FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
      BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
      if (trainPerc >= 100) {
        throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
      }
      InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
      InputSplit trainData = filesInDirSplit[0];
      InputSplit testData = filesInDirSplit[1];
      saveData(trainData,numClasses, transferLearningHelper ,picDir, true);
      saveData(testData,numClasses, transferLearningHelper ,picDir, false);
  }

  /**
   * 序列化特征数据到磁盘用于训练和测试
   * @param split
   * @throws IOException
   */
  private static void saveData(InputSplit split, int numClasses, TransferLearningHelper transferLearningHelper, String parentDir, boolean isTrain) throws IOException {
    ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
    recordReader.initialize(split);
    DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
    iter.setPreProcessor(new VGG16ImagePreProcessor());
    int trainDataSaved = 0;
    while (iter.hasNext()) {
      //进行特征化处理
      DataSet currentFeaturized = transferLearningHelper.featurize(iter.next());
      //保存到本地磁盘
      saveToDisk(parentDir, currentFeaturized, trainDataSaved, isTrain);
      trainDataSaved++;
    }
  }

  private static void saveToDisk(String parentDir, DataSet currentFeaturized, int iterNum, boolean isTrain) {
      String path = Paths.get(parentDir).getParent().toString();
      File fileFolder = isTrain ? new File(path,"trainFolder"): new File(path,"testFolder");
      if (iterNum == 0) {
          fileFolder.mkdirs();
      }
      String fileName = "object-" + featurizeExtractionLayer + "-";
      fileName += isTrain ? "train-" : "test-";
      fileName += iterNum + ".bin";
      currentFeaturized.save(new File(fileFolder,fileName));
  }
}

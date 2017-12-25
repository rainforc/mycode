package org.rainforc.deepleaining4j;

import java.io.File;
import java.util.Random;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.play.PlayUIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.rainforc.deepleaining4j.common.CustomImageRecordReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * 模型训练与测试
 * Created by qhp on 11/28/16.
 */
public class MyMnistModel {
  private static Logger LOG = LoggerFactory.getLogger(MyMnistModel.class);
  public static final String DATA_PATH = "E:\\machine-learning\\mnist";

    public static void main(String[] args) throws Exception {
//      Path dPath =  new Path(DATA_PATH+File.separator+"mnist_png.tar.gz");
//      Path parentPath =  dPath.getParent();
//      FileSystem fs;
//      try {
//        fs = dPath.getFileSystem(new Configuration());
//      } catch (IOException e) {
//          LOG.error("get file system error by path:"+dPath, e);
//          return;
//      }
//      if(fs!=null && !fs.exists(parentPath)){
//          LOG.info("start untar files ......");
//          FileUtil.extractTarOrZipOrJar(dPath.toString(), parentPath.toString());
//      }
      int height = 28;
      int width = 28;
      int channels = 1;
      int rngseed = 123;
      Random randNumGen = new Random(rngseed);
      //微批次大小指计算梯度和参数更新值时一次使用的样例数量
      int batchSize = 128;
      //输出的类别数
      int outputNum = 10;
      //完整地遍历数据集的次数
      int numEpochs = 4;
      // Define the File Paths
      File trainData = new File(DATA_PATH + "/training");
      File testData = new File(DATA_PATH + "/testing");

      //递归找到该目录下的所有文件
      FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
      FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);

      // Extract the parent path as the image label
      ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

      CustomImageRecordReader recordReader = new CustomImageRecordReader(height,width,channels,labelMaker);

      // Initialize the record reader
      // add a listener, to extract the name
      recordReader.initialize(train);
      recordReader.setListeners(new LogRecordListener());

      // DataSet Iterator
      DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

      // Scale pixel values to 0-1
      DataNormalization scaler = new ImagePreProcessingScaler(0,1);
      scaler.fit(dataIter);
      dataIter.setPreProcessor(scaler);

      //Set up network configuration:
      MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
          .iterations(1)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .learningRate(0.005)
          .seed(12345)
          .regularization(true)
          .l2(0.001)
          .weightInit(WeightInit.XAVIER)
          .updater(Updater.NESTEROVS)
          .list()
          .layer(0, new ConvolutionLayer.Builder(5, 5)
              //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
              .nIn(1)
              .stride(1, 1)
              .nOut(20)
              .activation(Activation.IDENTITY)
              .build())
          .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
              .kernelSize(2,2)
              .stride(2,2)
              .build())
          .layer(2, new ConvolutionLayer.Builder(5, 5)
              //Note that nIn need not be specified in later layers
              .stride(1, 1)
              .nOut(50)
              .activation(Activation.IDENTITY)
              .build())
          .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
              .kernelSize(2,2)
              .stride(2,2)
              .build())
          .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
              .nOut(500).build())
          .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
              .nOut(outputNum)
              .activation(Activation.SOFTMAX)
              .build())
          .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
          .backprop(true).pretrain(false).build();
      MultiLayerNetwork newModel = new MultiLayerNetwork(conf);
      newModel.init();
      newModel.setListeners(new ScoreIterationListener(10));

      //Initialize the user interface backend defalut port 9000
      //http://localhost:9000
      //UIServer uiServer = UIServer.getInstance();
      PlayUIServer uiServer = new PlayUIServer();
      uiServer.runMain(new String[] {"--uiPort", String.valueOf(9001)});
      //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
      //Then add the StatsListener to collect this information from the network, as it trains
      StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
      int listenerFrequency = 1;
      newModel.setListeners(new StatsListener(statsStorage, listenerFrequency));
      //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
      uiServer.attach(statsStorage);
      LOG.info("*****TRAIN MODEL********");
      long start = System.currentTimeMillis();
      for(int i = 0; i<numEpochs; i++){
          newModel.fit(dataIter);
      }
      long useTime = System.currentTimeMillis()-start;
      LOG.info("=========================train model total used {} ms", useTime);
      LOG.info("******EVALUATE MODEL******");
      outputData(recordReader, test, newModel, dataIter);
    }

  /**
   * 根据模型预测结果
   * @param recordReader
   * @param test
   * @param model
   * @param dataIter
   * @throws Exception
   */
  public static void outputData(RecordReader recordReader, FileSplit test, MultiLayerNetwork model, DataSetIterator dataIter) throws Exception{
      //重新初始化reader流
      recordReader.reset();
      recordReader.initialize(test);
      DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,128,1,10);
      DataNormalization scaler = new ImagePreProcessingScaler(0,1);
      scaler.fit(testIter);
      testIter.setPreProcessor(scaler);
    LOG.info(recordReader.getLabels().toString());
      Evaluation eval = new Evaluation(10);
      while(dataIter.hasNext()){
        DataSet next = dataIter.next();
        INDArray output = model.output(next.getFeatureMatrix());
        // Compare the Feature Matrix from the model
        // with the labels from the RecordReader
        eval.eval(next.getLabels(),output);
        LOG.info(eval.stats());
      }
  }

}

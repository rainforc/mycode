package org.rainforc.deepleaining4j;

import java.io.File;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
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
 * keras导入tensorflow模型
 * Created by qhp on 11/28/16.
 */
public class ImportKerasMnistModel {
  private static Logger log = LoggerFactory.getLogger(ImportKerasMnistModel.class);

    public static void main(String[] args) throws Exception {
      String DATA_PATH =  new ClassPathResource("mnist").getFile().getPath();
      int height = 28;
      int width = 28;
      int channels = 1;
      int rngseed = 123;
      Random randNumGen = new Random(rngseed);
      //微批次大小指计算梯度和参数更新值时一次使用的样例数量
      int batchSize = 3;
      //输出的类别数
      int outputNum = 10;
      //完整地遍历数据集的次数
      int numEpochs = 3;
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
      //BaseImageRecordReader.next()方法
      // 1、asMatrix获取特征矩阵组成第一个INDArray
      // 2、会根据PathLabelGenerator获取图片的标签并生成Writable，组成第二个INDArray
      // 3、依次调用RecordReaderDataSetIterator.next()、RecordReaderMultiDataSetIterator.next 将2个INDArray封装成DataSet对象以供训练
      DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

      // Scale pixel values to 0-1
      DataNormalization scaler = new ImagePreProcessingScaler(0,1);
      scaler.fit(dataIter);
      dataIter.setPreProcessor(scaler);

      //目前只支持keras 1.x api
      String modelFile = new ClassPathResource("mnist/mnist_model1.json").getFile().getPath();
      String weightFile = new ClassPathResource("mnist/mnist_weights1.h5").getFile().getPath();
      MultiLayerNetwork model =  KerasModelImport.importKerasSequentialModelAndWeights(modelFile,weightFile);

      model.setListeners(new ScoreIterationListener(10));
      model.init();

      // learning rate schedule in the form of <Iteration #, Learning Rate>
      Map<Integer, Double> lrSchedule = new HashMap<>();
      //迭代指定次数修改学习率
      lrSchedule.put(0, 0.01);
      lrSchedule.put(500, 0.005);
      lrSchedule.put(1000, 0.001);

      FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
          //每个微批次中的参数更新次数
          .iterations(1)
          //梯度标准化有时可以帮助避免梯度在神经网络训练过程中变得过大（即所谓的梯度膨胀问题，在循环神经网络中较常见）或过小
          //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
          //激活函数
          .activation(Activation.LEAKYRELU)
          //权重初始化
          .weightInit(WeightInit.XAVIER)
          //学习率
          .learningRate(0.01)
          /*
              Alternatively, you can use a learning rate schedule.
              NOTE: this LR schedule defined here overrides the rate set in .learningRate(). Also,
              if you're using the Transfer Learning API, this same override will carry over to
              your new model configuration.
          */
          .learningRateSchedule(lrSchedule)
          .learningRatePolicy(LearningRatePolicy.Schedule)
          //原则上，任意一种更新器都可以和任意一种优化算法组合起来 常用SGD+ Nesterov Momentum or Adam.
          //梯度下降算法
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          //更新器
          .updater(Updater.NESTEROVS)
          //正则化，防止过拟合，剔除过大或者过小的数据的影响
          .regularization(true).l2(1e-4)
          .seed(rngseed)
          .build();

      MultiLayerNetwork newModel =  new TransferLearning.Builder(model)
          .fineTuneConfiguration(fineTuneConf)
          //去掉最后一层，重新构造,最后一层输出层才会定义损失函数，而激活函数是每一层都需要定义
          .removeLayersFromOutput(1)
          .addLayer(new OutputLayer
              //损失函数
              .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
              .activation(Activation.SOFTMAX)
              .nIn(256)
              .nOut(10)
              //正则化，防止过拟合（丢弃法）
              //.dropOut(0.5)
              .build()).build();
      newModel.setListeners(new ScoreIterationListener(10));

      //Initialize the user interface backend defalut port 9000
      //http://localhost:9000
      UIServer uiServer = UIServer.getInstance();

      //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
      //Then add the StatsListener to collect this information from the network, as it trains
      StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
      int listenerFrequency = 1;
      newModel.setListeners(new StatsListener(statsStorage, listenerFrequency));
      log.info("======model summary: "+ newModel.summary());
      //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
      uiServer.attach(statsStorage);
      log.info("*****TRAIN MODEL********");
      long start = System.currentTimeMillis();
      for(int i = 0; i<numEpochs; i++){
          newModel.fit(dataIter);
      }
      long useTime = System.currentTimeMillis()-start;
      log.info("=========================train model total used {} ms", useTime);
      log.info("======new model summary: "+ newModel.summary());

//      log.info("******EVALUATE MODEL******");
     //outputData(recordReader, test, newModel);

      long start1 = System.currentTimeMillis();
      //Load the model
      InputStream modelStream = new ClassPathResource("mnist/mnist-model.zip").getInputStream();
      MultiLayerNetwork loadModel = ModelSerializer.restoreMultiLayerNetwork(modelStream);
      log.info("======loadModel  summary: "+ loadModel.summary());
      outputData(recordReader, test, loadModel);
      long useTime1 = System.currentTimeMillis()-start1;
      log.info("=========================predict model total used {} ms", useTime1);
    }

  /**
   * 根据模型预测结果
   * @param recordReader
   * @param test
   * @param model
   * @throws Exception
   */
  public static void outputData(RecordReader recordReader, FileSplit test, MultiLayerNetwork model) throws Exception{
      //重新初始化reader流
      recordReader.reset();
      recordReader.initialize(test);
      DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,2,128,10);
      DataNormalization scaler = new ImagePreProcessingScaler(0,1);
      scaler.fit(testIter);
      testIter.setPreProcessor(scaler);
      log.info(recordReader.getLabels().toString());
      Evaluation eval = new Evaluation(10);
      while(testIter.hasNext()){
        DataSet next = testIter.next();
        INDArray output = model.output(next.getFeatureMatrix());
        //System.out.println("predict result : "+output.argMax(1).toString());
        //System.out.println("actual result : "+next.getLabels().argMax(1).toString());
        // Compare the Feature Matrix from the model
        // with the labels from the RecordReader
        eval.eval(next.getLabels(),output);
        log.info(eval.stats());
      }
  }

}

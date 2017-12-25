package org.rainforc.deepleaining4j;

import java.io.InputStream;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * keras导入tensorflow模型
 * Created by qhp on 11/28/16.
 */
public class ImportKerasIrisModel {
  private static Logger log = LoggerFactory.getLogger(ImportKerasIrisModel.class);

    public static void main(String[] args) throws Exception {
      int numLinesToSkip = 0;
      char delimiter = ',';
      FileSplit training = new FileSplit(new ClassPathResource("iris/iris-training.txt").getFile());
      FileSplit test = new FileSplit(new ClassPathResource("iris/iris.csv").getFile());

      // Read the iris.txt file as a collection of records
      // 1、LineRecordReader.next()方法只需读取一条记录封装到List<Writable>，因为标签信息就在这条记录中
      // 2、会根据LabelIndex信息的生成标签相关的Writable，组成第二个INDArray
      // 3、依次调用RecordReaderDataSetIterator.next()、RecordReaderMultiDataSetIterator.next 将2个INDArray封装成DataSet对象以供训练
      RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
      recordReader.initialize(training);

      // label index
      int labelIndex = 4;
      // num of classes
      int numClasses = 3;
      // batchsize all
      int batchSize = 150;
      DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
      String modelFile = new ClassPathResource("iris/iris.json").getFile().getPath();
      String weightFile = new ClassPathResource("iris/iris_weights.h5").getFile().getPath();
      MultiLayerNetwork model =  KerasModelImport.importKerasSequentialModelAndWeights(modelFile,weightFile);

      FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
          .learningRate(0.005)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .updater(Updater.NESTEROVS)
          .seed(10)
          .regularization(true).l2(1e-4)
          .build();
      MultiLayerNetwork newModel =  new TransferLearning.Builder(model)
          .fineTuneConfiguration(fineTuneConf)
          .removeLayersFromOutput(1)
          .addLayer(new DenseLayer.Builder().nIn(4)
              .nOut(4)
              .activation(Activation.SIGMOID)
              .weightInit(WeightInit.XAVIER)
              .build())
          .addLayer(new OutputLayer
          .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .activation(Activation.SOFTMAX)
          .nIn(4)
          .nOut(3)
          .build())
          .build();

      newModel.init();
      newModel.setListeners(new ScoreIterationListener(10));

      log.info("*****TRAIN MODEL********");
      for(int i = 0; i<600; i++){
        newModel.fit(iterator);
      }

      outputData(recordReader, test, newModel);
//
//      //Save the model
//      File locationToSave = new File("E:\\opensource\\keras\\iris.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
//      ModelSerializer.writeModel(newModel, locationToSave, true);

      //Load the model
      InputStream modelStream = new ClassPathResource("iris/iris.zip").getInputStream();
      MultiLayerNetwork loadModel = ModelSerializer.restoreMultiLayerNetwork(modelStream);
      outputData(recordReader, test, loadModel);
  }

  public static void outputData(RecordReader recordReader, FileSplit test, MultiLayerNetwork model) throws Exception{
      //加载训练后的新模型
      recordReader.reset();
      recordReader.initialize(test);
      //每批次读取150条记录，每条记录的第4个字段为标签，一共有3类标签
      DataSet data = new RecordReaderDataSetIterator(recordReader,150,4,3).next();
      data.shuffle();
      Evaluation eva = new Evaluation(3);
      INDArray out = model.output(data.getFeatures());
      //预测的标签predict result label
      int[] predictedClasses = model.predict(data.getFeatureMatrix());

      //实际的标签expect result label
      //因为标签有三类，每条记录预测为三个类别的一个，取最大的一个值的索引即为预测分类的标签（分类从0开始）
      //labels [[0.00,  1.00,  0.00],[0.00,  0.00,  1.00],[0.00,  0.00,  1.00]...]]
      INDArray realClasses = data.getLabels().argMax(1);
      //realClasses [1.00,2.00,2.00 ...]
      for(int i=0;i< predictedClasses.length;i++) {
        log.info("===predict label:" + predictedClasses[i]);
        log.info("===real label:" + realClasses.getRow(i).toString());
      }
      //统计预测效果信息
      eva.eval(out,data.getLabels());
      log.info(eva.stats());
  }
}

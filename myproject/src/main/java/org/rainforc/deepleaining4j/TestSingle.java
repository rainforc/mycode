package org.rainforc.deepleaining4j;

import com.google.common.io.LineReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.apache.commons.lang.StringUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.rainforc.deepleaining4j.common.FileUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by qhp
 * 2017/12/19 19:18.
 */
public class TestSingle {
  private static Logger log = LoggerFactory.getLogger(TestSingle.class);

  public static void main(String[] args) throws Exception {
    //Load the model
    MultiLayerNetwork loadModel = ModelSerializer.restoreMultiLayerNetwork("E:\\machine-learning\\mnist\\mnist-model.zip");
    log.info("======loadModel  summary: "+ loadModel.summary());
    ParentPathLabelGenerator myLaber = new ParentPathLabelGenerator();
    List<File> mnistFiles = new LinkedList<>();
    FileUtil.listFiles(mnistFiles, Paths.get("E:\\machine-learning\\mnist\\testing\\9"),
        NativeImageLoader.ALLOWED_FORMATS, false);
    DataNormalization dataNormalization = new ImagePreProcessingScaler(0,1);
    mnistFiles = mnistFiles.subList(0,3);
    //predict mnist model
    //predictPic(mnistFiles, null, dataNormalization, myLaber, 28, 28, 1, 10, 1, loadModel, null, null, null);


    //predict vgg model
    ComputationGraph vggModel = ModelSerializer.restoreComputationGraph("E:\\machine-learning\\vgg16\\vgg16-257Model.zip");
    List<File> vggFiles = new LinkedList<>();
    FileUtil.listFiles(vggFiles, Paths.get("E:\\machine-learning\\vgg16\\257class-pic\\257_ObjectCategories\\010.beer-mug"),
        NativeImageLoader.ALLOWED_FORMATS, false);
    vggFiles = vggFiles.subList(0,3);
    DataNormalization vgg16ImagePreProcessor = new VGG16ImagePreProcessor();
    log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
    ComputationGraph vgg16 =  ModelSerializer.restoreComputationGraph("E:\\machine-learning\\vgg16\\vgg16_dl4j_inference.zip");
    predictPic(vggFiles, null, vgg16ImagePreProcessor, myLaber,224, 224, 3, 257, 2, null, vggModel, vgg16, "fc2");


    MultiLayerNetwork irisModel = ModelSerializer.restoreMultiLayerNetwork("E:\\opensource\\keras\\iris.zip");
    File file = new ClassPathResource("iris-training.txt").getFile();
    List<String>lines = new ArrayList<>();
    InputStream inputStream = new FileInputStream(file);
    LineReader lineReader = new LineReader(new InputStreamReader(inputStream));
    String line;
    while((line = lineReader.readLine())!=null){
      if(StringUtils.isNotEmpty(line)) {
        lines.add(line);
      }
    }

    //predict iris model
    //predictText(lines, ',',3,  irisModel);
  }

  /**
   * 预测指定的图片列表
   * @param picFiles    带预测图片列表
   * @param imageLoader   图片加载器
   * @param dataNormalization   数据标准化
   * @param labelGenerator 标签生成器
   * @param height 高度
   * @param width 宽度
   * @param channels  图片通道数
   * @param numClasses 总分类数
   * @param type 指定使用网络模型还是图模型  1 网络模型 2 图模型
   * @param network 已定型的网络模型
   * @param graph 已定型的图模型
   */
  public static void predictPic(List<File> picFiles, BaseImageLoader imageLoader, DataNormalization dataNormalization,
      PathLabelGenerator labelGenerator, int height, int width, int channels, int numClasses, int type,
      MultiLayerNetwork network, ComputationGraph graph, ComputationGraph originGraph, String frozenLayer){
    //use the TransferLearningHelper to freeze the specified vertices and below
    //NOTE: This is done in place! Pass in a cloned version of the model if you would prefer to not do this in place
    TransferLearningHelper transferLearningHelper = null;
    if(StringUtils.isNotEmpty(frozenLayer) && originGraph!=null ){
        transferLearningHelper = new TransferLearningHelper(originGraph, frozenLayer);
    }
    if(network!=null)
      log.info(network.summary());
    if(graph!=null)
      log.info(graph.summary());

      //String vggImagePath = "E:\\machine-learning\\vgg16\\101class-pic\\101_ObjectCategories\\accordion\\image_0001.jpg";
      //File vggFile = new File(imagePath);
      DataSet dataSet = FileUtil
          .getDataSetByImage(imageLoader, dataNormalization, picFiles, labelGenerator, height, width, channels, numClasses);
      if(transferLearningHelper!=null) {
        dataSet = transferLearningHelper.featurize(dataSet);
      }
      List<String>  predictClassNums = null;
      INDArray predictClassNumsArr = null;
      INDArray realClassNums = dataSet.getLabels().argMax(1);
      if(type==2){
          //单个输出
          INDArray[] outputs = graph.output(dataSet.getFeatureMatrix());
          predictClassNumsArr= outputs[0].argMax(1);
          //predictClassNumsArr = graph.outputSingle(dataSet.getFeatureMatrix()).argMax(1);
      }else {
          //直接预测标签值
          predictClassNums = network.predict(dataSet);
          //INDArray predict = network.output(dataSet.getFeatureMatrix());
          //predictClassNumsArr = predict.argMax(1);
      }
    for(int i=0; i<realClassNums.rows();i++) {
      log.info("===predict label :" + (predictClassNums==null? predictClassNumsArr.getRow(i):predictClassNums.get(i)));
      log.info("===real label :" + realClassNums.getRow(i));
    }

  }

  public static void predictText(List<String> lines,  char delimiter, int numClasses,MultiLayerNetwork network){
        Evaluation eva = new Evaluation(numClasses);
        DataSet dataSet = FileUtil.getDataSetByText(lines,delimiter,numClasses);
        INDArray  output = network.output(dataSet.getFeatureMatrix());
        eva.eval(output, dataSet.getLabels());
        log.info(eva.stats());
        List<String> predictLabels = network.predict(dataSet);
        INDArray realLabel = dataSet.getLabels().argMax(1);
        for(int i=0; i<predictLabels.size();i++) {
          log.info("===predict label :" + predictLabels.get(i));
          log.info("===real label :" + realLabel.getRow(i));
        }
  }

}

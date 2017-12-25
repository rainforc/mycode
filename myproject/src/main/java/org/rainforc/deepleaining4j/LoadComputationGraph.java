package org.rainforc.deepleaining4j;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by qhp
 * 2017/12/13 14:56.
 */
public class LoadComputationGraph {
    public static Logger LOG = LoggerFactory.getLogger(LoadComputationGraph.class);
    public static void main(String [] args) throws Exception{

      //Load the model
      InputStream modelStream = new ClassPathResource("vgg16/vgg16-102.zip").getInputStream();

      ComputationGraph computationGraph = ModelSerializer.restoreComputationGraph(modelStream);
      LOG.info(computationGraph.summary());
      //LOG.info(computationGraph.getConfiguration().toJson());
      Evaluation evaluation = computationGraph.evaluate(testIterator(102));
      LOG.info(evaluation.stats());
      //outputData(computationGraph);
    }

  public static void outputData( ComputationGraph model) throws Exception{
        DataSetIterator dataSetIterator = testIterator(102);
        Evaluation eval = new Evaluation(102);
        //批量预测10张图片
        while(dataSetIterator.hasNext()){
            DataSet next = dataSetIterator.next();
            INDArray[] output = model.output(next.getFeatures());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            eval.eval(next.getLabels(),output[0]);
            LOG.info(eval.stats());
        }
  }

  public static DataSetIterator trainIterator(int numClasses) throws Exception {
    String dirPath = new ClassPathResource("vgg16\\"+numClasses+"class-pic\\trainFolder").getFile().getPath();
    DataSetIterator existingTrainingData = new ExistingMiniBatchDataSetIterator(new File(dirPath),"object-fc2-test-%d.bin");
    DataSetIterator asyncTrainIter = new AsyncDataSetIterator(existingTrainingData);
    return asyncTrainIter;
  }
  public static DataSetIterator testIterator(int numClasses) throws Exception{
    String dirPath = new ClassPathResource("vgg16\\"+numClasses+"class-pic\\testFolder").getFile().getPath();

    DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(
        new File(dirPath),"object-fc2-test-%d.bin");
    DataSetIterator asyncTestIter = new AsyncDataSetIterator(existingTestData);
    return asyncTestIter;
  }
}

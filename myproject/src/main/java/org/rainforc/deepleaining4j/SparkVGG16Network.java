package org.rainforc.deepleaining4j;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import javax.imageio.spi.IIORegistry;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.rainforc.deepleaining4j.common.FileUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

/**
 * 读取dataset训练模型并保存模型文件及UIServer训练信息
 * Created by qhp
 * 2017/12/01 11:40.
 */
public class SparkVGG16Network {
  static {
    IIORegistry registry = IIORegistry.getDefaultInstance();
    registry.registerServiceProvider(new com.twelvemonkeys.imageio.plugins.jpeg.JPEGImageReaderSpi());
  }
    private static Logger LOG = LoggerFactory.getLogger(SparkVGG16Network.class);
    private static final String [] zipOrTarFiles = {"tar","jar","zip"};

    public static void main(String args[]) throws Exception{
        new SparkVGG16Network().entryPoint(args);
    }

    private void entryPoint(String[] args) throws Exception {
          LOG.info("args length is " + args.length + " , args are ");
          for (String arg : args) {
            LOG.info(arg + ",");
          }
          CommandLineParser parser = new DefaultParser();
          // create the Options
          Options options = buildOptions();
          CommandLine line = parser.parse(options, args);
          HelpFormatter formatter = new HelpFormatter();
          formatter.setWidth(120);
          // names = "-batchSizePerWorker", description = "Number of examples to fit each worker with"
          // 每个工作器每次训练操作时使用的样例数
          //int batchSizePerWorker = 32;
          if (line.getOptions().length == 0 || line.hasOption("help")) {
            formatter.printHelp("submit model", options);
            return;
          }
          if (!line.hasOption("dataPath")) {
              LOG.error("need set dataPath");
              formatter.printHelp("submit model", options);
              return;
          }

          int outputNum = Integer.parseInt(line.getOptionValue("outputNum", "10"));
          //数据目录可以是未解压的文件，也可以是已经解压后的父目录
          String dataPath = line.getOptionValue("dataPath");
          int numEpochs = Integer.parseInt(line.getOptionValue("numEpochs", "3"));
          int batchSize = Integer.parseInt(line.getOptionValue("batchSize", "128"));
          String dl4jFile = line.getOptionValue("dl4jFile","dl4jUiServerInfo.dl4j");
          boolean cache = Boolean.valueOf(line.getOptionValue("cache", "false"));
          String outPutPath = line.getOptionValue("outPutPath", "");
          String modelPath = line.getOptionValue("modelPath");

          //构建spark net
          SparkConf sparkConf = new SparkConf();
          sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
          sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
          sparkConf.set("spark.locality.wait", "0");
          //--conf spark.locality.wait=0
          sparkConf.setAppName("DL4JSpark numEpochs:"+numEpochs+"--cache:"+cache);
          JavaSparkContext sc = new JavaSparkContext(sparkConf);
          //hdfs://hdfsnn:28100/test
          Path dPath =  new Path(dataPath);
          FileSystem fs;
          try {
              fs = dPath.getFileSystem(new Configuration());
          } catch (IOException e) {
              LOG.error("get file system error by path:"+dPath, e);
              return;
          }
          //是否需要解压文件
          boolean flag = false;
          for(String suffix: zipOrTarFiles){
            if(dataPath.endsWith(suffix)){
              flag = true;
              break;
            }
          }
          if(flag){
              //解压文件
              FileUtil.extractTarOrZipOrJar(dataPath, outPutPath);
              //解压后更改数据目录,默认解压目录为父目录
              if(StringUtils.isEmpty(outPutPath)) {
                  outPutPath = dPath.getParent().toString();
              }
              LOG.info("finish extractTarOrZipOrJar from "+dataPath+" to "+outPutPath);
          }else {
              outPutPath = dataPath;
              LOG.info("direct read data from "+outPutPath);
          }

          String trainingPath = outPutPath+"/train";
          String testPath = outPutPath+"/test";

          //int examplesPerWorker = 8;      //i.e., minibatch size that each worker gets
          int averagingFrequency = 5;     //Frequency with which parameters are averaged
          ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
              .workerPrefetchNumBatches(2)    //Async prefetch 2 batches for each worker
              .averagingFrequency(averagingFrequency)
              .batchSizePerWorker(batchSize)
              .build();
          Pair<ComputationGraph,ComputationGraph> transferLearningHelper = getTransferHelPer(fs, outputNum, modelPath);
          //SparkComputationGraph sparkComputationGraph = new SparkComputationGraph(sc,transferLearningHelper.unfrozenGraph(),tm);
          SparkComputationGraph sparkComputationGraph = new SparkComputationGraph(sc,transferLearningHelper.getLeft(),tm);
          sparkComputationGraph.setListeners(new ScoreIterationListener(10));
          //*** Tell the network to collect training statistics. These will NOT be collected by default ***
          sparkComputationGraph.setCollectTrainingStats(true);
          //保存模型分析文件到本地(只支持本地)
          StatsStorage ss = new FileStatsStorage(new File(dl4jFile));
          sparkComputationGraph.setListeners(ss, Collections.singletonList(new StatsListener(null)));

          LOG.info("===========start training, path: "+trainingPath);
          for (int i = 0; i < numEpochs; i++) {
                sparkComputationGraph.fit(trainingPath);
                LOG.info("Completed Epoch {}", i);
          }
          //Delete the temp training files, now that we are done with them (if fitting for multiple epochs: would be re-used)
          tm.deleteTempFiles(sc);
          //从hdfs上读取数据进行测试模型
          //JavaRDD<DataSet> testData = sc.objectFile(testPath);

          JavaRDD<DataSet> testData = sc.binaryFiles(testPath + "/*").map(new LoadDataFunction());
          Evaluation eval = sparkComputationGraph.evaluate(testData);
          LOG.info("Eval stats BEFORE fit.....");
          LOG.info(eval.stats()+"\n");

          //Get the statistics:
          SparkTrainingStats stats = sparkComputationGraph.getSparkTrainingStats();

          //Export a HTML file containing charts of the various stats calculated during training
          StatsUtils.exportStatsAsHtml(stats, "/test/vgg16/"+outputNum+"/vgg16.html",sc);
          LOG.info("Training stats exported to {}", new File("/test/vgg16/"+outputNum+"/vgg16.html").getAbsolutePath());
          LOG.info("****************Example finished********************");
          LOG.info("==============save model...");
          //save model to hdfs
          Path hdfsPath = new Path("/test/vgg16/"+outputNum+"/vgg16Model.zip");
          FSDataOutputStream outputStream = fs.create(hdfsPath);
          ComputationGraph computationGraph = sparkComputationGraph.getNetwork();
          ModelSerializer.writeModel(computationGraph, outputStream, true);
    }

    private static Pair<ComputationGraph,ComputationGraph> getTransferHelPer(FileSystem fs, int numClasses, String modelPath){

       LOG.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
       ComputationGraph vgg16 = null;
       try {
          FSDataInputStream inputStream = fs.open(new Path(modelPath));
          vgg16 = ModelSerializer.restoreComputationGraph(inputStream);
       } catch (IOException e) {
          LOG.error("getTransferHelPer model error!", e);
       }

       LOG.info(vgg16.summary());

       //Decide on a fine tune configuration to use.
       //In cases where there already exists a setting the fine tune setting will
       //  override the setting for all layers that are not "frozen".
       FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
           .learningRate(3e-5)
           .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
           .updater(Updater.NESTEROVS)
           .seed(12345)
           .build();

       //Construct a new model with the intended architecture and print summary
       ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
           .fineTuneConfiguration(fineTuneConf)
           .setFeatureExtractor("fc2") //the specified layer and below are "frozen"
           .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
           .addLayer("predictions",
               new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                   .nIn(4096).nOut(numClasses)
                   .weightInit(WeightInit.DISTRIBUTION)
                   .dist(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                   .activation(Activation.SOFTMAX).build(),
               "fc2")
           .build();
       LOG.info(vgg16Transfer.summary());

       //Instantiate the transfer learning helper to fit and output from the featurized dataset
       //The .unfrozenGraph() is the unfrozen subset of the computation graph passed in.
       //If using with a UI or a listener attach them directly to the unfrozenGraph instance
       //With each iteration updated params from unfrozenGraph are copied over to the original model
       TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);
       LOG.info(transferLearningHelper.unfrozenGraph().summary());
       return ImmutablePair.of(transferLearningHelper.unfrozenGraph(),vgg16Transfer);
   }
    /**
   * 构建参数选项
   * @return
   */
  private static Options buildOptions() {
    Options options = new Options();
    Option outputNum = Option.builder("outputNum").type(Integer.class).longOpt("outputNum").hasArg(true).required(true).desc("set num of outputNum!").build();
    Option numEpochs = Option.builder("numEpochs").type(Integer.class).longOpt("numEpochs").hasArg(true).required(false).desc("set num of Epoch!").build();
    Option batchSize = Option.builder("batchSize").type(Integer.class).longOpt("batchSize").hasArg(true).required(false).desc("set num of batchSize per worker!").build();
    Option dataPath = Option.builder("dataPath").type(String.class).longOpt("dataPath").hasArg(true).required(true).desc("set where to read data, absolute path with auth and schema!").build();
    Option uiPort = Option.builder("uiPort").type(String.class).longOpt("uiPort").hasArg(true).required(false).desc("set ui server port ").build();
    Option dl4jFile = Option.builder("dl4jFile").type(String.class).longOpt("dl4jFile").hasArg(true).required(false).desc("save dl4jFile path ").build();
    Option cache = Option.builder("cache").type(String.class).longOpt("cache").hasArg(true).required(false).desc("set cache to meme or not ").build();
    Option modelPath = Option.builder("modelPath").type(String.class).longOpt("modelPath").hasArg(true).required(true).desc("set where to read model, absolute path with auth and schema!").build();
    Option outPutPath = Option.builder("outPutPath").type(String.class).longOpt("outPutPath").hasArg(true).required(false).desc("set where to extractFile, absolute path with auth and schema!").build();


    options.addOption(outputNum);
    options.addOption(numEpochs);
    options.addOption(batchSize);
    options.addOption(dataPath);
    options.addOption(uiPort);
    options.addOption(dl4jFile);
    options.addOption(cache);
    options.addOption(modelPath);
    options.addOption(outPutPath);

    options.addOption("h", "help", false, "help information");
    return options;
  }

  private static class LoadDataFunction implements Function<Tuple2<String, PortableDataStream>, DataSet> {
    @Override
    public DataSet call(Tuple2<String, PortableDataStream> v1) throws Exception {
      DataSet d = new DataSet();
      d.load(v1._2().open());
      return d;
    }
  }

}

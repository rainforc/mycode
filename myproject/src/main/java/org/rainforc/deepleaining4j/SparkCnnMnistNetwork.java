package org.rainforc.deepleaining4j;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;
import javax.imageio.spi.IIORegistry;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.storage.StorageLevel;
import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
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
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.earlystopping.SparkDataSetLossCalculator;
import org.deeplearning4j.spark.earlystopping.SparkEarlyStoppingTrainer;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.deeplearning4j.ui.play.PlayUIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.rainforc.deepleaining4j.common.FileUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 读取dataset训练模型并保存模型文件及UIServer训练信息
 * Created by qhp
 * 2017/12/01 11:40.
 */
public class SparkCnnMnistNetwork {
  static {
    IIORegistry registry = IIORegistry.getDefaultInstance();
    registry.registerServiceProvider(new com.twelvemonkeys.imageio.plugins.jpeg.JPEGImageReaderSpi());
  }
    private static Logger LOG = LoggerFactory.getLogger(SparkCnnMnistNetwork.class);

    public static void main(String args[]) throws Exception{
        new SparkCnnMnistNetwork().entryPoint(args);
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
    double score = Double.parseDouble(line.getOptionValue("score", "0.95"));
    int outputNum = Integer.parseInt(line.getOptionValue("outputNum", "10"));
    String dataPath = line.getOptionValue("dataPath");
    int maxEpochs = Integer.parseInt(line.getOptionValue("maxEpochs", "10"));
    //默认最多运行2小时
    int maxTime = Integer.parseInt(line.getOptionValue("maxTime", "120"));
    int batchSize = Integer.parseInt(line.getOptionValue("batchSize", "128"));
    //保存模型分析文件到本地(只支持本地)
    String dl4jFile = line.getOptionValue("dl4jFile","mnist.dl4j");
    boolean cache = Boolean.valueOf(line.getOptionValue("cache", "false"));
    String saveModelPath = line.getOptionValue("saveModelPath","/test/model.zip");
    int mNit = Integer.parseInt(line.getOptionValue("mNit", "3"));

    final List<String> lstLabelNames = Arrays.asList("0","1","2","3","4","5","6","7","8","9");  //Chinese Label
    final ImageLoader imageLoader = new ImageLoader(28, 28, 1);             //Load Image
    final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);    //Normalize
    //构建spark net
    SparkConf sparkConf = new SparkConf();
    sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
    sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
    sparkConf.set("spark.locality.wait", "0");
    //--conf spark.locality.wait=0
    sparkConf.setAppName("DL4JSpark maxEpochs:"+maxEpochs+"--cache:"+cache);
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
    //解压文件
    if(fs!=null && !fs.exists(new Path(dataPath,"mnist_png"))){
          FileUtil.extractTarOrZipOrJar(dataPath+File.separator+"mnist_png.tar.gz", dataPath);
    }

      Function imageFunc = new Function<String, DataSet>() {
        @Override
        public DataSet call(String imagePath) throws Exception {
          FileSystem fs = FileSystem.get(new Configuration());
          DataInputStream in = fs.open(new Path(imagePath));
          INDArray features = imageLoader.asRowVector(in);            //features tensor
          String[] tokens = imagePath.split("\\/");
          String label = tokens[tokens.length-1].split("\\.")[0];
          int intLabel = Integer.parseInt(label);
          //1行10列的数组
          INDArray labels = Nd4j.zeros(10);                           //labels tensor
          //添加标签，第0行第intLabel列的值为1
          labels.putScalar(0, intLabel, 1.0);
          DataSet trainData = new DataSet(features, labels);          //DataSet, wrapper of features and labels
          trainData.setLabelNames(lstLabelNames);
          scaler.preProcess(trainData);                               //normalize
          fs.close();
          return trainData;
        }
      };

      String trainingPath = dPath+"/serialize/train";
      String testPath = dPath+"/serialize/test";

//      assert fs != null;
//      if(!fs.exists(new Path(trainingPath))) {
//          fs.mkdirs(new Path(trainingPath));
//          LOG.info("start serialize training data ......");
//          //hdfs://hdfsnn:28100/test/training
//          FileStatus[] tariningList = fs.listStatus(new Path("test","mnist_png"+File.separator+"training"));
//          List<String> tariningFilePath = new ArrayList<>();
//          for( FileStatus fileStatus :  tariningList){
//            tariningFilePath.add(dPath + File.separator + fileStatus.getPath().getName());
//          }
//          LOG.info("trainingFilePath size: "+tariningFilePath.size());
//          JavaRDD<String> javaRDDImagePathTrain = sc.parallelize(tariningFilePath);
//          JavaRDD<DataSet> javaRDDImageTrain = javaRDDImagePathTrain.map(imageFunc);
//          //save training data
//          javaRDDImageTrain.saveAsObjectFile(trainingPath);
//      }
//
//      if(!fs.exists(new Path(testPath))) {
//        LOG.info("start serialize test data ......");
//        fs.mkdirs(new Path(testPath));
//        //hdfs://hdfsnn:28100/test/training
//        FileStatus[] testList = fs.listStatus(new Path("test","mnist_png"+File.separator+"test"));
//        List<String> testFilePath = new ArrayList<>();
//        for( FileStatus fileStatus :  testList){
//          testFilePath.add(dPath + File.separator + fileStatus.getPath().getName());
//        }
//        LOG.info("testFilePath size: "+testFilePath.size());
//        JavaRDD<String> javaRDDImagePathTest = sc.parallelize(testFilePath);
//        JavaRDD<DataSet> javaRDDImageTest = javaRDDImagePathTest.map(imageFunc);
//        //save test data
//        javaRDDImageTest.saveAsObjectFile(testPath);
//      }

      //int examplesPerWorker = 8;      //i.e., minibatch size that each worker gets
      int averagingFrequency = 5;     //Frequency with which parameters are averaged
      ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
          .workerPrefetchNumBatches(2)    //Async prefetch 2 batches for each worker
          .averagingFrequency(averagingFrequency)
          .batchSizePerWorker(batchSize)
          .build();

        //load image data from hdfs
        JavaRDD<DataSet> trainData = sc.objectFile(trainingPath);
        JavaRDD<DataSet> testData = sc.objectFile(testPath);

        MultiLayerConfiguration configuration = getConf(outputNum);
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, configuration, tm);
        sparkNetwork.setListeners(new ScoreIterationListener(10));
        //以序列化方式存储数据，优先内存再磁盘，只存一份，节约空间，重新读取需要反序列化消耗cpu
        if(cache) {
            LOG.info("===========cache file ");
            trainData.persist(StorageLevel.MEMORY_AND_DISK_SER());
        }
        //*** Tell the network to collect training statistics. These will NOT be collected by default ***
        sparkNetwork.setCollectTrainingStats(true);
        //保存模型分析文件到hdfs
        StatsStorage ss = new FileStatsStorage(new File(dl4jFile));
        sparkNetwork.setListeners(ss, Collections.singletonList(new StatsListener(null)));

//        LOG.info("===========start training, path: "+trainingPath);
//        for (int i = 0; i < maxEpochs; i++) {
//            sparkNetwork.fit(trainData);
//            LOG.info("Completed Epoch {}", i);
//        }


        //早停法，指在指定条件下可以停止训练
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder()
        //最多进行100次 epoch
        .epochTerminationConditions(new MaxEpochsTerminationCondition(maxEpochs))
        //连续3次没有改善
        .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(mNit))
        .evaluateEveryNEpochs(1)
        //最长运行40min
        .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(maxTime, TimeUnit.MINUTES))
        //准确率达到0.98
        .iterationTerminationConditions(new MaxScoreIterationTerminationCondition(score)) //Max of score
        .scoreCalculator(new SparkDataSetLossCalculator(testData, true, sc.sc()))     //Calculate test set score
        .modelSaver(saver)
        .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> sparkEarlyStoppingTrainer = new SparkEarlyStoppingTrainer(sc, tm, esConf, sparkNetwork.getNetwork(),trainData);
        EarlyStoppingResult<MultiLayerNetwork> result = sparkEarlyStoppingTrainer.fit();
        LOG.info("=============train result: "+result);
        //assertEquals(5, result.getTotalEpochs());
        //assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition, result.getTerminationReason());
        Map<Integer, Double> scoreVsIter = result.getScoreVsEpoch();
        String headLine = formatStringToSameLength("Iterator",30) + " | " + formatStringToSameLength("Score",10);
        LOG.info("=============scoreVsIter: ");
        LOG.info(headLine);
        for(Entry<Integer, Double> entry: scoreVsIter.entrySet()){
            String content = formatStringToSameLength(entry.getKey()+"",15) + " | " + formatStringToSameLength(entry.getValue()+"",10);
            LOG.info(content);
        }

        //assertEquals(5, scoreVsIter.size());
        String expDetails = esConf.getEpochTerminationConditions().get(0).toString();
        LOG.info("=============expDetails: "+expDetails);

        MultiLayerNetwork bestNetwork = result.getBestModel();


        //Delete the temp training files, now that we are done with them (if fitting for multiple epochs: would be re-used)
        tm.deleteTempFiles(sc);
        //Get the statistics:
        SparkTrainingStats stats = sparkNetwork.getSparkTrainingStats();
        //Export a HTML file containing charts of the various stats calculated during training
        StatsUtils.exportStatsAsHtml(stats, "CnnMnist.html",sc);
        LOG.info("Training stats exported to {}", new File("CnnMnist.html").getAbsolutePath());
        LOG.info("****************Example finished********************");

        LOG.info("==============save model...");
        //save model to hdfs
        Path hdfsPath = new Path(saveModelPath);
        FSDataOutputStream outputStream = fs.create(hdfsPath);
        //MultiLayerNetwork bestNetwork = sparkNetwork.getNetwork();
        ModelSerializer.writeModel(bestNetwork, outputStream, true);
//
//        FileStatus[] testingList = fs.listStatus(new Path(dataPath,"mnist_png"+File.separator+"testing"));
//        List<String> testingFilePath = new ArrayList();
//        for( FileStatus fileStatus :  testingList){
//          testingFilePath.add(dPath + File.separator + fileStatus.getPath().getName());
//        }
//        LOG.info("testingFilePath size: "+testingFilePath.size());
//        JavaRDD<String> javaRDDImagePathTest = sc.parallelize(testingFilePath);
//        JavaRDD<DataSet> javaRDDImageTrainTest = javaRDDImagePathTest.map(imageFunc);
//        LOG.info("start test model ......");
//        Evaluation testEvaluation = sparkNetwork.evaluate(javaRDDImageTrainTest);
//        LOG.info(testEvaluation.stats());
  }

  /**
   * 获取网络配置
   * @param outputNum
   * @return
   */
  private static MultiLayerConfiguration getConf(int outputNum){
    // learning rate schedule in the form of <Iteration #, Learning Rate>
    Map<Integer, Double> lrSchedule = new HashMap<>();
    //迭代指定次数修改学习率
    lrSchedule.put(0, 0.01);
    //例如这里是在神经网络迭代第500次的时候，学习率更改为0.005
    lrSchedule.put(500, 0.005);
    lrSchedule.put(1000, 0.001);
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .iterations(1)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(0.01)
        .learningRateSchedule(lrSchedule)
        .learningRateDecayPolicy(LearningRatePolicy.Schedule)
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
        //全连接层得到的参数个数为：
        //input (((28-4)/2-4)/2)*50=800
        //output 500
        //total 800*500=40W
        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
            .nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
        .backprop(true).pretrain(false).build();
    return conf;
  }
  /**
   * 构建参数选项
   * @return
   */
  private static Options buildOptions() {
    Options options = new Options();
    Option outputNum = Option.builder("outputNum").type(Integer.class).longOpt("outputNum").hasArg(true).required(false).desc("set num of outputNum!").build();
    Option maxEpochs = Option.builder("maxEpochs").type(Integer.class).longOpt("maxEpochs").hasArg(true).required(false).desc("set num of maxEpochs to stop!").build();
    Option batchSize = Option.builder("batchSize").type(Integer.class).longOpt("batchSize").hasArg(true).required(false).desc("set num of batchSize!").build();
    Option dataPath = Option.builder("dataPath").type(String.class).longOpt("dataPath").hasArg(true).required(true).desc("set where to read data, absolute path with auth and schema!").build();
    Option saveModelPath = Option.builder("saveModelPath").type(String.class).longOpt("saveModelPath").hasArg(true).required(false).desc("set where to save model, absolute path with auth and schema!").build();
    Option dl4jFile = Option.builder("dl4jFile").type(String.class).longOpt("dl4jFile").hasArg(true).required(false).desc("save dl4jFile path ").build();
    Option cache = Option.builder("cache").type(String.class).longOpt("cache").hasArg(true).required(false).desc("set cache to meme or not ").build();
    Option score = Option.builder("score").type(Double.class).longOpt("score").hasArg(true).required(false).desc("the score of stop!").build();
    Option maxTime = Option.builder("maxTime").type(Integer.class).longOpt("maxTime").hasArg(true).required(false).desc("the time of stop!").build();
    Option mNit = Option.builder("mNit").type(Integer.class).longOpt("mNit").hasArg(true).required(false).desc("the max no improve time of stop!").build();

    options.addOption(outputNum);
    options.addOption(maxEpochs);
    options.addOption(maxTime);
    options.addOption(batchSize);
    options.addOption(dataPath);
    options.addOption(saveModelPath);
    options.addOption(dl4jFile);
    options.addOption(cache);
    options.addOption(score);
    options.addOption(mNit);

    options.addOption("h", "help", false, "help information");
    return options;
  }


  /**
   * 创建spark network并添加UI监听器
   * @param sc spark上下文
   * @param outputNum 输出的分类数目
   * @param batchSizePerWorker 每个worker节点训练的样例数
   * @param uiPort UI server监听的端口
   * @return
   */
  private static SparkDl4jMultiLayer constructSparkModel(JavaSparkContext sc , int  outputNum, int batchSizePerWorker, int uiPort, int batchSize){
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


    //-------------------------------------------------------------
    //Set up the Spark-specific configuration
        /* How frequently should we average parameters (in number of minibatches)?
        Averaging too frequently can be slow (synchronization + serialization costs) whereas too infrequently can result
        learning difficulties (i.e., network may not converge) */
    //Configuration for Spark training: see https://deeplearning4j.org/distributed for explanation of these configuration options
//    VoidConfiguration voidConfiguration = VoidConfiguration.builder()
//
//        /**
//         * This can be any port, but it should be open for IN/OUT comms on all Spark nodes
//         */
//        .unicastPort(40123)
//
//        /**
//         * if you're running this example on Hadoop/YARN, please provide proper netmask for out-of-spark comms
//         */
//        .networkMask("10.1.1.0/24")
//
//        /**
//         * However, if you're running this example on Spark standalone cluster, you can rely on Spark internal addressing via $SPARK_PUBLIC_DNS env variables announced on each node
//         */
//        //.controllerAddress(useSparkLocal ? "127.0.0.1" : null)
//        .build();

    TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
        //默认为此值
        .storageLevel(StorageLevel.MEMORY_ONLY_SER())
        // encoding threshold. Please check https://deeplearning4j.org/distributed for details
        //.updatesThreshold(1e-3)
        .rddTrainingApproach(RDDTrainingApproach.Export)
        .batchSizePerWorker(batchSizePerWorker)
        .workerPrefetchNumBatches(4)
        .build();

    SparkDl4jMultiLayer sparkNet =  new SparkDl4jMultiLayer(sc, conf, tm);
    SparkTrainingStats stats = sparkNet.getSparkTrainingStats();    //获取收集到的统计信息
    try {
        StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc);
    } catch (Exception e) {
        LOG.error("exportStatsAsHtml ERROR", e);
    }
    //添加UIServer监听器
    //Initialize the user interface backend
    PlayUIServer uiServer = new PlayUIServer();
    uiServer.runMain(new String[] {"--uiPort", String.valueOf(uiPort)});

    //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
    //Then add the StatsListener to collect this information from the network, as it trains
    StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
    int listenerFrequency = 1;
    sparkNet.setListeners(new StatsListener(statsStorage, listenerFrequency));

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage);
    //Create the Spark network
    return sparkNet;
  }
  //以空格右填充字符串的长度以对齐
  public static String formatStringToSameLength(String str,int len){
    return String.format("%1$-" + len + "s", str);
  }
}

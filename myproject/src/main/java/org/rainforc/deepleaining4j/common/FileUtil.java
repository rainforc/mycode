package org.rainforc.deepleaining4j.common;

import com.google.common.base.Splitter;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.apache.commons.io.filefilter.SuffixFileFilter;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by qhp
 * 2017/12/12 15:33.
 */
public class FileUtil {
  private static final int BUFFER_SIZE = 4096;
  public static Logger LOG = LoggerFactory.getLogger(FileUtil.class);
  private static final String [] allowedExtensions = {"bin"};
  public static void main(String []args) throws IOException{
    String trainDataPath = "E:\\machine-learning\\vgg16\\256class-pic\\trainFolder";
    String testDataPath = "E:\\machine-learning\\vgg16\\256class-pic\\testFolder";
    List<Pair<String, String>> iterateDirs = new ArrayList<>();
    iterateDirs.add(ImmutablePair.of(trainDataPath, "train"));
    iterateDirs.add(ImmutablePair.of(testDataPath, "test"));
    String targetZip = "E:\\machine-learning\\vgg16\\256class-pic\\256train.zip";
    getZipFile(iterateDirs, targetZip, allowedExtensions, false);
    String imagePath = "E:/machine-learning/mnist/testing/7/6852.png";
    List<File> files = new ArrayList<>();
    files.add(new File(imagePath));
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    int height = 28;
    int width = 28;
    int channels = 1;
    DataNormalization dataNormalization = new ImagePreProcessingScaler(0,1);
    DataSet dataSet = getDataSetByImage(null, dataNormalization, files, labelMaker, height, width, channels, 10);
    System.out.println(dataSet.asList().size());

  }
  /**
   * 将源文件解压到目标路径
   * @param filePath 源文件 tar.gz
   * @param outputPath 目标目录
   * @throws IOException
   */
  public static void extractTarOrZipOrJar(String filePath, String outputPath) throws IOException {
    int fileCount = 0;
    int dirCount = 0;
    if(StringUtils.isEmpty(outputPath)){
        outputPath = new Path(filePath).getParent().toString();
    }
    //源目录
    Path originalPath = new Path(filePath);
    FileSystem fs = originalPath.getFileSystem(new Configuration());
    byte data[] = new byte[BUFFER_SIZE];
    InputStream inputStream = fs.open(originalPath);
    System.out.print("Extracting files");
    if (filePath.endsWith(".zip") || filePath.endsWith(".jar")) {
      //getFromOrigin the zip file content
      ZipInputStream zis = new ZipInputStream(inputStream);
      ZipEntry ze = zis.getNextEntry();
      while (ze != null) {
        String fileName = ze.getName();
        if (ze.isDirectory()) {
          fs.mkdirs(new Path(outputPath + File.separator + fileName));
          zis.closeEntry();
          ze = zis.getNextEntry();
          dirCount++;
          continue;
        }else {
          int count;
          Path outPath = new Path(outputPath + File.separator + fileName);
          FSDataOutputStream out = fs.create(outPath);
          BufferedOutputStream dest = new BufferedOutputStream(out, BUFFER_SIZE);
          //FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
          //BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
          while ((count = zis.read(data, 0, BUFFER_SIZE)) != -1) {
              dest.write(data, 0, count);
          }
          dest.close();
          fileCount++;
        }
        zis.closeEntry();
        ze = zis.getNextEntry();
        if (fileCount % 1000 == 0)
          System.out.print(".");
      }
      zis.close();
    }else if(filePath.endsWith(".tar")) {
      try (TarArchiveInputStream tais = new TarArchiveInputStream(
          new GzipCompressorInputStream(new BufferedInputStream(inputStream)))) {
        TarArchiveEntry entry;

        /** Read the tar entries using the getNextEntry method **/
        while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
          //System.out.println("Extracting file: " + entry.getName());

          //Create directories as required
          if (entry.isDirectory()) {
            fs.mkdirs(new Path(outputPath + File.separator + entry.getName()));
            //new File(outputPath + entry.getName()).mkdirs();
            dirCount++;
          } else {
            int count;
            Path outPath = new Path(outputPath + File.separator + entry.getName());
            FSDataOutputStream out = fs.create(outPath);
            BufferedOutputStream dest = new BufferedOutputStream(out, BUFFER_SIZE);
            //FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
            //BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
            while ((count = tais.read(data, 0, BUFFER_SIZE)) != -1) {
              dest.write(data, 0, count);
            }
            dest.close();
            fileCount++;
          }
          if (fileCount % 1000 == 0)
            System.out.print(".");
        }
      }
    }

    System.out.println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath);
  }

  /**
   * 压缩指定目录的文件到zip中
   * @param sourceDirs 待压缩的目录列表及对应子目录名称
   * @param targetZip 压缩后的目标文件
   * @param allowedExtensions 指定的文件后缀
   * @param recursive 是否要递归搜索
   */
  public static void getZipFile(List<Pair<String, String>> sourceDirs , String targetZip, String [] allowedExtensions, boolean recursive){
    try {
      OutputStream zipOut = new FileOutputStream(targetZip);
      ZipOutputStream zos = new ZipOutputStream(zipOut);
      for (Pair<String, String> sourcePair : sourceDirs) {
        String sourceDir = sourcePair.getLeft();
        String parentDirName = sourcePair.getRight();
        File file = new File(sourceDir);
        if (file.isDirectory()) {
          List<File> fileNames = new LinkedList<>();
          listFiles(fileNames, Paths.get(sourceDir),
              allowedExtensions, recursive);
          if (CollectionUtils.isNotEmpty(fileNames)) {
            for (File imageFile : fileNames) {
              InputStream inputStream = new FileInputStream(imageFile);
              ZipEntry zipEntry;
              if(org.apache.commons.lang3.StringUtils.isEmpty(parentDirName)) {
                zipEntry = new ZipEntry( imageFile.getName());
              }else {
                zipEntry = new ZipEntry(parentDirName + "/" + imageFile.getName());
              }
              zos.putNextEntry(zipEntry);
              IOUtils.copy(inputStream, zos);
              IOUtils.closeQuietly(inputStream);
            }
          } else {
            LOG.info("The directory is null for dir : " + file.getName());
          }
          zos.flush();
        } else {
          LOG.info("The target is not a directory, please check it!");
        }
      }
      IOUtils.closeQuietly(zos);
    }catch (IOException e) {
      LOG.error("getZipFile error!", e);
    }
  }

  /**
   * 获取指定目录下指定格式下的所有文件
   * @param fileNames
   * @param dir
   * @param allowedFormats
   * @param recursive
   * @return
   */
  public static void listFiles(List<File> fileNames, java.nio.file.Path dir, String[] allowedFormats,
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
            fileNames.add(path.toFile());
          } else {
            if (filter.accept(path.toFile())) {
              fileNames.add(path.toFile());
            }
          }
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  /**
   * 根据图片路径和标签生成器，封装好DataSet以便dl4j模型预测结果
   * @param totalClasses
   * @param imageLoader
   * @param dataNormalization
   * @param images
   * @param pathLabelGenerator
   * @param height
   * @param width
   * @param channels
   * @return
   */
  public static DataSet getDataSetByImage( BaseImageLoader imageLoader,
      DataNormalization dataNormalization, List<File> images, PathLabelGenerator pathLabelGenerator, int height, int width, int channels, int totalClasses){
    if (imageLoader == null) {
        imageLoader = new NativeImageLoader(height, width, channels, null);
    }
    int rows = images.size();
    INDArray labelArr = Nd4j.zeros(rows, totalClasses);
    //初始化多维图片集矩阵
    int shape [] = {rows, channels, height, height};
    INDArray featuresArr = Nd4j.create(shape);
    //List<List<Writable>> writablesList = new ArrayList<>();
    for(int i=0; i<images.size(); i++) {
      File image = images.get(i);
      try {
        INDArray row = imageLoader.asMatrix(image);
//        Nd4j.getAffinityManager().ensureLocation(row, AffinityManager.Location.DEVICE);
//        List<Writable> writables = RecordConverter.toRecord(row);
//        writablesList.add(writables);
        //将单张图片数组值赋给目标多维数组
        putExample(featuresArr, row, i);
        if (pathLabelGenerator != null) {
          Writable label = pathLabelGenerator.getLabelForPath(image.getPath());
          int numClass = label.toInt();
          //创建2维数组，第二维包含numClass个元素，并且用0填充
          labelArr.putScalar(i, numClass, 1);
        }
      } catch (Exception e) {
        LOG.error("getDataSetByImage error!", e);
      }
    }
    DataSet dataSet =  new DataSet(featuresArr, labelArr);
    //数据标准化处理
    if(dataNormalization!=null) {
      //dataNormalization.fit(dataSet);
      dataNormalization.preProcess(dataSet);
    }
    dataSet.setLabelNames(getLabelNames(totalClasses));
    return dataSet;
  }

  /**
   * 根据多行文本生成特征信息和标签信息封装成DataSet
   * @return
   */
  public static DataSet getDataSetByText(List<String> lines,  char delimiter, int totalClasses){
    Splitter splitter = Splitter.on(delimiter).trimResults().omitEmptyStrings();
    INDArray labelArr = Nd4j.zeros(lines.size(), totalClasses);
    int dim1 = lines.size();
    int dim2 = 0;
    INDArray featuresArr =null;
    for(int i=0;i<lines.size();i++) {
        String line = lines.get(i);
        List<String> list = splitter.splitToList(line);
        //去掉最后一个标签属性
        //list.remove(list.size()-1);
        List<Writable> writables = new ArrayList<>();
        for (String value : list) {
           writables.add(new Text(value));
        }
        if(dim2 == 0){
            dim2 = writables.size()-1;
        }
        if(featuresArr == null){
            featuresArr = Nd4j.create(dim1, dim2);
        }
        //设置标签矩阵
        Writable label = writables.get(dim2);
        //INDArray arr = Nd4j.create(1, writables.size());
        //INDArray featuresArr = RecordConverter.toArray(writables);
        labelArr.putScalar(i, label.toInt(), 1.0);

        //设置特征矩阵
        writables.remove(dim2);
        for(int j=0;j<writables.size();j++){
            featuresArr.putScalar(i, j, writables.get(j).toDouble());
        }
    }

    DataSet dataSet = new DataSet(featuresArr, labelArr);
    //设置标签名称
    dataSet.setLabelNames(getLabelNames(totalClasses));
    return dataSet;
  }

  /**
   * @param totalClasses
   * @return
   */
  private static List<String> getLabelNames(int totalClasses){
    List<String> labels = new ArrayList<>();
    for(int i=0; i<totalClasses;i++){
      labels.add(i+"");
    }
    return labels;
  }

  /**
   * 设置目标多维数组的子数组值
   * @param arr 目标多维数组
   * @param singleExample 子数组
   * @param exampleIdx 索引位置
   */
  private  static void putExample(INDArray arr, INDArray singleExample, int exampleIdx) {
    switch (arr.rank()) {
      case 2:
        arr.put(new INDArrayIndex[] {NDArrayIndex.point(exampleIdx), NDArrayIndex.all()}, singleExample);
        break;
      case 3:
        arr.put(new INDArrayIndex[] {NDArrayIndex.point(exampleIdx), NDArrayIndex.all(), NDArrayIndex.all()},
            singleExample);
        break;
      case 4:
        arr.put(new INDArrayIndex[] {NDArrayIndex.point(exampleIdx), NDArrayIndex.all(), NDArrayIndex.all(),
            NDArrayIndex.all()}, singleExample);
        break;
      default:
        throw new RuntimeException("Unexpected rank: " + arr.rank());
    }
  }

}

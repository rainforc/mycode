package org.rainforc.deepleaining4j.common;

import com.google.common.base.Splitter;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;

/**
 * 下载所有图片数据，并将其身份名字标签作为父目录
 * Created by qhp
 * 2017/12/19 11:00.
 */
public class GetVggFaceData {
  public final static Splitter spaceSplitter = Splitter.on(" ").trimResults().omitEmptyStrings();
  private final static Logger log = org.slf4j.LoggerFactory.getLogger(GetVggFaceData.class);
  public static final String[] ALLOWED_FORMATS = {"bmp", "gif", "jpg", "jpeg", "jp2", "pbm", "pgm", "ppm", "pnm",
      "png", "tif", "tiff", "exr", "webp", "BMP", "GIF", "JPG", "JPEG", "JP2", "PBM", "PGM", "PPM", "PNM",
      "PNG", "TIF", "TIFF", "EXR", "WEBP"};
  private volatile static LongAdder longAdder = new LongAdder();
  private final static ExecutorService executorService = new ThreadPoolExecutor(300, 300, 0L, TimeUnit.MILLISECONDS,
      new LinkedBlockingQueue<Runnable>(), new ThreadFactoryBuilder().setNameFormat("downLoad Image #%d").setDaemon(true).build());
  //存放所有图片路径和imageUrl
  //private static Map<String, String> map = new ConcurrentHashMap<String, String>();
  private static BlockingQueue<Pair<String, String>> imageInfoQueue = new LinkedBlockingQueue<>(3000000);
  private static BlockingQueue<File> uncheckedImageQueue = new LinkedBlockingQueue<File>(3000000);

  //下载的目标文件存储位置
  public final static String outPutPath = "E:\\machine-learning\\vgg16\\face-pic";
  public static void main(String []args) throws Exception{
        //原始文件，存储了图片下载地址其对应图片任务的名字
        String  filePath = "E:\\machine-learning\\vgg_face_dataset\\files";
        List<File> fileList = new ArrayList<File>();
        FileUtil.listFiles(fileList, Paths.get(filePath), null, false);
        CountDownLatch collectCountDownLatch = new CountDownLatch(fileList.size());
        int downLoadCount = 300;
        CountDownLatch downLoadCountDownLatch = new CountDownLatch(downLoadCount);

        //收集下载链接和图片名称
        for(int i=0; i<fileList.size();i++){
            String number = appendZero(i+"", 4);
            GetImageThread downloadImageThread = new GetImageThread(number, fileList.get(i),collectCountDownLatch);
            executorService.submit(downloadImageThread);
        }
        collectCountDownLatch.await();
        log.info("===============start downLoad files, total "+imageInfoQueue.size()+" images!");

        //下载图片
        for(int i=0; i<downLoadCount; i++) {
            DownLoadImageThread thread = new DownLoadImageThread(downLoadCountDownLatch);
            executorService.submit(thread);
        }
        downLoadCountDownLatch.await();
        log.info("===============finish GetVggFaceData, total "+longAdder.longValue()+" images!");

        //检查图片完整性删除失效的图片
//        String imagesPath = "E:\\machine-learning\\vgg16\\face-pic";
//        FileUtil.listFiles(uncheckedImageQueue, Paths.get(imagesPath), ALLOWED_FORMATS, true);
//        int threadCount = 10;
//        CountDownLatch checkCountDownLatch = new CountDownLatch(threadCount);
//        for(int i=0; i<threadCount; i++) {
//          CheckImageThread checkImageThread = new CheckImageThread(checkCountDownLatch);
//          executorService.submit(checkImageThread);
//        }
//        checkCountDownLatch.await();
//        log.info("===============finish check image, total "+uncheckedImageQueue.size()+" images!");

  }

  /**
   * 比如 1 补齐到4位变为0001
   * @param str
   * @param num
   * @return
   */
  private static String appendZero(String str, int num){
      //按照num位补齐
      int append = num-str.length();
      for(int j=0;j<append;j++){
          str = "0"+str;
      }
      return str;
  }

  static class GetImageThread implements Runnable{
    private File sourceFile;
    private CountDownLatch countDownLatch;
    private String numClass;
    public GetImageThread(String numClass, File sourceFile, CountDownLatch countDownLatch) {
      this.sourceFile = sourceFile;
      this.countDownLatch = countDownLatch;
      this.numClass = numClass;
    }

    @Override
    public void run() {
      String line;
      String fileName = sourceFile.getName();
      log.info("============start download image from file:"+fileName);

      //去掉文件名的txt后缀,目录名变为类似 0001.Arahama_berum
      String name = numClass + "."+fileName.substring(0, fileName.length()-4);
      try {
        LineNumberReader lineNumberReader = new LineNumberReader(new InputStreamReader(new FileInputStream(sourceFile)));
        while ((line = lineNumberReader.readLine())!=null){
          List<String> valueList = spaceSplitter.splitToList(line);
          String imgUrl = valueList.get(1);
          savePic(imgUrl, name);
          longAdder.increment();
        }
      } catch (Throwable e) {
        log.error("read line error from file: " +fileName);
      }
      countDownLatch.countDown();
      log.info("========finish download file: "+fileName+" , total files: "+longAdder.longValue());
    }

    private  void savePic(String imgUrl, String parentName){
      if (imgUrl == null)
        throw new UnsupportedOperationException("imgUrl is null.");
      String localFilename = new File(imgUrl).getName();
      File dir = new File(outPutPath, parentName);
      dir.mkdir();
      String suffix = "";
      int localNameIndex = localFilename.lastIndexOf(".");
      if(localNameIndex<=0 || !Arrays.asList(ALLOWED_FORMATS).contains(localFilename.substring(localNameIndex+1))) {
          int index = imgUrl.lastIndexOf(".");
          int end = imgUrl.lastIndexOf("?");
          if (end <= index) {
            suffix = imgUrl.substring(index);
          } else {
            suffix = imgUrl.substring(index, end);
          }
          //路径中不包含图像后缀，默认为.jpg
          if (!Arrays.asList(ALLOWED_FORMATS).contains(suffix)) {
            suffix = ".jpg";
          }
      }
      String fullPath = dir+File.separator+localFilename+suffix;
      ImmutablePair<String, String> pair = ImmutablePair.of(fullPath, imgUrl);
      imageInfoQueue.offer(pair);
    }
  }


  static class DownLoadImageThread implements Runnable{
    private CountDownLatch countDownLatch;
    public DownLoadImageThread(CountDownLatch countDownLatch) {
        this.countDownLatch = countDownLatch;
    }

    @Override
    public void run() {
      while (!imageInfoQueue.isEmpty()) {
        Pair<String, String> imageInfo = imageInfoQueue.poll();
        String fullPath = imageInfo.getLeft();
        String imgUrl = imageInfo.getRight();
        File cachedFile = new File(fullPath);
        if (!cachedFile.exists()) {
          log.info("Downloading image to " + cachedFile.toString());
          //try {
            //FileUtils.copyURLToFile(new URL(imgUrl), cachedFile);
            //FileUtils.copyURLToFile(new URL(imgUrl), cachedFile, 3000, 5000);
          copyURLToFile(imgUrl, cachedFile, 3000, 12000);
          //} catch (Throwable e) {
            //删除异常的文件
            //targetFile.delete();
            //log.error("download image error! url :" + imgUrl, e);
          //}
        } else {
          log.info("image is exist for url : " + imgUrl);
        }
        longAdder.increment();
        if(longAdder.longValue()%1000 == 0){
            log.info("============================help gc! ");
            System.gc();
        }
      }
      countDownLatch.countDown();
    }
  }

  private static void copyURLToFile(String imgUrl, File targetFile,  int connectionTimeout, int readTimeout){
    URL source;
    try {
        source = new URL(imgUrl);
        URLConnection connection = source.openConnection();
        connection.setConnectTimeout(connectionTimeout);
        connection.setReadTimeout(readTimeout);
        InputStream input = connection.getInputStream();
        //copyInputStreamToFile(input, targetFile);
        FileOutputStream output = FileUtils.openOutputStream(targetFile);
        try {
          IOUtils.copy(input, output);
          output.flush();
          output.close(); // don't swallow close Exception if copy completes normally
        } finally {
          IOUtils.closeQuietly(output);
        }
    } catch (Throwable e) {
        //删除异常的文件
        targetFile.delete();
        log.error("download image error! url :" + imgUrl, e);
    }
  }

  static class CheckImageThread implements Runnable{
    private CountDownLatch countDownLatch;
    public CheckImageThread(CountDownLatch countDownLatch) {
      this.countDownLatch = countDownLatch;
    }

    @Override
    public void run() {
      while (!uncheckedImageQueue.isEmpty()) {
        File imageInfo = uncheckedImageQueue.poll();
        if (imageInfo.exists() && imageInfo.length() < 10*1024) {
          log.info("image file is too small ,delete image: " + imageInfo.getAbsolutePath());
        }
      }
      countDownLatch.countDown();
    }
  }

}

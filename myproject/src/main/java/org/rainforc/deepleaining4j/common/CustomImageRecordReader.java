package org.rainforc.deepleaining4j.common;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 支持读取其它文件系统，比如hdfs,默认为本地文件系统
 * Created by qhp
 * 2017/11/30 18:27.
 */
public class CustomImageRecordReader extends ImageRecordReader {
  private static final int BUFFER_SIZE = 4096;
  private static final Logger LOG = LoggerFactory.getLogger(CustomImageRecordReader.class);

  public CustomImageRecordReader(int height, int width, int channels,
      PathLabelGenerator labelGenerator) {
    super(height, width, channels, labelGenerator);
  }

  @Override
  public List<Writable> next() {

    if (iter != null) {
      List<Writable> ret;
      File image = iter.next();
      Path originalPath = new Path(image.getAbsolutePath());
      FileSystem fs = null;
      try {
         fs = originalPath.getFileSystem(new Configuration());
      } catch (IOException e) {
          LOG.error("get file system error by path:"+originalPath, e);
      }

      currentFile = image;
      INDArray row;
      if (image.isDirectory())
        return next();
      try {
        invokeListeners(image);
        InputStream inputStream;
        if(fs!=null){
            inputStream = fs.open(originalPath);
            row = imageLoader.asMatrix(inputStream);
        }else {
           row = imageLoader.asMatrix(image);
        }
        Nd4j.getAffinityManager().ensureLocation(row, AffinityManager.Location.DEVICE);
        ret = RecordConverter.toRecord(row);
        if (appendLabel || writeLabel)
          ret.add(new IntWritable(labels.indexOf(getLabel(image.getPath()))));
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return ret;
    } else if (record != null) {
      hitImage = true;
      invokeListeners(record);
      return record;
    }
    throw new IllegalStateException("No more elements");
  }
}

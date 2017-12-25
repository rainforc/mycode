package org.rainforc.deepleaining4j;

import java.io.File;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.FileStatsStorage;

/**
 * Created by qhp
 * 2017/12/06 16:06.
 */
public class TestLoadUIServerInfo {
    public static void main(String []args){
        //如果文件已存在：从其中加载数据
        StatsStorage statsStorage = new FileStatsStorage(new File("E:\\machine-learning\\vgg16\\vgg16-101.dl4j"));
        UIServer uiServer = UIServer.getInstance();

      //PlayUIServer uiServer = new PlayUIServer();
      //uiServer.runMain(new String[] {"--uiPort", String.valueOf(9001)});
        uiServer.attach(statsStorage);
    }
}

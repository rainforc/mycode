package org.rainforc.deepleaining4j;

import java.io.File;
import java.util.Collection;
import java.util.Iterator;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by qhp
 * 2017/12/14 10:12.
 */
public class GoogleNewsModel {
       static Logger LOGGER = LoggerFactory.getLogger("org.deeplearning4j.test.GoogleNewsModel");
      public static void main(String []args) throws Exception{
            File file =  new ClassPathResource("word2vec/GoogleNews-vectors-negative300.bin.gz").getFile();
            Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(file);
            //word2Vec.fit();
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            //Alternative: new FileStatsStorage(File) - see UIStorageExample
            uiServer.attach(statsStorage);
            double sim = word2Vec.similarity("test","devolope");
            Collection<String> list = word2Vec.wordsNearestSum("home", 3);
            WeightLookupTable weightLookupTable = word2Vec.lookupTable();
            Iterator<INDArray> vectors = weightLookupTable.vectors();
            INDArray wordVectorMatrix = word2Vec.getWordVectorMatrix("technology");
            double[] wordVector = word2Vec.getWordVector("technology");
            LOGGER.info("GoogleNewsModel......");
            //save model
            //WordVectorSerializer.writeWord2VecModel(word2Vec,"E:\\machine-learning\\Word2vec\\law_sentence.txt");

      }
}

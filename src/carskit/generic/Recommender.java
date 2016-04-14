// Copyright (C) 2015 Yong Zheng
//
// This file is part of CARSKit.
//
// CARSKit is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CARSKit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CARSKit. If not, see <http://www.gnu.org/licenses/>.
//

package carskit.generic;


import com.google.common.collect.*;
import com.google.common.primitives.Doubles;
import happy.coding.io.FileConfiger;
import happy.coding.io.FileIO;
import happy.coding.io.LineConfiger;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.math.Measures;
import happy.coding.math.Randoms;
import happy.coding.math.Sims;
import happy.coding.math.Stats;
import happy.coding.system.Dates;
import happy.coding.system.Debug;

import java.text.SimpleDateFormat;
import java.util.*;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

import carskit.data.processor.DataDAO;
import carskit.data.structure.SparseMatrix;
import carskit.data.setting.*;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.data.MatrixEntry;

import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.cache.LoadingCache;
import sun.org.mozilla.javascript.ast.Block;


public abstract class Recommender implements Runnable{

    // default temporary file directory
    public static String workingPath;
    /************************************ Static parameters for all recommenders ***********************************/
    // configer
    public static FileConfiger cf;
    // matrix of rating data
    public static SparseMatrix rateMatrix;

    // params used for multiple runs
    public static Map<String, List<Float>> params = new HashMap<>();

    // Guava cache configuration
    protected static String cacheSpec;

    // number of cpu cores used for parallelization
    protected static int numCPUs;

    // verbose
    protected static boolean verbose = true;
    // significance output
    protected static boolean significanceOutput = false;

    // line configer for item ranking, evaluation
    protected static LineConfiger rankOptions, algoOptions;

    // is ranking/rating prediction
    public static boolean isRankingPred;
    // threshold to binarize ratings
    public static float binThold;
    // the ratio of validation data split from training data
    public static float validationRatio;
    // is diversity-based measures used
    protected static boolean isDiverseUsed;
    // early-stop criteria
    protected static Measure earlyStopMeasure = null;
    // is save model
    protected static boolean isSaveModel = false;
    // view of rating predictions
    public static String view;

    // rate DAO object
    public static DataDAO rateDao;


    // number of recommended items
    protected static int numRecs, numIgnore;

    // a list of rating scales
    protected static List<Double> ratingScale;
    // number of rating levels
    protected static int numLevels;
    // Maximum, minimum values of rating scales
    protected static double maxRate, minRate;

    // init mean and standard deviation
    protected static double initMean, initStd;
    // small value for initialization
    protected static double smallValue = 0.01;

    // number of nearest neighbors
    protected static int knn;
    // similarity measure
    protected static String similarityMeasure;
    // number of shrinkage
    protected static int similarityShrinkage;

    /**
     * An indicator of initialization of static fields. This enables us to control when static fields are initialized,
     * while "static block" will be always initialized or executed. The latter could cause unexpected exceptions when
     * multiple runs (with different configuration files) are conducted sequentially, because some static settings will
     * not be override in such a "staic block".
     */
    public static boolean resetStatics = true;

    /************************************ Recommender-specific parameters ****************************************/
    // algorithm's name
    public String algoName;
    // current fold
    protected int fold;
    // fold information
    protected String foldInfo;
    // is output recommendation results
    protected boolean isResultsOut = true;
    // candidateItems filter
    protected String ratedItemsFilter;
    private static final Set<String> CANDIDATE_ITEMS_FILTERS = new HashSet<String>(Arrays.asList(new String[] {"none", "user", "user-context"}));

    // number of users, items, ratings
    protected int numUsers, numItems, numRates;

    // user-vector cache, item-vector cache
    protected LoadingCache<Integer, SparseVector> userCache, itemCache;

    // user-items cache, item-users cache
    protected LoadingCache<Integer, List<Integer>> userItemsCache, itemUsersCache;

    // rating matrix for training, validation and test
    protected SparseMatrix trainMatrix, validationMatrix, testMatrix;
    protected boolean isUserSplitting=false, isItemSplitting=false, isCARSRecommender=false;
    protected Table<Integer, Integer, Integer> userIdMapper, itemIdMapper;

    protected librec.data.SparseMatrix train; // this is traditional 2D rating matrix, transformed from either original data, or the splitting process

    // upper symmetric matrix of item-item correlations
    protected SymmMatrix corrs;

    // performance measures
    public Map<Measure, Double[]> measures;
    // global average of training rates
    protected double globalMean;

    /**
     * Recommendation measures
     *
     */
    public enum Measure {
        /* prediction-based measures */
        MAE, RMSE, NMAE, rMAE, rRMSE, MPE, Perplexity,
        /* ranking-based measures */
        D5, D10, Pre, Pre5, Pre10, Rec, Rec5, Rec10, MAP, MAPS, MRR, NDCG, AUC,
        /* execution time */
        TrainTime, TestTime,
        /* loss value */
        Loss
    }


    public Recommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {

        // config recommender
        if (cf == null || rateMatrix == null) {
            Logs.error("Recommender is not well configed");
            System.exit(-1);
        }

        numUsers = rateDao.numUsers();
        numItems = rateDao.numItems();

        // static initialization (reset), only done once
        if (resetStatics) {
            // change the indicator
            resetStatics = false;

            ratingScale = rateDao.getRatingScale();
            minRate = ratingScale.get(0);
            maxRate = ratingScale.get(ratingScale.size() - 1);
            numLevels = ratingScale.size();

            //numUsers = rateDao.numUsers();
            //numItems = rateDao.numItems();

            initMean = 0.0;
            initStd = 0.1;

            cacheSpec = cf.getString("guava.cache.spec", "maximumSize=200,expireAfterAccess=2m");

            rankOptions = cf.getParamOptions("item.ranking");
            isRankingPred = rankOptions.isMainOn();
            isDiverseUsed = rankOptions.contains("-diverse");
            numRecs = rankOptions.getInt("-topN", -1);
            numIgnore = rankOptions.getInt("-ignore", -1);

            LineConfiger evalOptions = cf.getParamOptions("evaluation.setup");
            view = evalOptions.getString("--test-view", "all");

            String earlyStop = evalOptions.getString("--early-stop");
            if (earlyStop != null) {
                for (Measure m : Measure.values()) {
                    if (m.name().equalsIgnoreCase(earlyStop)) {
                        earlyStopMeasure = m;
                        break;
                    }
                }
            }

            int numProcessors = Runtime.getRuntime().availableProcessors();
            numCPUs = evalOptions.getInt("-cpu", numProcessors);
            Randoms.seed(evalOptions.getLong("--rand-seed", System.currentTimeMillis())); // initial random seed

            // output options
            LineConfiger outputOptions = cf.getParamOptions("output.setup");
            if (outputOptions != null) {
                verbose = outputOptions.isOn("-verbose", true);
                significanceOutput = outputOptions.isOn("-significancedata", true);
                isSaveModel = outputOptions.contains("--save-model");
            }

            knn = cf.getInt("num.neighbors", 20);
            similarityMeasure = cf.getString("similarity", "PCC");
            similarityShrinkage = cf.getInt("num.shrinkage", 30);
        }


        this.trainMatrix = trainMatrix;
        this.testMatrix = testMatrix;

        // fold info
        this.fold = fold;
        foldInfo = fold > 0 ? " fold [" + fold + "]" : "";

        // whether to write out results
        LineConfiger outputOptions = cf.getParamOptions("output.setup");
        if (outputOptions != null) {
            //isResultsOut = outputOptions.isMainOn();
            isResultsOut = true;
        }

        // global mean
        globalMean=trainMatrix.getGlobalAvg();

        // class name as the default algorithm name
        algoName = this.getClass().getSimpleName();
        // get parameters of an algorithm
        algoOptions = getModelParams(algoName);
        // get the filter parameter
        ratedItemsFilter = cf.getString("items.filter", "none");
        if (!CANDIDATE_ITEMS_FILTERS.contains(ratedItemsFilter)) {
            Logs.error("Filter {} defined in items.filter is invalid", ratedItemsFilter);
            System.exit(-1);
        }

        // compute item-item correlations
        if (isRankingPred && isDiverseUsed)
            corrs = new SymmMatrix(numItems);
    }

    protected LineConfiger getModelParams(String algoName) {
        return cf.contains(algoName) ? cf.getParamOptions(algoName) : null;
    }


    protected double predict(int u, int j, int c) throws Exception {
        return globalMean;
    }

    protected double predict(int u, int j) throws Exception {
        return globalMean;
    }

    /**
     * predict a specific rating for user u on item j. It is useful for evalution which requires predictions are
     * bounded.
     *
     * @param u
     *            user id
     * @param j
     *            item id
     *
     * @param c
     *            context id
     * @param bound
     *            whether to bound the prediction
     * @return prediction
     */

    protected double predict(int u, int j, int c, boolean bound) throws Exception {
        double pred = predict(u, j,c);

        if (bound) {
            if (pred > maxRate)
                pred = maxRate;
            if (pred < minRate)
                pred = minRate;
        }

        return pred;
    }

    public void execute() throws Exception {

        Stopwatch sw = Stopwatch.createStarted();
        if (Debug.ON) {
            // learn a recommender model
            initModel();

            // show algorithm's configuration
            // printAlgoConfig();

            // build the model
            buildModel();

            // post-processing after building a model, e.g., release intermediate memory to avoid memory leak
            postModel();
        } else {
            /**
             * load a learned model: this code will not be executed unless "Debug.OFF" mainly for the purpose of
             * exemplifying how to use the saved models
             */
            loadModel();
        }
        long trainTime = sw.elapsed(TimeUnit.MILLISECONDS);

        // evaluation
        if (verbose)
            Logs.debug("{}{} evaluate test data ... ", algoName, foldInfo);
        measures = isRankingPred ? evalRankings() : evalRatings();
        String measurements = getEvalInfo(measures);
        sw.stop();
        long testTime = sw.elapsed(TimeUnit.MILLISECONDS) - trainTime;

        // collecting results
        measures.put(Measure.TrainTime, new Double[] { (double) trainTime });
        measures.put(Measure.TestTime, new Double[] { (double) testTime } );

        String evalInfo = algoName + foldInfo + ": " + measurements + "\tTime: "
                + Dates.parse(measures.get(Measure.TrainTime)[0].longValue()) + ", "
                + Dates.parse(measures.get(Measure.TestTime)[0].longValue());
        if (!isRankingPred)
            evalInfo += "\tView: " + view;

        if (fold > 0)
        Logs.debug(evalInfo);

        if (isSaveModel)
            saveModel();
    }



    protected double correlation(SparseVector iv, SparseVector jv) {
        return correlation(iv, jv, similarityMeasure);
    }

    /**
     * Compute the correlation between two vectors for a specific method
     *
     * @param iv
     *            vector i
     * @param jv
     *            vector j
     * @param method
     *            similarity method
     * @return the correlation between vectors i and j; return NaN if the correlation is not computable.
     */
    protected double correlation(SparseVector iv, SparseVector jv, String method) {

        // compute similarity
        List<Double> is = new ArrayList<>();
        List<Double> js = new ArrayList<>();

        for (Integer idx : jv.getIndex()) {
            if (iv.contains(idx)) {
                is.add(iv.get(idx));
                js.add(jv.get(idx));
            }
        }

        double sim = 0;
        switch (method.toLowerCase()) {
            case "cos":
                // for ratings along the overlappings
                sim = Sims.cos(is, js);
                break;
            case "cos-binary":
                // for ratings along all the vectors (including one-sided 0s)
                sim = iv.inner(jv) / (Math.sqrt(iv.inner(iv)) * Math.sqrt(jv.inner(jv)));
                break;
            case "msd":
                sim = Sims.msd(is, js);
                break;
            case "cpc":
                sim = Sims.cpc(is, js, (minRate + maxRate) / 2.0);
                break;
            case "exjaccard":
                sim = Sims.exJaccard(is, js);
                break;
            case "pcc":
            default:
                sim = Sims.pcc(is, js);
                break;
        }

        // shrink to account for vector size
        if (!Double.isNaN(sim)) {
            int n = is.size();
            int shrinkage = cf.getInt("num.shrinkage");
            if (shrinkage > 0)
                sim *= n / (n + shrinkage + 0.0);
        }

        return sim;
    }

    /**
     * @return the evaluation information of a recommend
     */
    public static String getEvalInfo(Map<Measure, Double[]> measures) {
        String evalInfo = null;
        if (isRankingPred) {
            if (isDiverseUsed)
                evalInfo = String.format("Pre5: %.6f, Pre10: %.6f, Rec5: %.6f, Rec10: %.6f, AUC: %.6f, MAP: %.6f, NDCG: %.6f, MRR: %.6f, D5: %.6f, D10: %.6f, Pre: %s, Rec: %s, MAPs: %s",
                        measures.get(Measure.Pre5)[0], measures.get(Measure.Pre10)[0], measures.get(Measure.Rec5)[0],
                        measures.get(Measure.Rec10)[0], measures.get(Measure.AUC)[0], measures.get(Measure.MAP)[0],
                        measures.get(Measure.NDCG)[0], measures.get(Measure.MRR)[0], measures.get(Measure.D5)[0],
                        measures.get(Measure.D10)[0], Arrays.toString(measures.get(Measure.Pre)), Arrays.toString(measures.get(Measure.Rec)), Arrays.toString(measures.get(Measure.MAPS)));
            else
                evalInfo = String.format("Pre5: %.6f, Pre10: %.6f, Rec5: %.6f, Rec10: %.6f, AUC: %.6f, MAP: %.6f, NDCG: %.6f, MRR: %.6f, Pre: %s, Rec: %s, MAPs: %s",
                        measures.get(Measure.Pre5)[0],
                        measures.get(Measure.Pre10)[0], measures.get(Measure.Rec5)[0], measures.get(Measure.Rec10)[0],
                        measures.get(Measure.AUC)[0], measures.get(Measure.MAP)[0], measures.get(Measure.NDCG)[0],
                        measures.get(Measure.MRR)[0], Arrays.toString(measures.get(Measure.Pre)), Arrays.toString(measures.get(Measure.Rec)), Arrays.toString(measures.get(Measure.MAPS)));

        } else {
            evalInfo = String.format("MAE: %.6f, RMSE: %.6f, NAME: %.6f, rMAE: %.6f, rRMSE: %.6f, MPE: %.6f", measures.get(Measure.MAE)[0],
                    measures.get(Measure.RMSE)[0], measures.get(Measure.NMAE)[0], measures.get(Measure.rMAE)[0],
                    measures.get(Measure.rRMSE)[0], measures.get(Measure.MPE)[0]);

            // for some graphic models
            if (measures.containsKey(Measure.Perplexity)) {
                evalInfo += String.format(",%.6f", measures.get(Measure.Perplexity)[0]);
            }
        }

        return evalInfo;
    }

    /**
     * @return the evaluation results of rating predictions
     */
    protected Map<Measure, Double[]> evalRatings() throws Exception {

        List<String> preds = null;
        String toFile = null;
        if (isResultsOut) {
            preds = new ArrayList<String>(1500);
            preds.add("userId\titemId\tcontexts\trating\tprediction"); // optional: file header
            Date now = new Date(System.currentTimeMillis());
            SimpleDateFormat sdf = new SimpleDateFormat("ddMMyyHHmmssSSS");
            toFile = workingPath + algoName + "-rating-predictions" + foldInfo + "-" + sdf.format(now) + ".txt"; // the output-file name
            FileIO.deleteFile(toFile); // delete possibly old files
        }

        double sum_maes = 0, sum_mses = 0, sum_r_maes = 0, sum_r_rmses = 0, sum_perps = 0;
        int numCount = 0, numPEs = 0;

        for (MatrixEntry me : testMatrix) {
            double rate = me.get();

            int ui = me.row();
            int ctx = me.column();
            int u=rateDao.getUserIdFromUI(ui);
            int j=rateDao.getItemIdFromUI(ui);

            double pred = predict(u,j, ctx, true);
            if (Double.isNaN(pred))
                continue;

            // perplexity: for some graphic model
            double perp = perplexity(u, j, pred);
            sum_perps += perp;

            // rounding prediction to the closest rating level
            double rPred = Math.round(pred / minRate) * minRate;

            double err = Math.abs(rate - pred); // absolute predictive error
            double r_err = Math.abs(rate - rPred);

            sum_maes += err;
            sum_mses += err * err;

            sum_r_maes += r_err;
            sum_r_rmses += r_err * r_err;

            numCount++;

            // output predictions
            if (isResultsOut) {
                // restore back to the original user/item id
                preds.add(rateDao.getUserId(u) + "\t" + rateDao.getItemId(j) + "\t" + rateDao.getContextSituationFromInnerId(ctx) + "\t" + rate + "\t" + (float) pred);
                if (preds.size() >= 1000) {
                    FileIO.writeList(toFile, preds, true);
                    preds.clear();
                }
            }
        }

        if (isResultsOut && preds.size() > 0) {
            FileIO.writeList(toFile, preds, true);
            Logs.debug("{}{} has writeen rating predictions to {}", algoName, foldInfo, toFile);
        }

        double mae = sum_maes / numCount;
        double rmse = Math.sqrt(sum_mses / numCount);

        double r_mae = sum_r_maes / numCount;
        double r_rmse = Math.sqrt(sum_r_rmses / numCount);

        Map<Measure, Double[]> measures = new HashMap<>();
        measures.put(Measure.MAE, new Double[] { mae });
        // normalized MAE: useful for direct comparison among different data sets with distinct rating scales
        measures.put(Measure.NMAE, new Double[] { mae / (maxRate - minRate) });
        measures.put(Measure.RMSE, new Double[] { rmse });

        // MAE and RMSE after rounding predictions to the closest rating levels
        measures.put(Measure.rMAE, new Double[] { r_mae });
        measures.put(Measure.rRMSE, new Double[] { r_rmse });

        // measure zero-one loss
        measures.put(Measure.MPE, new Double[] { (numPEs + 0.0) / numCount });

        // perplexity
        if (sum_perps > 0) {
            measures.put(Measure.Perplexity, new Double[] { Math.exp(sum_perps / numCount) });
        }

        return measures;
    }

    public void setIdMappers(Table<Integer, Integer, Integer> uid, Table<Integer, Integer, Integer> iid)
    {
        this.userIdMapper = uid;
        this.itemIdMapper = iid;

        if(uid!=null) isUserSplitting=true;
        if(iid!=null) isItemSplitting=true;

        this.algoName=this.getAlgoName();
    }

    protected String getAlgoName()
    {
        if(isUserSplitting && isItemSplitting)
            this.algoName = "UISplitting-"+this.algoName;
        else
        {
            if(isUserSplitting==true)
                this.algoName = "UserSplitting-"+this.algoName;
            else if(isItemSplitting==true)
                this.algoName = "ItemSplitting-"+this.algoName;
        }
        return this.algoName;
    }

    protected librec.data.SparseMatrix createTraditionalSparseMatrixBySplitting()
    {
        Table<Integer, Integer, Double> newtable = HashBasedTable.create();
        Multimap<Integer, Integer> colMap = HashMultimap.create();
        HashMultimap<String, Double> records=HashMultimap.create();
        int maxUser=-1, maxItem=-1;

        for(MatrixEntry me:this.trainMatrix){
            int ui=me.row();
            int ctx=me.column();
            int u=rateDao.getUserIdFromUI(ui);
            int j=rateDao.getItemIdFromUI(ui);

            if(isUserSplitting)
                u = userIdMapper.contains(u,ctx) ? userIdMapper.get(u,ctx) : u;
            if(isItemSplitting)
                j = itemIdMapper.contains(j,ctx) ? itemIdMapper.get(j,ctx) : j;
            records.put(u + "," + j, me.get());
            if(u>maxUser) maxUser=u;
            if(j>maxItem) maxItem=j;
        }

        for(String key:records.keySet()){
            String[] ids=key.split(",");
            int u=Integer.valueOf(ids[0]);
            int j=Integer.valueOf(ids[1]);
            newtable.put(u, j, Stats.mean(records.get(key)));
            colMap.put(j, u);
        }
        maxUser++;
        maxItem++;
        this.numUsers= (numUsers<maxUser)? maxUser: numUsers;
        this.numItems= (numItems<maxItem)? maxItem: numItems;

        //Logs.debug("Fold["+fold+"]: numUsers = "+numUsers + ", numItems = "+numItems);
        librec.data.SparseMatrix sm=new librec.data.SparseMatrix(numUsers, numItems, newtable, colMap);
        return sm;
    }



    protected double perplexity(int u, int j, double r) throws Exception {
        return 0;
    }

    /**
     * @return the evaluation results of ranking predictions
     */

    protected Map<Measure, Double[]> evalRankings() throws Exception {

        HashMap<Integer, HashMultimap<Integer, Integer>> uciList=rateDao.getUserCtxList(testMatrix);


        HashMap<Integer, HashMultimap<Integer, Integer>> uciList_train = null;
        if (!ratedItemsFilter.equals("none")) {
            // If filtering is enabled, get all the ratings in the training set
            uciList_train = rateDao.getUserCtxList(trainMatrix);
        }
        int capacity = uciList.keySet().size();

        // initialization capacity to speed up
        List<Double> ds5 = new ArrayList<>(isDiverseUsed ? capacity : 0);
        List<Double> ds10 = new ArrayList<>(isDiverseUsed ? capacity : 0);

        List<Double> precs5 = new ArrayList<>(capacity);
        List<Double> precs10 = new ArrayList<>(capacity);
        List<Map<Integer, Double>> precs = new ArrayList<>(capacity);
        List<Double> recalls5 = new ArrayList<>(capacity);
        List<Double> recalls10 = new ArrayList<>(capacity);
        List<Map<Integer, Double>> recalls = new ArrayList<>(capacity);
        List<Map<Integer, Double>> aps_at = new ArrayList<>(capacity);
        List<Double> aps = new ArrayList<>(capacity);
        List<Double> rrs = new ArrayList<>(capacity);
        List<Double> aucs = new ArrayList<>(capacity);
        List<Double> ndcgs = new ArrayList<>(capacity);

        // candidate items for all users: here only training items
        // use HashSet instead of ArrayList to speedup removeAll() and contains() operations: HashSet: O(1); ArrayList: O(log n).
        Set<Integer> candItems = rateDao.getItemList(trainMatrix);

        List<String> preds = null;
        List<String> significanceData = null;
        String toFile = null;
        String sigToFile = null;
        int numTopNRanks = numRecs < 0 ? 10 : numRecs;
        String[] fileName = cf.getPath("dataset.ratings").split("/");
        if (isResultsOut) {
            preds = Collections.synchronizedList(new ArrayList<String>());
            preds.add("# userId: recommendations in (itemId, ranking score) pairs, where a correct recommendation is denoted by symbol *."); // optional: file header
            Date now = new Date(System.currentTimeMillis());
            SimpleDateFormat sdf = new SimpleDateFormat("ddMMyyHHmmssSSS");
            toFile = workingPath
                    + String.format("top-%d-%s-%s-%s.txt", numTopNRanks, algoName, fileName[fileName.length-1], sdf.format(now)); // the output-file name
            FileIO.deleteFile(toFile); // delete possibly old files
        }
        if (significanceOutput) {
            significanceData = Collections.synchronizedList(new ArrayList<String>());
            significanceData.add("userid,metric,N,value"); // optional: file header
            Date now = new Date(System.currentTimeMillis());
            SimpleDateFormat sdf = new SimpleDateFormat("ddMMyyHHmmssSSS");
            sigToFile = workingPath
                    + String.format("significance-%s-%s-%s.txt", algoName, fileName[fileName.length-1], sdf.format(now)); // the output-file name
            FileIO.deleteFile(sigToFile); // delete possibly old files
        }

        if (verbose)
            Logs.debug("{}{} has candidate items: {}", algoName, foldInfo, candItems.size());

        // ignore items for all users: most popular items
        if (numIgnore > 0) {
            List<Map.Entry<Integer, Integer>> itemDegs = new ArrayList<>();
            for (Integer j : candItems) {
                itemDegs.add(new SimpleImmutableEntry<Integer, Integer>(j, rateDao.getRatingCountByItem(trainMatrix,j)));
            }
            Lists.sortList(itemDegs, true);
            int k = 0;
            for (Map.Entry<Integer, Integer> deg : itemDegs) {

                // ignore these items from candidate items
                candItems.remove(deg.getKey());
                if (++k >= numIgnore)
                    break;
            }
        }


        class UserWorker extends Thread {
            private final BlockingQueue<Integer> queue;
            private final HashMap<Integer, HashMultimap<Integer, Integer>> uciList;
            private final HashMap<Integer, HashMultimap<Integer, Integer>> uciList_train;
            private final Set<Integer> candItems;
            private final int capacity;
            private final int numTopNRanks;

            private Recommender par;

            // Synchronized lists
            List<String> preds;
            List<String> significanceData;
            List<Double> ds5;
            List<Double> ds10;
            List<Double> precs5;
            List<Double> precs10;
            List<Map<Integer, Double>> precs;
            List<Map<Integer, Double>> recalls;
            List<Map<Integer, Double>> aps_at;
            List<Double> recalls5;
            List<Double> recalls10;

            List<Double> aucs;
            List<Double> ndcgs;
            List<Double> aps;
            List<Double> rrs;

            public UserWorker(BlockingQueue<Integer> q, HashMap<Integer, HashMultimap<Integer, Integer>> uciList,
                              HashMap<Integer, HashMultimap<Integer, Integer>> uciList_train, Set<Integer> candItems,
                              int capacity, int numTopNRanks, List<String> preds, List<String> significanceData, List<Double> ds5, List<Double> ds10,
                              List<Double> precs5, List<Double> precs10, List<Map<Integer, Double>> precs,
                              List<Map<Integer, Double>> recalls, List<Map<Integer, Double>> aps_at,
                              List<Double> recalls5, List<Double> recalls10, List<Double> aucs, List<Double> ndcgs,
                              List<Double> aps, List<Double> rrs, Recommender par) {
                this.queue = q;
                this.uciList = uciList;
                this.uciList_train = uciList_train;
                this.candItems = candItems;
                this.capacity = capacity;
                this.numTopNRanks = numTopNRanks;
                this.par = par;

                this.preds = preds;
                this.significanceData = significanceData;
                this.ds5 = ds5;
                this.ds10 = ds10;
                this.precs5 = precs5;
                this.precs10 = precs10;
                this.precs = precs;
                this.recalls = recalls;
                this.aps_at = aps_at;
                this.recalls5 = recalls5;
                this.recalls10 = recalls10;
                this.aucs = aucs;
                this.ndcgs = ndcgs;
                this.aps = aps;
                this.rrs = rrs;
            }

            public void run() {
                try {
                    Logs.info("Thread " + Thread.currentThread().getId() + " started");
                    while (true) {
                        Integer u = queue.take();
                        if (u == Integer.MIN_VALUE) {
                            Logs.info("Thread " + Thread.currentThread().getId() + " stopped");
                            return;
                        } else {
                            try {
                                testUser(u);
                            }
                            catch (Exception e) {
                                Logs.error("Exception {} in testUser", e);
                            }
                        }
                    }
                }
                catch (InterruptedException ie) {

                    // Nothing
                }
                catch (Exception e) {
                    Logs.error("Exception {{}}", e);
                }
            }

            private void testUser(int u) throws Exception {
                Multimap<Integer, Integer> cis = uciList.get(u);

                int c_capacity = cis.keySet().size();

                List<Double> c_ds5 = new ArrayList<>(isDiverseUsed ? c_capacity : 0);
                List<Double> c_ds10 = new ArrayList<>(isDiverseUsed ? c_capacity : 0);

                List<Double> c_precs5 = new ArrayList<>(c_capacity);
                List<Double> c_precs10 = new ArrayList<>(c_capacity);
                List<Map<Integer, Double>> c_precs = new ArrayList<>(c_capacity);
                List<Double> c_recalls5 = new ArrayList<>(c_capacity);
                List<Double> c_recalls10 = new ArrayList<>(c_capacity);
                List<Map<Integer, Double>> c_recalls = new ArrayList<>(c_capacity);
                List<Map<Integer, Double>> c_aps_at = new ArrayList<>(c_capacity);
                List<Double> c_aps = new ArrayList<>(c_capacity);
                List<Double> c_rrs = new ArrayList<>(c_capacity);
                List<Double> c_aucs = new ArrayList<>(c_capacity);
                List<Double> c_ndcgs = new ArrayList<>(c_capacity);

                // Create a ratedItems set to enable filtering. Remains empty if no filter enabled (items.filter=none)
                Set<Integer> ratedItems = new HashSet<Integer>();

                HashMultimap<Integer, Integer> cList_train = null;
                if (!ratedItemsFilter.equals("none")) {
                    // Get the ratings in the training set by the current user
                    cList_train = (uciList_train.containsKey(u)) ? uciList_train.get(u) : HashMultimap.<Integer, Integer>create();
                    if (ratedItemsFilter.equals("user")) {
                        // If the filter is on user, then this is the ratedItems set
                        ratedItems = Sets.newHashSet(cList_train.values());
                    }
                }

                // for each ctx
                for (int c : cis.keySet()) {

                    if (verbose && ((u + 1) % 100 == 0))
                        Logs.debug("{}{} evaluates progress: {} / {}", algoName, foldInfo, u + 1, capacity);

                    // number of candidate items for all users
                    int numCands = candItems.size();

                    // get positive items from test matrix
                    Collection<Integer> posItems = cis.get(c);
                    List<Integer> correctItems = new ArrayList<>();

                    // intersect with the candidate items
                    for (Integer j : posItems) {
                        if (candItems.contains(j))
                            correctItems.add(j);
                    }

                    if (correctItems.size() == 0)
                        continue; // no testing data for user u

                    if (ratedItemsFilter.equals("user-context")) {
                        // If filter is on user-context, then get the ratings in this context if present in cList_train
                        if (cList_train.containsKey(c)) {
                            ratedItems = cList_train.get(c);
                        }
                    }

                    // Log ratedItems
                /* String ratedItemsString = "";
                for (Integer i : ratedItems) {
                    ratedItemsString += rateDao.getItemId(i).toString() + " ";
                }
                Logs.debug("ratedItems user {} (filter {}): {}", rateDao.getUserId(u).toString(), ratedItemsFilter, ratedItemsString); */

                    // predict the ranking scores (unordered) of all candidate items
                    List<Map.Entry<Integer, Double>> itemScores = new ArrayList<>(Lists.initSize(candItems));
                    for (final Integer j : candItems) {
                        if (ratedItemsFilter.equals("none") || !ratedItems.contains(j)) {
                            final double rank = par.ranking(u, j, c);
                            if (!Double.isNaN(rank)) {
                                itemScores.add(new SimpleImmutableEntry<Integer, Double>(j, rank));
                            }
                        } else {
                            numCands--;
                        }
                    }

                    if (itemScores.size() == 0)
                        continue; // no recommendations available for user u

                    // order the ranking scores from highest to lowest: List to preserve orders
                    Lists.sortList(itemScores, true);
                    List<Map.Entry<Integer, Double>> recomd = (numRecs <= 0 || itemScores.size() <= numRecs) ? itemScores
                            : itemScores.subList(0, numRecs);

                    List<Integer> rankedItems = new ArrayList<>();
                    StringBuilder sb = new StringBuilder();
                    int count = 0;
                    for (Map.Entry<Integer, Double> kv : recomd) {
                        Integer item = kv.getKey();
                        rankedItems.add(item);

                        if (isResultsOut && count < numTopNRanks) {
                            // restore back to the original item id
                            sb.append("(").append(rateDao.getItemId(item));

                            if (posItems.contains(item))
                                sb.append("*"); // indicating correct recommendation

                            sb.append(", ").append(kv.getValue().floatValue()).append(")");

                            if (++count >= numTopNRanks)
                                break;
                            if (count < numTopNRanks)
                                sb.append(", ");
                        }
                    }

                    int numDropped = numCands - rankedItems.size();
                    double AUC = Measures.AUC(rankedItems, correctItems, numDropped);
                    double AP = Measures.AP(rankedItems, correctItems);
                    double nDCG = Measures.nDCG(rankedItems, correctItems);
                    double RR = Measures.RR(rankedItems, correctItems);

                    List<Integer> cutoffs = new LinkedList<Integer>();
                    for (int i = 1; i <= numTopNRanks; i++) {
                        cutoffs.add(i);
                    }
                    if (numTopNRanks < 5) cutoffs.add(5);
                    if (numTopNRanks < 10) cutoffs.add(10);

                    Map<Integer, Double> precsAt = Measures.PrecAt(rankedItems, correctItems, cutoffs);
                    Map<Integer, Double> recallsAt = Measures.RecallAt(rankedItems, correctItems, cutoffs);
                    Map<Integer, Double> APAt = APAt(rankedItems, correctItems, cutoffs);

                    c_precs5.add(precsAt.get(5));
                    c_precs10.add(precsAt.get(10));
                    c_precs.add(precsAt);
                    c_recalls5.add(recallsAt.get(5));
                    c_recalls10.add(recallsAt.get(10));
                    c_recalls.add(recallsAt);
                    c_aps_at.add(APAt);

                    c_aucs.add(AUC);
                    c_aps.add(AP);
                    c_rrs.add(RR);
                    c_ndcgs.add(nDCG);


                    // diversity
                    if (isDiverseUsed) {
                        double d5 = diverseAt(rankedItems, 5);
                        double d10 = diverseAt(rankedItems, 10);

                        c_ds5.add(d5);
                        c_ds10.add(d10);
                    }

                    // output predictions
                    if (isResultsOut) {
                        // restore back to the original user id
                        preds.add(rateDao.getUserId(u) + ", " + rateDao.getContextSituationFromInnerId(c) + ": " + sb.toString());
                        /*if (preds.size() >= 1000) {
                            FileIO.writeList(toFile, preds, true);
                            preds.clear();
                        }*/
                    }
                } // end a context

                // calculate metrics for a specific user averaged by contexts
                ds5.add(isDiverseUsed ? Stats.mean(c_ds5) : 0.0);
                ds10.add(isDiverseUsed ? Stats.mean(c_ds10) : 0.0);
                precs5.add(Stats.mean(c_precs5));
                precs10.add(Stats.mean(c_precs10));
                if (c_precs.size() > 0) {
                    Map<Integer, Double> map_precs = meanOverListOfMaps(c_precs);
                    Map<Integer, Double> map_recalls = meanOverListOfMaps(c_recalls);
                    Map<Integer, Double> map_maps = meanOverListOfMaps(c_aps_at);
                    precs.add(map_precs);
                    recalls.add(map_recalls);
                    aps_at.add(map_maps);
                    if (significanceOutput) {
                        String uid = rateDao.getUserId(u);
                        significanceData.add(uid + ",prec,1," + map_precs.get(1));
                        significanceData.add(uid + ",prec,5," + map_precs.get(5));
                        significanceData.add(uid + ",prec,10," + map_precs.get(10));
                        significanceData.add(uid + ",prec,20," + map_precs.get(20));
                        significanceData.add(uid + ",recall,1," + map_recalls.get(1));
                        significanceData.add(uid + ",recall,5," + map_recalls.get(5));
                        significanceData.add(uid + ",recall,10," + map_recalls.get(10));
                        significanceData.add(uid + ",recall,20," + map_recalls.get(20));
                        significanceData.add(uid + ",map,1," + map_maps.get(1));
                        significanceData.add(uid + ",map,5," + map_maps.get(5));
                        significanceData.add(uid + ",map,10," + map_maps.get(10));
                        significanceData.add(uid + ",map,20," + map_maps.get(20));
                    }
                }
                recalls5.add(Stats.mean(c_recalls5));
                recalls10.add(Stats.mean(c_recalls10));

                aucs.add(Stats.mean(c_aucs));
                ndcgs.add(Stats.mean(c_ndcgs));
                aps.add(Stats.mean(c_aps));
                rrs.add(Stats.mean(c_rrs));
            }
        }



        // Get results for each user in multithreaded way
        BlockingQueue<Integer> queue = new SynchronousQueue<Integer>();

        int cores = Runtime.getRuntime().availableProcessors();
        Logs.info("Starting calculation over " + cores + " cores");
        for (int i = 0; i < cores; i++) {
            UserWorker w = new UserWorker(queue, uciList, uciList_train, candItems, capacity, numTopNRanks,
                    preds, significanceData, ds5, ds10, precs5, precs10, precs, recalls, aps_at, recalls5, recalls10, aucs, ndcgs, aps, rrs, this);
            w.start();
        }
        // for each test user
        for (int u:uciList.keySet()) {
            queue.put(u);
        }
        for (int i = 0; i < cores; i++) {
            queue.put(Integer.MIN_VALUE);
        }

        Logs.info("Continue in single thread");

        // write results out first
        if (isResultsOut && preds != null && preds.size() > 0) {
            FileIO.writeList(toFile, preds, true);
            if (significanceOutput) {
                FileIO.writeList(sigToFile, significanceData, true);
                Logs.debug("Written significance data to {}", sigToFile);
            }

            Logs.debug("{}{} has writeen item recommendations to {}", algoName, foldInfo, toFile);
        }

        // measure the performance
        Map<Measure, Double[]> measures = new HashMap<>();
        measures.put(Measure.D5, isDiverseUsed ? new Double[] { Stats.mean(ds5) } : new Double[] { 0.0 });
        measures.put(Measure.D10, isDiverseUsed ? new Double[] { Stats.mean(ds10) } : new Double[] { 0.0 });
        measures.put(Measure.Pre5, new Double[] { Stats.mean(precs5) });
        measures.put(Measure.Pre10, new Double[] { Stats.mean(precs10) });
        measures.put(Measure.Rec5, new Double[] { Stats.mean(recalls5) });
        measures.put(Measure.Rec10, new Double[] { Stats.mean(recalls10) });
        Double[] precs_array = new Double[numTopNRanks];
        Double[] recalls_array = new Double[numTopNRanks];
        Double[] maps_array = new Double[numTopNRanks];
        if (precs.size() > 0) {
            Map<Integer, Double> precs_map = meanOverListOfMaps(precs);
            Map<Integer, Double> recalls_map = meanOverListOfMaps(recalls);
            Map<Integer, Double> maps_map = meanOverListOfMaps(aps_at);
            for (int i = 1; i <= numTopNRanks; i++) {
                precs_array[i - 1] = precs_map.get(i);
                recalls_array[i - 1] = recalls_map.get(i);
                maps_array[i - 1] = maps_map.get(i);
            }
        }
        else {
            for (int i = 1; i <= numTopNRanks; i++) {
                precs_array[i - 1] = 0.0;
                recalls_array[i - 1] = 0.0;
                maps_array[i - 1] = 0.0;
            }
        }
        measures.put(Measure.Pre, precs_array);
        measures.put(Measure.Rec, recalls_array);
        measures.put(Measure.AUC, new Double[] { Stats.mean(aucs) });
        measures.put(Measure.NDCG, new Double[] { Stats.mean(ndcgs) });
        measures.put(Measure.MAP, new Double[] { Stats.mean(aps) });
        measures.put(Measure.MAPS, maps_array);
        measures.put(Measure.MRR, new Double[] { Stats.mean(rrs) });

        return measures;
    }

    public static <T> Map<Integer, Double> APAt(List<T> rankedList, List<T> groundTruth, List<Integer> ns) {
        HashMap ap_at_n = new HashMap();
        Iterator var5 = ns.iterator();

        while(var5.hasNext()) {
            int n = ((Integer)var5.next()).intValue();
            ap_at_n.put(Integer.valueOf(n), Double.valueOf(APAt(rankedList, groundTruth, n)));
        }

        return ap_at_n;
    }

    public static <T> double APAt(List<T> rankedList, List<T> groundTruth, int at) {
        int hits = 0;
        double sum_precs = 0.0D;
        int n = 0;

        for(int m = Math.min(rankedList.size(), at); n < m; ++n) {
            Object item = rankedList.get(n);
            if(groundTruth.contains(item)) {
                ++hits;
                sum_precs += (double)hits / ((double)n + 1.0D);
            }
        }

        if(hits > 0) {
            return sum_precs / (double)groundTruth.size();
        } else {
            return 0.0D;
        }
    }


    private Map<Integer, Double> meanOverListOfMaps(List<Map<Integer, Double>> list) {
        Map<Integer, Double> list_avg = new HashMap<Integer, Double>();
        for (int i=1; i<=list.get(0).size(); i++) {
            double[] i_values = new double[list.size()];
            for (int j=0; j<list.size(); j++) {
                i_values[j] = list.get(j).get(i);
            }
            double i_avg = Stats.mean(i_values);
            list_avg.put(i, i_avg);
        }
        return list_avg;
    }

    /**
     * determine whether the rating of a user-item (u, j) is used to predicted
     *
     */
    protected boolean isTestable(int u, int j) {
        String uiid=u+","+j;
        int rowid=rateDao.getUserItemId(uiid);
        switch (view) {
            case "cold-start":
                return trainMatrix.rowSize(rowid) < 5 ? true : false;
            case "all":
            default:
                return true;
        }
    }

    /**
     *
     * @param rankedItems
     *            the list of ranked items to be recommended
     * @param cutoff
     *            cutoff in the list
     *
     * @return diversity at a specific cutoff position
     */
    protected double diverseAt(List<Integer> rankedItems, int cutoff) {

        int num = 0;
        double sum = 0.0;
        for (int id = 0; id < cutoff; id++) {
            int i = rankedItems.get(id);
            SparseVector iv = trainMatrix.column(i);

            for (int jd = id + 1; jd < cutoff; jd++) {
                int j = rankedItems.get(jd);

                double corr = corrs.get(i, j);
                if (corr == 0) {
                    // if not found
                    corr = correlation(iv, trainMatrix.column(j));
                    if (!Double.isNaN(corr))
                        corrs.set(i, j, corr);
                }

                if (!Double.isNaN(corr)) {
                    sum += (1 - corr);
                    num++;
                }
            }
        }

        return 0.5 * (sum / num);
    }

    protected double ranking(int u, int j, int c) throws Exception {
        return predict(u, j, c, false);
    }



    protected SymmMatrix buildCorrs(boolean isUser) {
        Logs.debug("Build {} similarity matrix ...", isUser ? "user" : "item");

        int count = isUser ? numUsers : numItems;
        SymmMatrix corrs = new SymmMatrix(count);

        for (int i = 0; i < count; i++) {
            SparseVector iv = isUser ? train.row(i) : train.column(i);
            if (iv.getCount() == 0)
                continue;
            // user/item itself exclusive
            for (int j = i + 1; j < count; j++) {
                SparseVector jv = isUser ? train.row(j) : train.column(j);
                if(jv.getCount() == 0 )
                    continue;
                double sim = correlation(iv, jv);

                if (!Double.isNaN(sim))
                    corrs.set(i, j, sim);
            }
        }

        return corrs;
    }


    /**
     * initilize recommender model
     */
    protected void initModel() throws Exception {
        if(isItemSplitting==false && isUserSplitting==false && isCARSRecommender==false) // return a 2D rating matrix based on original CARS data set
            train=rateDao.toTraditionalSparseMatrix(trainMatrix);
        else if(isCARSRecommender==false) // return a 2D rating matrix based on transformation from splitting mappers
            train=this.createTraditionalSparseMatrixBySplitting();
    }

    /**
     * Learning method: override this method to build a model, for a model-based method. Default implementation is
     * useful for memory-based methods.
     *
     */
    protected void buildModel() throws Exception {
    }

    /**
     * After learning model: release some intermediate data to avoid memory leak
     */
    protected void postModel() throws Exception {
    }

    /**
     * Serializing a learned model (i.e., variable data) to files.
     */
    protected void saveModel() throws Exception {
    }

    /**
     * Deserializing a learned model (i.e., variable data) from files.
     */
    protected void loadModel() throws Exception {
    }

    private void printAlgoConfig() {
        String algoInfo = toString();

        Class<? extends Recommender> cl = this.getClass();
        // basic annotation
        String algoConfig = cl.getAnnotation(Configuration.class).value();

        // additional algorithm-specific configuration
        if (cl.isAnnotationPresent(AddConfiguration.class)) {
            AddConfiguration add = cl.getAnnotation(AddConfiguration.class);

            String before = add.before();
            if (!Strings.isNullOrEmpty(before))
                algoConfig = before + ", " + algoConfig;

            String after = add.after();
            if (!Strings.isNullOrEmpty(after))
                algoConfig += ", " + after;
        }

        if (!algoInfo.isEmpty()) {
            if (!algoConfig.isEmpty())
                Logs.debug("{}: [{}] = [{}]", algoName, algoConfig, algoInfo);
            else
                Logs.debug("{}: {}", algoName, algoInfo);
        }
    }

    /**
     * logistic function g(x)
     */
    protected double g(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    /**
     * gradient value of logistic function g(x)
     */
    protected double gd(double x) {
        return g(x) * g(-x);
    }

    /**
     * Check if ratings have been binarized; useful for methods that require binarized ratings;
     */
    protected void checkBinary() {
        if (binThold < 0) {
            Logs.error("val.binary.threshold={}, ratings must be binarized first! Try set a non-negative value.",
                    binThold);
            System.exit(-1);
        }
    }

    public void run() {
        try {
            execute();
        } catch (Exception e) {
            // capture error message
            Logs.error(e.getMessage());

            e.printStackTrace();
        }
    }
}

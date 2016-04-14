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

package carskit.alg.baseline.cf;


import carskit.generic.Recommender;
import happy.coding.io.LineConfiger;
import happy.coding.io.Strings;

import org.apache.commons.lang3.StringUtils;


import java.io.BufferedReader;
import java.io.FileReader;
import java.text.SimpleDateFormat;
import java.util.*;
import happy.coding.io.Logs;

/**
 * UserKNN: Resnick, Paul, et al. "GroupLens: an open architecture for collaborative filtering of netnews." Proceedings of the 1994 ACM conference on Computer supported cooperative work. ACM, 1994.
 * <p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 * Implemented using python (much faster) by Bas van Kortenhof
 * @author Yong Zheng
 *
 */

public class UserKNNPy extends Recommender {

    // user: nearest neighborhood
    private HashMap<String, Double> similarities;


    public UserKNNPy(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "UserKNNTree";

    }


    @Override
    protected void initModel() throws Exception {
        super.initModel();
        String trainPath = cf.getPath("dataset.ratings");
        String testPath = "";
        LineConfiger evalOptions = cf.getParamOptions("evaluation.setup");
        if (evalOptions.getMainParam().toLowerCase().equals("test-set")) {
            testPath = evalOptions.getString("-f");
        } else {
            throw new Exception("evaluation test set has to be defined");
        }
        String simUtil = cf.getPath("usersimutility");

        String[] fileName = cf.getPath("dataset.ratings").split("/");
        Date now = new Date(System.currentTimeMillis());
        SimpleDateFormat sdf = new SimpleDateFormat("ddMMyyHHmmssSSS");
        String simFile = workingPath
                + String.format("neighbors-%d-%s-%s-%s.txt", knn, algoName, fileName[fileName.length-1], sdf.format(now));

        try {
            Logs.debug("Starting kNN script");
            String[] script = new String[]{
                    "python",
                    simUtil,
                    trainPath,
                    testPath,
                    simFile,
                    Integer.toString(knn)
                    //String.format("%s %s %s %d", trainPath, testPath, simFile, knn)
            };
            Logs.debug("Script command: {}", StringUtils.join(script, " "));
            Process p = Runtime.getRuntime().exec(StringUtils.join(script, " "));
            int res = p.waitFor();
            similarities = new HashMap<String, Double>();

            BufferedReader br = new BufferedReader(new FileReader(simFile));
            try {
                String line = br.readLine();

                while (line != null) {
                    String[] parts = line.split("@@");
                    similarities.put(parts[0] + "@@" + parts[1], Double.valueOf(parts[2]));

                    line = br.readLine();
                }
            } finally {
                br.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {

        if(isUserSplitting)
            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
        if(isItemSplitting)
            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;

            return predict(u,j);
    }

    @Override
    protected double predict(int u, int j) throws Exception {

        String index = rateDao.getUserId(u) + "@@" + rateDao.getItemId(j);
        if (similarities.containsKey(index)) {
            return similarities.get(index);
        }
        else {
            return 0;
        }
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
    }

}

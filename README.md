# CARSKit

### Introduction
-------------------

**CARSKit** (https://github.com/irecsys/CARSKit/) is an open-source Java-based context-aware recommendation engine, where it can be used, modified and distributed under the terms of the GNU General Public License. (Java version 1.7 or higher required). It is specifically designed for context-aware recommendations. 


### GPL License
-------------------

CARSKit is [free software](http://www.gnu.org/philosophy/free-sw.html): you can redistribute it and/or modify it under the terms of the [GNU General Public License (GPL)](http://www.gnu.org/licenses/gpl.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. CARSKit is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with CARSKit. If not, see http://www.gnu.org/licenses/.

### Architecture and Design
----------------------------

![CARSKit Structure](http://students.depaul.edu/~yzheng8/images/CARSKit_Design.png)

### Algorithms
---------------

* **Traditional recommendation algorithms**: The implementations of those algorithms (such as UserKNN, BiasedMF, SVD++, SLIM, etc) are from LibRec-V1.3. Those algorithms can be used in two ways: 1). run a traditional recommendation algorithm directly on the context-aware data set to compete with the context-aware recommendation algorithms; 2). run contextual recommendation algorithms based on transformation, e.g., run a traditional recommendation algorithm after data transformation (e.g., by item splitting).
* **Context-aware recommendation algorithms**: CARSKit simply divides it into two categories: Transformation Algorithms and Adaptation Algorithms. The transformation algorithms try to convert the mulidimensional recommendation problem into traditional 2-dimensional problem, so that the traditional recommendation algorithms can still be used. In contrast to those algorithms based on transformation algorithms, the adaptation algorithms focus on their effect on building algorithms by adapting to the multidimensional rating space. Most of the algorithms inside belong to the contextual modeling algorithms, such as Context-aware Matrix Facatorization (CAMF), Tensor Factorization (TF), Contextual Sparse Linear Method (CSLIM), Factorization Machines (FM), etc.

### Reference
-------------

Please cite the following papers if you use CARSKit in your research:

1. Yong Zheng, Bamshad Mobasher, Robin Burke. "CARSKit: A Java-Based Context-aware Recommendation Engine", Proceedings of the 15th IEEE International Conference on Data Mining (ICDM) Workshops, Atlantic City, NJ, USA, Nov 2015

If you use CARSKit in your research papers, please let us know. And I will create a list of publications using this toolkit.

### Downloads
* **[CARSKit-v0.2.0.jar](https://raw.githubusercontent.com/irecsys/CARSKit/master/jar/CARSKit-v0.2.0.jar)**
* **[CARSKit-v0.1.0.jar](https://raw.githubusercontent.com/irecsys/CARSKit/master/jar/CARSKit-v0.1.0.jar)**

### Data Sets
--------------

A list of context-aware data sets can be found here: http://tiny.cc/contextdata <br/>

### User's Guide
--------------
The guide can be found and downloaded from http://arxiv.org/abs/1511.03780<br/>
Note: if there are missing values or EMPTY values in the context information, you should mark those values as NA (i.e., Not Available) which will be viewed as a special context condition in the data.

### Acknowledgement
--------------------

I would like to show our gratitude to Dr. Guibing Guo (the author of LibRec) for his comments and suggestions on the development of CARSKit.

### Release Notes
------------------
**2016/02, Version 0.3.0** (Upcoming)
* **What's New**:
 - add more context-aware recommendation algorithms
 - add some functions for prediction and ranking
 - rename some packages and algorithms
 - optimize the coding

**2015/11/09, Version 0.2.0**
* **What's New**:
 - add Tensor Factorization (CPTF) into CARSKit
 - update API documents and the configuration file setting.conf
 - release user's guide, "A User's Guide to CARSKit"
 - optimize the coding


**2015/09/25, Initial Release Version 0.1.0**
* **Notes and updates**: 
  - This is the initial release. 
  - Most algorithms and functions have been completed except the tensor factorization (TF). 
  - For TF, please refer to the latest version of LibRec.







## Replication Package


This is the replication package for the paper: Bridging the MAS approach Gap: A Network Flow
Analysis Approach for Malware Detection

### Abstract

The Android platform’s dominance in the smartphone market has made it a prime target for cyberattacks, particularly through repackaged applications—popular apps modified to embed malicious code that exploits user data. Prior research has shown that the MAS approach, which relies on calls to sensitive APIs for malware classification, performs poorly against certain malware families that employ advanced evasion techniques. To address this limitation, we introduce DroidXPflow, an extension of the MAS approach that classifies repackaged applications as benign or malicious using machine learning models trained on collected network flow data. Like the MAS approach, DroidXPflow employs an automated test generator to execute apps; however, instead of analyzing calls to sensitive APIs, it captures and analyzes the resulting network traffic. We evaluate multiple learning algorithms on a comprehensive dataset of benign and malicious applications. Experimental results demonstrate that DroidXPflow significantly improves detection accuracy over the MAS approach, confirming that network traffic provides a valuable signal for identifying sophisticated malware and strengthening the overall security of Android applications. Furthermore, DroidXPflow successfully detects malware from families that prior studies have shown the MAS approach consistently failed to identify.

### Malware Dataset

At our work, we used a curated dataset of 4,076 app pairs (original/repackaged) from the study by Costa et al., which is available at CSV [file](https://github.com/droidxp/ML/blob/master/large_ds.csv). Using this dataset, we utilized the [DroidXP](https://github.com/droidxp/benchmark) infrastructure for data collection. With DroidXP, we collected all PCAP files during the test executions, which were performed by the test generator tool, Droidbot, for 180 seconds for each app (a total of 5,887 apps). After this exploratory step, we leveraged [CICflowMeter](https://github.com/ahlashkari/CICFlowMeter) to extract the most relevant features from all the generated PCAP files for our study.

After the exploratory step, we combined the initial 76 features (excluding Destination Port and Hash) with 15 additional relevant destination ports, and applied 9 statistical functions using a Jupyter Notebook [script](https://github.com/droidxp/ML/blob/master/features_enge.ipynb) file. To increase the efficiency of our models, we selected the 20 most relevant features based on Gini Importance or Mean Decrease in Impurity, using a Python [script](https://github.com/droidxp/ML/blob/master/clearFile.py). After this procedure, we obtained a [dataset](https://github.com/droidxp/ML/blob/master/cleaned_file.csv), containing the 20 most relevant features from all 4,076 app pairs (5,887 apps).

### Data Analysis

With this [dataset](https://github.com/droidxp/ML/blob/master/clearFile.py), we used various Python scripts to compute the metrics for different machine learning models, such as: Precision, recall, F1-score and AUC (Area Under Curve). Each script generated a respective CSV file containing the model predictions, as presented below:

+ Random Forest - [RF](https://github.com/droidxp/ML/blob/master/DecisionTree.py). [CSV](https://github.com/droidxp/ML/blob/master/RandomForest.csv) file.
+ Logistic Regrresion - [LR](https://github.com/droidxp/ML/blob/master/LogisticRegression.py). [CSV](https://github.com/droidxp/ML/blob/master/LogisticRegression.csv) file.
+ Liner Discriminant Analysis - [LDA](https://github.com/droidxp/ML/blob/master/Lda.py). [CSV](https://github.com/droidxp/ML/blob/master/Lda.csv) file.
+ Quadratic Discriminant Analysis - [QDA](https://github.com/droidxp/ML/blob/master/Qda.py). [CSV](https://github.com/droidxp/ML/blob/master/Qda.csv) file.
+ Multi-layer Perceptron - [MLP](https://github.com/droidxp/ML/blob/master/Mlp.py). [CSV](https://github.com/droidxp/ML/blob/master/mlp.csv) file.
+ Support Vector Machines - [SVM](https://github.com/droidxp/ML/blob/master/Svm.py). [CSV](https://github.com/droidxp/ML/blob/master/svm.csv) file.

The following Python [code](https://github.com/droidxp/ML/blob/master/allAlgorithms.py) prints the comparison of all algorithms.

## Replication Package


This is the replication package for the paper: Bridging the MAS approach Gap: A Network Flow
Analysis Approach for Malware Detection

### Abstract

As mobile technologies become part of modern society, Android’s dominance in the global smartphone market has made it a prime target for cyberattacks. The Android platform
faces a growing threat from malware, particularly repackaged apps that embed malicious code to exploit user data. Conventional sandbox-based detection systems, such as the Mining Android
Sandbox (MAS), rely on analyzing sensitive API interactions but exhibit limitations against complex threats that evade in-device behavioral monitoring. To address this gap, we introduce a complementary approach centered on network flow analysis. Our detection methodology uses a test generator tool to stimulate app activity and TcpDump to collect subsequent network traffic data. Machine learning models are then applied to this data to uncover latent malicious communications. Results demonstrate that our method achieves superior detection accuracy when compared to the MAS approach, validating network traffic as a critical signal for identifying advanced malware and enhancing the overall security posture of Android applications.

### Malware Dataset

At our work, we used a curated dataset of 4,076 app pairs (original/repackaged) from the study by Costa et al., which is available at CSV [file](https://github.com/droidxp/paper-droidxptrace-results/blob/main/TSE/large_ds.csv). Using this dataset, we utilized the [DroidXP](https://github.com/droidxp/benchmark) infrastructure for data collection. With DroidXP, we collected all PCAP files during the test executions, which were performed by the test generator tool, Droidbot, for 180 seconds for each app (a total of 5,887 apps). After this exploratory step, we leveraged [CICflowMeter](https://github.com/ahlashkari/CICFlowMeter) to extract the most relevant features from all the generated PCAP files for our study.

After the exploratory step, we combined the initial 76 features (excluding Destination Port and Hash) with 15 additional relevant destination ports, and applied 9 statistical functions using a Jupyter Notebook [script](https://github.com/droidxp/ML/blob/master/features_enge.ipynb) file. To increase the efficiency of our models, we selected the 20 most relevant features based on Gini Importance or Mean Decrease in Impurity, using a Python [script](https://github.com/droidxp/ML/blob/master/clearFile.py). After this procedure, we obtained a [dataset](https://github.com/droidxp/ML/blob/master/cleaned_file.csv), containing the 20 most relevant features from all 4,076 app pairs (5,887 apps).

### Data Analysis

With this [dataset](https://github.com/droidxp/ML/blob/master/clearFile.py), we used various Python scripts to compute the metrics for different machine learning models, such as: Precision, recall, F1-score and AUC (Area Under Curve). Each script generated a respective CSV file containing the model predictions, as presented below:

+ [Random Forest](https://github.com/droidxp/ML/blob/master/DecisionTree.py). [CSV](https://github.com/droidxp/ML/blob/master/RandomForest.csv) file.
+ [Logistic Regression](https://github.com/droidxp/ML/blob/master/LogisticRegression.py). [CSV](https://github.com/droidxp/ML/blob/master/LogisticRegression.csv) file.
+ Liner Discriminant Analysis - [LDA](https://github.com/droidxp/ML/blob/master/Lda.py). [CSV](https://github.com/droidxp/ML/blob/master/Lda.csv) file.
+ Quadratic Discriminant Analysis - [QDA](https://github.com/droidxp/ML/blob/master/Qda.py). [CSV](https://github.com/droidxp/ML/blob/master/Qda.csv) file.

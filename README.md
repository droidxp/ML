## Replication Package


This is the replication package for the paper: Using Network Flow Data and Machine Learning as Support for Mining Android Sandbox

### Abstract

As mobile technologies become part of modern society, Android’s dominance in the global smartphone market has made it
a prime target for cyberattacks. The Android platform faces a growing threat from malware, particularly repackaged apps that embed
malicious code to exploit user data. While sandbox-based approaches, such as Mining Android Sandbox (MAS), have been effective in
detecting repackaged malware by analyzing sensitive API interactions, they often fall short in identifying more complex threats. This
paper introduces a complementary approach that integrates network flow analysis with MAS approach to improve malware detection.
By analyzing the network traffic generated by Android applications, alongside behavioral data from MAS approach, this method
uncovers malicious activities that may remain hidden within the app’s behavior. Using machine learning on network traffic data
collected via TcpDump, combined with MAS approach insights, our method improves the accuracy of malware detection. Results
demonstrate that this combined approach significantly improves the identification of malware, offering a more comprehensive solution
for securing Android applications against evolving threats.

### Malware Dataset

At our work we used the curated dataset of 4,076 app pairs (original/repackage) from Costa et al. study, available on the following CSV [file](https://github.com/droidxp/paper-droidxptrace-results/blob/main/TSE/large_ds.csv). With this apps, we take advantage of the [DroidXP](https://github.com/droidxp/benchmark) infrastructure for data collection. With DroidXP we collect all PCAP files, during the test execution performed by the test generator tool Droidbot for 180 seconds, for each app pair (original/repackage). After the exploratory step, we take advange of [CICflowMeter](https://github.com/ahlashkari/CICFlowMeter) to extract the most relevant features for our study from all generated PCAP files. The final result from this exploratory step is compiled in this [zip](https://unbbr-my.sharepoint.com/:u:/g/personal/180040723_aluno_unb_br/EQi-3p0Rg1xDtsUt7hwGTHABJaCwOB2DbbWDFjUrG5jZ8A?e=KWXkOu). Because of space issues at the repository, this zip file is stored in an external repository.

When merge all flow feature set, involving the combination of the initial 76 features (excluding Destination Port and Hash), 15 more relevant destination Ports, and 9 statistical functions, we generate a [partial](https://unbbr-my.sharepoint.com/:x:/g/personal/180040723_aluno_unb_br/Eclh5qBUIblAvKj7EPU-RcMB7c4YfXF2ezKXiWh-gij5tw?e=5JBOs3) final file, taking advantage of Jupyter Notebook [script](https://github.com/droidxp/ML/blob/master/features_enge.ipynb) file.

import pandas as pd
from os.path import dirname, join
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

current_dir = dirname(__file__)

file_path = join(current_dir, "large_ds.csv")
TSE = pd.read_csv(file_path)

file_path = join(current_dir, "cleaned_file.csv")
CLEAR = pd.read_csv(file_path)

file_path = join(current_dir, "RandomForest.csv")
FOREST = pd.read_csv(file_path)

FINAL_DS_MAS_FLOW = pd.merge(CLEAR, FOREST[['Index','Pred']], left_on='index', right_on='Index', how='left')
FINAL_DS_MAS_FLOW = FINAL_DS_MAS_FLOW.drop(['Index'], axis=1)
FINAL_DS_MAS_FLOW = pd.merge(FINAL_DS_MAS_FLOW, TSE[['sha256','apidetected']], left_on='hash', right_on='sha256', how='left')
FINAL_DS_MAS_FLOW = FINAL_DS_MAS_FLOW.drop(['sha256'], axis=1)
FINAL_DS_MAS_FLOW = FINAL_DS_MAS_FLOW.dropna()

TP = FINAL_DS_MAS_FLOW.loc[(FINAL_DS_MAS_FLOW['malicious'] == 1.0) & ((FINAL_DS_MAS_FLOW['Pred'] == 1.0) | (FINAL_DS_MAS_FLOW['apidetected'] == True))]
FP = FINAL_DS_MAS_FLOW.loc[(FINAL_DS_MAS_FLOW['malicious'] == 0.0) & ((FINAL_DS_MAS_FLOW['Pred'] == 1.0) | (FINAL_DS_MAS_FLOW['apidetected'] == True))]
FN = FINAL_DS_MAS_FLOW.loc[(FINAL_DS_MAS_FLOW['malicious'] == 1.0) & ((FINAL_DS_MAS_FLOW['Pred'] == 0.0) & (FINAL_DS_MAS_FLOW['apidetected'] == False))]
PRECISION = len(TP)/(len(TP)+len(FP))
RECALL = len(TP)/(len(TP)+len(FN))
F_one =  2*((PRECISION*RECALL)/(PRECISION+RECALL))

print('Total Samples: %0d'%len(FINAL_DS_MAS_FLOW))
print('Total True Positive: %0d'%len(TP))
print('Total False Positive: %0d'%len(FP))
print('Total False Negative: %0d'%len(FN))
print('Precision: %0f'%PRECISION)
print('Recall: %0f'%RECALL)
print('F_one Score: %0f'%F_one)

TP_FLOW = FINAL_DS_MAS_FLOW.loc[(FINAL_DS_MAS_FLOW['malicious'] == 1.0) & ((FINAL_DS_MAS_FLOW['Pred'] == 1.0) & (FINAL_DS_MAS_FLOW['apidetected'] == False))]
TP_MAS = FINAL_DS_MAS_FLOW.loc[(FINAL_DS_MAS_FLOW['malicious'] == 1.0) & ((FINAL_DS_MAS_FLOW['Pred'] == 0.0) & (FINAL_DS_MAS_FLOW['apidetected'] == True))]
TP_MAS_FLOW = FINAL_DS_MAS_FLOW.loc[(FINAL_DS_MAS_FLOW['malicious'] == 1.0) & ((FINAL_DS_MAS_FLOW['Pred'] == 1.0) & (FINAL_DS_MAS_FLOW['apidetected'] == True))]


FINAL_DS_MAS_FLOW.rename(columns={'Pred': 'Flow_Network_Analysis', 'apidetected': 'MAS_Analysis'}, inplace=True)
FINAL_DS_MAS_FLOW = FINAL_DS_MAS_FLOW.drop(['index'], axis=1)
FINAL_DS_MAS_FLOW.to_csv("final_ds_mas_flow.csv", index_label = 'index', index = False)



only_flow = len(set(TP_FLOW['hash']))
only_mas = len(set(TP_MAS['hash']))
only_mas_flow = len(set(TP_MAS_FLOW['hash']))

venn2(subsets = (only_flow, only_mas, only_mas_flow), 
      set_labels = ('Group A',  
                    'Group B'), 
      set_colors=("orange", 
                  "blue"),alpha=0.7) 


plt.show()

print(only_flow)
print(only_mas)
print(only_mas_flow)

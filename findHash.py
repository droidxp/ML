from os.path import dirname, join
import pandas as pd

current_dir = dirname(__file__)
file_path = join(current_dir, "large_ds.csv")
MAS = pd.read_csv(file_path)
print(MAS['sha256'].value_counts())
HASH_MAS = MAS['sha256']

#print(HASH_MAS)

current_dir = dirname(__file__)
file_path = join(current_dir, "final_ds_mas_flow.csv")
FLOW = pd.read_csv(file_path)
print(FLOW['hash'].value_counts())
HASH_FLOW = FLOW['hash']


DIFF = pd.concat([HASH_MAS,HASH_FLOW]).drop_duplicates(keep=False)
print(DIFF)
DIFF.to_csv('diff.csv', index=False)
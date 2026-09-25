import pandas as pd
import os

data_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"
df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=1000)

all_matches = []
for _, row in df_gt.iterrows():
    m = str(row['matched_entity_ids'])
    if m and m != 'nan':
        for x in m.split(','):
            all_matches.append((row['source1_entity_id'], x.strip()))

df_pairs = pd.DataFrame(all_matches, columns=['s1', 'cand'])
print(f"Total true pairs in sample: {len(df_pairs)}")

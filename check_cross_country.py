import pandas as pd
import os

data_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"
df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", nrows=100000, usecols=['entity_id', 'country'])
s1_country = dict(zip(df_s1['entity_id'], df_s1['country']))

df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", nrows=200000, usecols=['entity_id', 'country'])
s2_country = dict(zip(df_s2['entity_id'], df_s2['country']))

cross_country = 0
checked = 0
for chunk in pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", chunksize=100000):
    for _, row in chunk.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id in s1_country:
            c1 = s1_country[s1_id]
            matched = str(row['matched_entity_ids']).split(',')
            for mid in matched:
                if mid in s2_country:
                    c2 = s2_country[mid]
                    checked += 1
                    if c1 != c2:
                        cross_country += 1
    if checked > 1000:
        break

print(f"Checked matches: {checked}, Cross-country matches: {cross_country}")

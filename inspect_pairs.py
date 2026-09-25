import pandas as pd
import os

data_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=5)
s1_targets = set(df_gt['source1_entity_id'])
all_s2_targets = set()
for m in df_gt['matched_entity_ids']:
    for mid in m.split(','):
        if mid.startswith('S2-'):
            all_s2_targets.add(mid)

s1_recs = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(s1_targets)]
    for _, row in m.iterrows():
        s1_recs[row['entity_id']] = row.to_dict()
    if len(s1_recs) == len(s1_targets):
        break

s2_recs = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(all_s2_targets)]
    for _, row in m.iterrows():
        s2_recs[row['entity_id']] = row.to_dict()
    if len(s2_recs) == len(all_s2_targets):
        break

for _, row in df_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1_data = s1_recs.get(s1_id, {})
    print("="*60)
    print(f"S1: {s1_id} | Name: '{s1_data.get('business_name')}' | Addr: '{s1_data.get('business_address')}' | Country: {s1_data.get('country')}")
    matches = row['matched_entity_ids'].split(',')
    for mid in matches:
        if mid in s2_recs:
            s2_data = s2_recs[mid]
            print(f"  --> S2 Match: {mid} | Name: '{s2_data.get('business_name')}' | Addr: '{s2_data.get('business_address')}' | Country: {s2_data.get('country')}")

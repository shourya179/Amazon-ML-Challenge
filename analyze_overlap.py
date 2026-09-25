import pandas as pd
import os
import re

data_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"
df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=2000)

s1_to_matches = {}
all_s23_ids = set()
for _, row in df_gt.iterrows():
    s1_id = row['source1_entity_id']
    m_str = str(row['matched_entity_ids'])
    if m_str and m_str != 'nan':
        m_list = [x.strip() for x in m_str.split(',') if x.strip()]
        if m_list:
            s1_to_matches[s1_id] = m_list
            all_s23_ids.update(m_list)

print(f"Targeting {len(s1_to_matches)} S1 entities with {len(all_s23_ids)} matching S2/S3 IDs")

# Load S1
s1_recs = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(s1_to_matches)]
    for _, row in m.iterrows():
        s1_recs[row['entity_id']] = row.to_dict()
    if len(s1_recs) == len(s1_to_matches):
        break

# Load S2
s2_recs = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(all_s23_ids)]
    for _, row in m.iterrows():
        s2_recs[row['entity_id']] = row.to_dict()
    if len(s2_recs) == len(all_s23_ids):
        break

# Load S3
s3_recs = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(all_s23_ids)]
    for _, row in m.iterrows():
        s3_recs[row['entity_id']] = row.to_dict()
    if len(s2_recs) + len(s3_recs) >= len(all_s23_ids):
        break

all_target_recs = {**s2_recs, **s3_recs}

def clean_toks(s):
    if not isinstance(s, str): return set()
    s = re.sub(r'[^a-zA-Z0-9]', ' ', s.lower())
    return set(s.split())

def clean_nums(s):
    if not isinstance(s, str): return set()
    return set(re.findall(r'\b\d+\b', s))

# Analyze statistics
name_tok_overlap = 0
addr_tok_overlap = 0
num_overlap = 0
either_overlap = 0
total_pairs = 0

for s1_id, matches in s1_to_matches.items():
    if s1_id not in s1_recs: continue
    s1 = s1_recs[s1_id]
    s1_name_toks = clean_toks(s1['business_name'])
    s1_addr_toks = clean_toks(s1['business_address'])
    s1_nums = clean_nums(str(s1['business_address']))

    for mid in matches:
        if mid not in all_target_recs: continue
        tgt = all_target_recs[mid]
        tgt_name_toks = clean_toks(tgt['business_name'])
        tgt_addr_toks = clean_toks(tgt['business_address'])
        tgt_nums = clean_nums(str(tgt['business_address']))

        total_pairs += 1
        has_name_ov = len(s1_name_toks & tgt_name_toks) > 0
        has_addr_ov = len(s1_addr_toks & tgt_addr_toks) > 0
        has_num_ov = len(s1_nums & tgt_nums) > 0

        if has_name_ov: name_tok_overlap += 1
        if has_addr_ov: addr_tok_overlap += 1
        if has_num_ov: num_overlap += 1
        if has_name_ov or has_addr_ov or has_num_ov: either_overlap += 1

print(f"Total True Pairs Analyzed: {total_pairs}")
print(f"Pairs with Name Token Overlap: {name_tok_overlap} ({name_tok_overlap/total_pairs*100:.2f}%)")
print(f"Pairs with Address Token Overlap: {addr_tok_overlap} ({addr_tok_overlap/total_pairs*100:.2f}%)")
print(f"Pairs with Address Number Overlap: {num_overlap} ({num_overlap/total_pairs*100:.2f}%)")
print(f"Pairs with EITHER Name, Address, or Number Overlap: {either_overlap} ({either_overlap/total_pairs*100:.2f}%)")
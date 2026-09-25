import pandas as pd
import os
import re
import time
from collections import defaultdict

data_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("Loading 10,000 ground truth samples...")
df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=10000)
s1_set = set(df_gt['source1_entity_id'])

# Ground truth mapping
gt_map = {}
for _, row in df_gt.iterrows():
    m_str = str(row['matched_entity_ids'])
    if m_str and m_str != 'nan':
        gt_map[row['source1_entity_id']] = set(m_str.split(','))
    else:
        gt_map[row['source1_entity_id']] = set()

# Load S1 records
s1_records = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(s1_set)]
    for _, row in m.iterrows():
        s1_records[row['entity_id']] = row.to_dict()
    if len(s1_records) == len(s1_set):
        break

# Target IDs in ground truth
all_true_targets = set()
for s in gt_map.values():
    all_true_targets.update(s)

print(f"Loaded {len(s1_records)} S1 entities with {len(all_true_targets)} true matches.")

# Load S2 and S3 targets + some distractors (e.g. 50,000 rows each)
s2_records = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", chunksize=100000):
    for _, row in chunk.iterrows():
        if row['entity_id'] in all_true_targets or len(s2_records) < 50000:
            s2_records[row['entity_id']] = row.to_dict()
    if len(s2_records) >= 50000 and all_true_targets.issubset(set(s2_records.keys()) | set(getattr(dict, 'dummy', {}))):
        break

s3_records = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", chunksize=100000):
    for _, row in chunk.iterrows():
        if row['entity_id'] in all_true_targets or len(s3_records) < 50000:
            s3_records[row['entity_id']] = row.to_dict()
    if len(s3_records) >= 50000:
        break

all_target_records = {**s2_records, **s3_records}
print(f"Total target candidates loaded (including distractors): {len(all_target_records)}")

LEGAL = {'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}

def get_keys(rec):
    keys = []
    name = str(rec.get('business_name', '')).lower()
    addr = str(rec.get('business_address', '')).lower()
    country = str(rec.get('country', '')).strip().lower()

    # Clean name tokens
    name_clean = re.sub(r'[^a-zA-Z0-9]', ' ', name)
    tokens = [t for t in name_clean.split() if len(t) >= 3 and t not in LEGAL]

    # Clean digits
    digits = re.findall(r'\b\d+\b', addr)

    # 1. Primary brand token + country
    if tokens:
        keys.append(f"{country}:name:{tokens[0]}")
        if len(tokens) >= 2:
            keys.append(f"{country}:name2:{tokens[0]}_{tokens[1]}")

    # 2. Number + first word of address
    addr_clean = re.sub(r'[^a-zA-Z0-9]', ' ', addr)
    addr_tokens = [t for t in addr_clean.split() if len(t) >= 3]
    if digits:
        for d in digits[:2]:
            keys.append(f"{country}:num:{d}")
            if addr_tokens:
                keys.append(f"{country}:num_addr:{d}_{addr_tokens[0]}")

    return keys

# Build inverted index
t0 = time.time()
inv_index = defaultdict(list)
for cid, rec in all_target_records.items():
    for k in get_keys(rec):
        inv_index[k].append(cid)
t_index = time.time() - t0
print(f"Index built in {t_index:.2f}s with {len(inv_index)} keys.")

# Query candidates for each S1 entity
t0 = time.time()
found_matches = 0
total_matches = 0
total_candidates = 0

for s1_id, rec in s1_records.items():
    true_set = gt_map.get(s1_id, set())
    total_matches += len(true_set)
    keys = get_keys(rec)

    # Aggregate candidate votes
    cand_counts = defaultdict(int)
    for k in keys:
        if k in inv_index:
            for cid in inv_index[k]:
                cand_counts[cid] += 1

    # Keep top 30 candidates by vote count
    top_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    cand_ids = {c[0] for c in top_cands}
    total_candidates += len(cand_ids)

    found_matches += len(true_set & cand_ids)

t_query = time.time() - t0
print(f"Queried {len(s1_records)} entities in {t_query:.2f}s ({len(s1_records)/t_query:.0f} entities/sec).")
print(f"Candidate Recall: {found_matches}/{total_matches} ({found_matches/total_matches*100:.2f}%)")
print(f"Average candidates per S1 entity: {total_candidates / len(s1_records):.1f}")

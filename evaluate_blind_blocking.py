import os
import re
import time
from collections import defaultdict

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("--> [1/4] Loading 5,000 Validation S1 entities and their Ground Truth...")
val_gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            val_gt[p[0]] = set(p[1].split(','))
        else:
            val_gt[p[0]] = set()
        if len(val_gt) >= 5000:
            break

val_s1_ids = set(val_gt.keys())
total_true_s2 = sum(len([x for x in s if x.startswith('S2-')]) for s in val_gt.values())
total_true_s3 = sum(len([x for x in s if x.startswith('S3-')]) for s in val_gt.values())
print(f"Validation S1 entities: {len(val_s1_ids)}")
print(f"True S2 matches to find: {total_true_s2}, True S3 matches to find: {total_true_s3}")

print("--> [2/4] Loading S1 Records...")
val_s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4 and p[0] in val_s1_ids:
            val_s1_records[p[0]] = {
                'id': p[0],
                'name': p[1],
                'addr': p[2],
                'country': p[3].strip().lower()
            }
        if len(val_s1_records) == len(val_s1_ids):
            break

LEGAL = {'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}

def extract_blocking_keys(name, addr, country):
    keys = []
    # 1. Clean name tokens
    name_clean = re.sub(r'[^a-zA-Z0-9]', ' ', name.lower())
    toks = [t for t in name_clean.split() if len(t) >= 2 and t not in LEGAL]
    if toks:
        keys.append(f"{country}:b1:{toks[0]}")
        if len(toks) >= 2:
            keys.append(f"{country}:b2:{toks[0]}_{toks[1]}")

    # 2. Address tokens and numbers
    addr_clean = re.sub(r'[^a-zA-Z0-9]', ' ', addr.lower())
    addr_toks = [t for t in addr_clean.split() if len(t) >= 3]
    nums = re.findall(r'\b\d+\b', addr)
    if nums:
        for n in nums[:2]:
            keys.append(f"{country}:n:{n}")
            if addr_toks:
                keys.append(f"{country}:na:{n}_{addr_toks[0]}")
                if len(addr_toks) >= 2:
                    keys.append(f"{country}:na2:{n}_{addr_toks[1]}")

    return keys

def build_index(filepath, max_rows=1500000):
    idx = defaultdict(list)
    t0 = time.time()
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            count += 1
            p = line.strip().split('\t')
            if len(p) >= 4:
                eid = p[0]
                int_id = int(eid[3:])
                c = p[3].strip().lower()
                for k in extract_blocking_keys(p[1], p[2], c):
                    idx[k].append(int_id)
            if count >= max_rows:
                break
    print(f"Indexed {count} rows in {time.time() - t0:.2f}s. Unique keys: {len(idx)}")
    return idx

print("--> [3/4] Indexing Source 2...")
idx_s2 = build_index(os.path.join(train_dir, "train_source2.tsv"), max_rows=2000000)

print("--> [4/4] Indexing Source 3...")
idx_s3 = build_index(os.path.join(train_dir, "train_source3.tsv"), max_rows=2000000)

def retrieve_top_k(keys, idx, prefix, k=25):
    counts = defaultdict(int)
    for key in keys:
        if key in idx:
            for int_id in idx[key]:
                counts[int_id] += 1
    top_ids = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
    return [f"{prefix}-{x[0]}" for x in top_ids]

# Query validation entities BLINDLY
t0 = time.time()
found_s2 = 0
found_s3 = 0
total_candidates = 0

for s1_id, rec in val_s1_records.items():
    true_matches = val_gt[s1_id]
    keys = extract_blocking_keys(rec['name'], rec['addr'], rec['country'])

    cands_s2 = retrieve_top_k(keys, idx_s2, "S2", k=25)
    cands_s3 = retrieve_top_k(keys, idx_s3, "S3", k=25)

    cand_set = set(cands_s2) | set(cands_s3)
    total_candidates += len(cand_set)

    found_s2 += len(true_matches & set(cands_s2))
    found_s3 += len(true_matches & set(cands_s3))

query_time = time.time() - t0
print("\n=======================================================")
print(f"BLIND CANDIDATE BLOCKING EVALUATION RESULTS:")
print(f"Queried {len(val_s1_records)} entities in {query_time:.2f}s ({len(val_s1_records)/query_time:.0f} entities/s)")
print(f"S2 Recall: {found_s2}/{total_true_s2} ({found_s2/total_true_s2*100:.2f}%)")
print(f"S3 Recall: {found_s3}/{total_true_s3} ({found_s3/total_true_s3*100:.2f}%)")
print(f"Overall Recall: {(found_s2 + found_s3)/(total_true_s2 + total_true_s3)*100:.2f}%")
print(f"Average candidate pool per S1 entity: {total_candidates / len(val_s1_records):.1f}")
print("=======================================================")

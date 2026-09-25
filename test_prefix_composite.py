import os
import re
import time
import math
from collections import defaultdict

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

# Load 1,000 US S1 entities and their S2 targets
val_gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            s2 = [x for x in p[1].split(',') if x.startswith('S2-')]
            if s2:
                val_gt[p[0]] = set(s2)
        if len(val_gt) >= 1000:
            break

val_s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in val_gt and p[3].strip().lower() == 'us':
            val_s1_records[p[0]] = p

total_true = sum(len(val_gt[s]) for s in val_s1_records)
print(f"Testing {len(val_s1_records)} US S1 entities with {total_true} true S2 targets.")

LEGAL = {'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}

def clean_toks(s):
    return [t for t in re.sub(r'[^a-zA-Z0-9]', ' ', s.lower()).split() if len(t) >= 2]

def get_nums(s):
    res = []
    for n in re.findall(r'\b\d+\b', s):
        try:
            norm = str(int(n))
            if 1 <= len(norm) <= 7:
                res.append(norm)
        except ValueError:
            pass
    return res

def get_composite_keys(name, addr):
    name_words = [t for t in clean_toks(name) if t not in LEGAL]
    addr_words = [t for t in clean_toks(addr) if t not in {'street', 'st', 'road', 'rd', 'ave', 'avenue', 'suite', 'unit', 'dr', 'drive'}]
    nums = get_nums(addr)

    keys = []
    # 1. High-precision composite: name_word + num
    for nw in name_words[:5]:
        for nm in nums[:3]:
            keys.append(f"WN:{nw}_{nm}")
            if len(nw) >= 4:
                keys.append(f"P3N:{nw[:3]}_{nm}")

    # 2. High-precision composite: name_word1 + name_word2
    if len(name_words) >= 2:
        keys.append(f"WW:{name_words[0]}_{name_words[1]}")
        if len(name_words) >= 3:
            keys.append(f"WW:{name_words[0]}_{name_words[2]}")

    # 3. Medium-precision composite: addr_word + num
    for aw in addr_words[:3]:
        for nm in nums[:3]:
            keys.append(f"AN:{aw}_{nm}")

    # 4. Standalone name words
    for nw in name_words[:4]:
        keys.append(f"W:{nw}")

    return keys

print("Indexing ALL ~3 Million US S2 rows...")
t0 = time.time()
idx = defaultdict(list)
us_count = 0
with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4 and p[3].strip().lower() == 'us':
            us_count += 1
            int_id = int(p[0][3:])
            for k in get_composite_keys(p[1], p[2]):
                idx[k].append(int_id)

print(f"Indexed {us_count} rows in {time.time() - t0:.2f}s. Unique keys: {len(idx)}")

N = float(us_count)
key_weights = {}
for k, id_list in idx.items():
    df = len(id_list)
    if df > 5000:
        continue
    idf = math.log((N + 1.0) / (df + 1.0))
    if k.startswith("WN:") or k.startswith("WW:"):
        key_weights[k] = idf * 2.5
    elif k.startswith("P3N:") or k.startswith("AN:"):
        key_weights[k] = idf * 1.5
    else:
        key_weights[k] = idf

# Query S1 entities
print("Querying validation entities...")
t0 = time.time()
found = 0
total_cands = 0

for s1_id, p in val_s1_records.items():
    true_set = val_gt[s1_id]
    keys = get_composite_keys(p[1], p[2])

    cand_scores = defaultdict(float)
    for k in keys:
        if k in key_weights:
            w = key_weights[k]
            for int_id in idx[k]:
                cand_scores[int_id] += w

    top_cands = {f"S2-{x[0]}" for x in sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:35]}
    total_cands += len(top_cands)
    found += len(true_set & top_cands)

q_time = time.time() - t0
print("\n=======================================================")
print(f"PREFIX-ENHANCED COMPOSITE US S2 BLOCKING EVALUATION:")
print(f"Queried {len(val_s1_records)} entities in {q_time:.2f}s")
print(f"Blind Candidate Recall: {found}/{total_true} ({found/total_true*100:.2f}%)")
print(f"Average candidates per S1 entity: {total_cands / len(val_s1_records):.1f}")
print("=======================================================")

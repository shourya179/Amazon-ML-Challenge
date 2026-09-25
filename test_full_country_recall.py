import os
import re
import time
from collections import defaultdict

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("--> [1/4] Loading 3,000 US Validation S1 entities and their Ground Truth...")
val_gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            # keep only S2 targets
            s2_targets = {x for x in p[1].split(',') if x.startswith('S2-')}
            if s2_targets:
                val_gt[p[0]] = s2_targets
        if len(val_gt) >= 3000:
            break

val_s1_ids = set(val_gt.keys())
total_true_s2 = sum(len(s) for s in val_gt.values())
print(f"Validation S1 entities: {len(val_s1_ids)}, True S2 targets to find: {total_true_s2}")

print("--> [2/4] Loading S1 Records...")
val_s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in val_s1_ids:
            val_s1_records[p[0]] = {'id': p[0], 'name': p[1], 'addr': p[2], 'country': p[3].strip().lower()}
        if len(val_s1_records) == len(val_s1_ids):
            break

LEGAL = {'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}

def get_clean_nums(text):
    nums = re.findall(r'\b\d+\b', text)
    res = set()
    for n in nums:
        try:
            # normalize by stripping leading zeroes
            norm = str(int(n))
            if len(norm) <= 8: # skip huge random strings
                res.add(norm)
        except ValueError:
            pass
    return res

def extract_blocking_keys(name, addr):
    keys = []
    # 1. Clean name tokens
    name_clean = re.sub(r'[^a-zA-Z0-9]', ' ', name.lower())
    toks = [t for t in name_clean.split() if len(t) >= 2 and t not in LEGAL]
    if toks:
        keys.append(f"w0:{toks[0]}")
        if len(toks) >= 2:
            keys.append(f"w1:{toks[1]}")
            keys.append(f"w01:{toks[0]}_{toks[1]}")

    # 2. Normalized numbers from address
    nums = get_clean_nums(addr)
    addr_clean = re.sub(r'[^a-zA-Z0-9]', ' ', addr.lower())
    addr_toks = [t for t in addr_clean.split() if len(t) >= 3]

    for n in nums:
        keys.append(f"num:{n}")
        if addr_toks:
            keys.append(f"na:{n}_{addr_toks[0]}")

    # 3. First significant address word
    if addr_toks:
        keys.append(f"addr0:{addr_toks[0]}")

    return keys

print("--> [3/4] Indexing ALL ~3 Million US Records of Source 2...")
t0 = time.time()
idx_s2 = defaultdict(list)
us_s2_count = 0

with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4 and p[3].strip().lower() == 'us':
            us_s2_count += 1
            int_id = int(p[0][3:])
            for k in extract_blocking_keys(p[1], p[2]):
                idx_s2[k].append(int_id)

print(f"Indexed ALL {us_s2_count} US S2 rows in {time.time() - t0:.2f}s! Unique keys: {len(idx_s2)}")

# Query validation S1 entities
print("--> [4/4] Querying validation entities...")
t0 = time.time()
found_s2 = 0
total_cands = 0

for s1_id, rec in val_s1_records.items():
    true_targets = val_gt[s1_id]
    keys = extract_blocking_keys(rec['name'], rec['addr'])

    counts = defaultdict(int)
    for k in keys:
        if k in idx_s2:
            for int_id in idx_s2[k]:
                counts[int_id] += 1

    # Take top 30 candidates by key overlap count
    top_cands = [f"S2-{x[0]}" for x in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:30]]
    total_cands += len(top_cands)
    found_s2 += len(true_targets & set(top_cands))

q_time = time.time() - t0
print("\n=======================================================")
print(f"FULL POPULATION US S2 BLOCKING EVALUATION:")
print(f"Queried {len(val_s1_records)} entities against ALL {us_s2_count} targets in {q_time:.2f}s")
print(f"Blind Candidate Recall: {found_s2}/{total_true_s2} ({found_s2/total_true_s2*100:.2f}%)")
print(f"Average candidates per S1 entity: {total_cands / len(val_s1_records):.1f}")
print("=======================================================")

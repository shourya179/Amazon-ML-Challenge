import os
import re
import sys
import math
import time
from collections import defaultdict

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("="*70)
print("STEP 2: STANDARDIZED BENCHMARK OF 3 BLOCKER STRATEGIES")
print("="*70)
sys.stdout.flush()

# 1. Load exact same 2,000 US Validation S1 entities and their S2 ground truth
val_gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            s2_targets = {x for x in p[1].split(',') if x.startswith('S2-')}
            if s2_targets:
                val_gt[p[0]] = s2_targets
        if len(val_gt) >= 2000:
            break

val_s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in val_gt and p[3].strip().lower() == 'us':
            val_s1_records[p[0]] = {
                'id': p[0], 'name': p[1], 'addr': p[2], 'country': 'us'
            }

# Filter val_gt to only US entities present in records
val_gt = {sid: val_gt[sid] for sid in val_s1_records}
total_val_entities = len(val_s1_records)
total_true_targets = sum(len(s) for s in val_gt.values())

print(f"Validation Set: {total_val_entities} US S1 entities")
print(f"True S2 targets to retrieve: {total_true_targets}")
sys.stdout.flush()

LEGAL = {'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}

def clean_toks(s):
    return [t for t in re.sub(r'[^a-zA-Z0-9]', ' ', s.lower()).split() if len(t) >= 2]

def get_clean_nums(text):
    res = []
    for n in re.findall(r'\b\d+\b', text):
        try:
            norm = str(int(n))
            if 1 <= len(norm) <= 7:
                res.append(norm)
        except ValueError:
            pass
    return res

# --- Definition of Key Extractors for the 3 Blockers ---

# Blocker 1: Old Blocker
def keys_blocker_1(name, addr):
    toks = [t for t in clean_toks(name) if t not in LEGAL]
    nums = get_clean_nums(addr)
    keys = []
    if toks:
        keys.append(f"w0:{toks[0]}")
    if nums:
        keys.append(f"num:{nums[0]}")
    return keys

# Blocker 2: IDF Blocker
def keys_blocker_2(name, addr):
    toks = [t for t in clean_toks(name) if t not in LEGAL]
    nums = get_clean_nums(addr)
    keys = []
    for t in toks[:4]:
        keys.append(f"w:{t}")
        if len(t) >= 4:
            keys.append(f"p3:{t[:4]}")
    if len(toks) >= 2:
        keys.append(f"w2:{toks[0]}_{toks[1]}")
    for n in nums[:2]:
        keys.append(f"num:{n}")
    return keys

# Blocker 3: Prefix-Enhanced Composite Blocker
def keys_blocker_3(name, addr):
    name_words = [t for t in clean_toks(name) if t not in LEGAL]
    addr_words = [t for t in clean_toks(addr) if t not in {'street', 'st', 'road', 'rd', 'ave', 'avenue', 'suite', 'unit', 'dr', 'drive'}]
    nums = get_clean_nums(addr)
    keys = []
    for nw in name_words[:5]:
        for nm in nums[:3]:
            keys.append(f"WN:{nw}_{nm}")
            if len(nw) >= 4:
                keys.append(f"P3N:{nw[:3]}_{nm}")
    if len(name_words) >= 2:
        keys.append(f"WW:{name_words[0]}_{name_words[1]}")
        if len(name_words) >= 3:
            keys.append(f"WW:{name_words[0]}_{name_words[2]}")
    for aw in addr_words[:3]:
        for nm in nums[:3]:
            keys.append(f"AN:{aw}_{nm}")
    for nw in name_words[:4]:
        keys.append(f"W:{nw}")
    return keys

# --- Loading and Streaming Target US S2 Data ---
print("\nLoading raw lines of US S2 records for standardized benchmarking...")
sys.stdout.flush()
target_rows = []
t0 = time.time()
with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4 and p[3].strip().lower() == 'us':
            target_rows.append((int(p[0][3:]), p[1], p[2]))

print(f"Loaded ALL {len(target_rows):,} US S2 records in {time.time() - t0:.2f}s.")
sys.stdout.flush()

# --- Benchmark Runner ---
def benchmark_blocker(name, key_fn, use_idf=False, use_composite_weights=False, top_k=35):
    print(f"\nEvaluating: [{name}]...")
    sys.stdout.flush()

    # 1. Build Index
    t_idx_start = time.time()
    idx = defaultdict(list)
    for int_id, b_name, b_addr in target_rows:
        for k in key_fn(b_name, b_addr):
            idx[k].append(int_id)
    t_idx = time.time() - t_idx_start
    print(f"  Indexed in {t_idx:.2f}s. Unique keys: {len(idx):,}")
    sys.stdout.flush()

    # 2. Weights
    N = float(len(target_rows))
    weights = {}
    if use_idf or use_composite_weights:
        for k, id_list in idx.items():
            df = len(id_list)
            if df > 5000: continue
            idf = math.log((N + 1.0) / (df + 1.0))
            if use_composite_weights:
                if k.startswith("WN:") or k.startswith("WW:"):
                    weights[k] = idf * 2.5
                elif k.startswith("P3N:") or k.startswith("AN:"):
                    weights[k] = idf * 1.5
                else:
                    weights[k] = idf
            else:
                weights[k] = idf

    # 3. Query Validation Entities
    t_query_start = time.time()
    found = 0
    total_candidates = 0

    for s1_id, rec in val_s1_records.items():
        true_set = val_gt[s1_id]
        keys = key_fn(rec['name'], rec['addr'])

        cand_scores = defaultdict(float)
        for k in keys:
            if use_idf or use_composite_weights:
                if k in weights:
                    w = weights[k]
                    for int_id in idx[k]:
                        cand_scores[int_id] += w
            else:
                if k in idx:
                    for int_id in idx[k]:
                        cand_scores[int_id] += 1.0

        top_cands = {f"S2-{x[0]}" for x in sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]}
        total_candidates += len(top_cands)
        found += len(true_set & top_cands)

    t_query = time.time() - t_query_start
    recall_pct = (found / total_true_targets) * 100.0
    avg_cands = total_candidates / total_val_entities

    print(f"  Results for [{name}]:")
    print(f"    Blind Recall: {found}/{total_true_targets} ({recall_pct:.2f}%)")
    print(f"    Avg Candidates per S1: {avg_cands:.1f}")
    print(f"    Index Time: {t_idx:.2f}s | Query Time: {t_query:.2f}s (Speed: {total_val_entities/t_query:.0f} entities/s)")
    sys.stdout.flush()

    return {
        'name': name,
        'recall': recall_pct,
        'found': found,
        'total': total_true_targets,
        'avg_cands': avg_cands,
        'idx_time': t_idx,
        'query_time': t_query
    }

# Run all 3 blockers
res1 = benchmark_blocker("1. Old Blocker (Word0 + Num, raw counts)", keys_blocker_1, use_idf=False, use_composite_weights=False, top_k=35)
res2 = benchmark_blocker("2. IDF Blocker (Words + Nums + Prefix, IDF ranked)", keys_blocker_2, use_idf=True, use_composite_weights=False, top_k=35)
res3 = benchmark_blocker("3. Prefix-Composite Blocker (WN + WW + P3N + AN + W)", keys_blocker_3, use_idf=False, use_composite_weights=True, top_k=35)

print("\n" + "="*80)
print("FINAL STEP 2 COMPARISON TABLE (Evaluated on ALL 3,016,817 US S2 Records):")
print("="*80)
print(f"{'Blocker Strategy':<42} | {'Blind Recall':<14} | {'Avg Cands':<10} | {'Query Speed'}")
print("-"*80)
for r in [res1, res2, res3]:
    print(f"{r['name']:<42} | {r['recall']:>6.2f}% ({r['found']}/{r['total']}) | {r['avg_cands']:>9.1f} | {total_val_entities/r['query_time']:>6.0f} ent/s")
print("="*80)

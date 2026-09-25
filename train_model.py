#!/usr/bin/env python3
"""
Accelerated Model Training & Calibration Script for Business Entity Resolution
Uses RapidFuzz and LightGBM for lightning-fast training and threshold calibration.
"""

import os
import sys
import gc
import math
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz.distance.Levenshtein import normalized_similarity as levenshtein_ratio
from rapidfuzz.distance.JaroWinkler import similarity as jaro_winkler_similarity

WORKSPACE_DIR = r"c:\Users\Shourya\Desktop\Amazon ML Challenge"
CLEAN_DIR = os.path.join(WORKSPACE_DIR, "cleaned_data", "train")
GT_PATH = os.path.join(WORKSPACE_DIR, "student_resource", "dataset", "train", "train_ground_truth.tsv")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LGB_PATH = os.path.join(OUTPUT_DIR, "model_lgb.txt")
MODEL_PKL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
SRC_LGB_PATH = os.path.join(WORKSPACE_DIR, "code", "business_entity_resolution", "src", "model_lgb.txt")
SRC_PKL_PATH = os.path.join(WORKSPACE_DIR, "code", "business_entity_resolution", "src", "model.pkl")

print("=" * 70)
print("ACCELERATED LEAK-FREE MODEL TRAINING & CALIBRATION")
print("=" * 70)
sys.stdout.flush()

# 1. Load Ground Truth
NUM_TRAIN_S1 = 15000
print(f"--> [1/5] Loading Ground Truth for {NUM_TRAIN_S1:,} S1 entities...")
gt_map = {}
with open(GT_PATH, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            gt_map[p[0]] = set(p[1].split(','))
        else:
            gt_map[p[0]] = set()
        if len(gt_map) >= NUM_TRAIN_S1:
            break

all_needed_s1 = set(gt_map.keys())
print(f"Loaded {len(gt_map):,} ground-truth entities.")
sys.stdout.flush()

# 2. Load Cleaned S1 Records
print("--> [2/5] Loading Cleaned S1 Records...")
s1_dict = {}
with open(os.path.join(CLEAN_DIR, "clean_train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in all_needed_s1:
            clean_n = p[1] if len(p) > 1 else ""
            core_n = p[2] if len(p) > 2 else clean_n
            clean_a = p[3] if len(p) > 3 else ""
            nums = p[4].split() if len(p) > 4 and p[4].strip() else []
            country = p[5].strip().lower() if len(p) > 5 else ""
            s1_dict[p[0]] = {
                'id': p[0],
                'clean_name': clean_n,
                'core_name': core_n,
                'clean_addr': clean_a,
                'nums': nums,
                'country': country
            }
            if len(s1_dict) == len(all_needed_s1):
                break

print(f"Loaded {len(s1_dict):,} S1 entity records.")
sys.stdout.flush()

def get_char_ngrams(s: str, n: int = 3) -> Set[str]:
    s = f"^{s}$"
    if len(s) < n: return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def get_composite_keys(core_name: str, clean_addr: str, nums: List[str]) -> List[str]:
    name_words = core_name.split()
    addr_words = [t for t in clean_addr.split() if t not in {'street', 'st', 'road', 'rd', 'ave', 'avenue', 'suite', 'unit', 'dr', 'drive'}]

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

def extract_pair_features(s1: dict, tgt: dict, blocking_score: float = 0.0) -> Dict[str, float]:
    s1_name, c_name = s1['clean_name'], tgt['clean_name']
    s1_core, c_core = s1['core_name'], tgt['core_name']
    s1_addr, c_addr = s1['clean_addr'], tgt['clean_addr']

    s1_n_toks, c_n_toks = set(s1_name.split()), set(c_name.split())
    s1_c_toks, c_c_toks = set(s1_core.split()), set(c_core.split())
    s1_a_toks, c_a_toks = set(s1_addr.split()), set(c_addr.split())

    s1_nums, c_nums = set(s1['nums']), set(tgt['nums'])

    n_inter = len(s1_n_toks & c_n_toks)
    n_union = len(s1_n_toks | c_n_toks)
    name_jaccard = n_inter / n_union if n_union > 0 else 0.0

    c_inter = len(s1_c_toks & c_c_toks)
    c_union = len(s1_c_toks | c_c_toks)
    core_jaccard = c_inter / c_union if c_union > 0 else 0.0

    min_n = min(len(s1_n_toks), len(c_n_toks))
    name_containment = n_inter / min_n if min_n > 0 else 0.0

    a_inter = len(s1_a_toks & c_a_toks)
    a_union = len(s1_a_toks | c_a_toks)
    addr_jaccard = a_inter / a_union if a_union > 0 else 0.0
    min_a = min(len(s1_a_toks), len(c_a_toks))
    addr_containment = a_inter / min_a if min_a > 0 else 0.0

    num_inter = len(s1_nums & c_nums)
    num_union = len(s1_nums | c_nums)
    digit_jaccard = num_inter / num_union if num_union > 0 else 0.0

    s1_w0 = s1_core.split()[0] if s1_core else ""
    c_w0 = c_core.split()[0] if c_core else ""
    first_word_match = 1.0 if (s1_w0 and s1_w0 == c_w0) else 0.0
    first_char_match = 1.0 if (s1_name and c_name and s1_name[0] == c_name[0]) else 0.0

    s1_ng = get_char_ngrams(s1_name, 3)
    c_ng = get_char_ngrams(c_name, 3)
    name_char3_jaccard = len(s1_ng & c_ng) / max(1, len(s1_ng | c_ng))

    s1_ang = get_char_ngrams(s1_addr, 3)
    c_ang = get_char_ngrams(c_addr, 3)
    addr_char3_jaccard = len(s1_ang & c_ang) / max(1, len(s1_ang | c_ang))

    return {
        'feat_name_exact': 1.0 if s1_name == c_name else 0.0,
        'feat_core_exact': 1.0 if s1_core == c_core else 0.0,
        'feat_name_lev': float(levenshtein_ratio(s1_name, c_name)),
        'feat_core_lev': float(levenshtein_ratio(s1_core, c_core)),
        'feat_name_jw': float(jaro_winkler_similarity(s1_name, c_name)),
        'feat_core_jw': float(jaro_winkler_similarity(s1_core, c_core)),
        'feat_name_token_jaccard': name_jaccard,
        'feat_core_token_jaccard': core_jaccard,
        'feat_name_containment': name_containment,
        'feat_name_char3_jaccard': name_char3_jaccard,
        'feat_first_word_match': first_word_match,
        'feat_first_char_match': first_char_match,
        'len_name_diff': abs(len(s1_name) - len(c_name)) / max(1, max(len(s1_name), len(c_name))),
        'feat_addr_exact': 1.0 if (s1_addr and s1_addr == c_addr) else 0.0,
        'feat_addr_lev': float(levenshtein_ratio(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0,
        'feat_addr_jw': float(jaro_winkler_similarity(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0,
        'feat_addr_token_jaccard': addr_jaccard,
        'feat_addr_containment': addr_containment,
        'feat_addr_char3_jaccard': addr_char3_jaccard,
        'digit_overlap': float(num_inter),
        'digit_jaccard': digit_jaccard,
        'has_common_digit': 1.0 if num_inter > 0 else 0.0,
        'blocking_score': float(blocking_score),
        'is_source2': 1.0 if tgt['id'].startswith('S2-') else 0.0
    }

# 3. Candidate Generation
print("--> [3/5] Generating Leak-Free Candidates from Cleaned S2 & S3...")
sys.stdout.flush()

all_s1_candidates = defaultdict(list)
all_needed_target_ids = set()

def process_target_source(filename: str, prefix: str, max_rows: int = 1500000):
    t0 = time.time()
    idx = defaultdict(list)
    path = os.path.join(CLEAN_DIR, filename)
    print(f"  Streaming & indexing {filename} (up to {max_rows:,} rows)...")
    sys.stdout.flush()

    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 6:
                eid = p[0]
                int_id = int(eid[3:])
                core_n = p[2]
                clean_a = p[3]
                nums = p[4].split() if p[4].strip() else []
                country = p[5].strip().lower()

                for k in get_composite_keys(core_n, clean_a, nums):
                    idx[(country, k)].append(int_id)
                cnt += 1
                if cnt >= max_rows:
                    break

    N = float(cnt)
    print(f"  Indexed {cnt:,} records in {time.time() - t0:.2f}s. Calculating IDFs...")
    sys.stdout.flush()

    weights = {}
    for k, id_list in idx.items():
        df = len(id_list)
        if df > 5000: continue
        idf = math.log((N + 1.0) / (df + 1.0))
        weights[k] = idf * 2.5 if (k[1].startswith("WN:") or k[1].startswith("WW:")) else idf

    print(f"  Scoring candidates for {len(s1_dict):,} S1 entities against {prefix}...")
    sys.stdout.flush()

    for s1_id, s1 in s1_dict.items():
        c = s1['country']
        keys = get_composite_keys(s1['core_name'], s1['clean_addr'], s1['nums'])
        scores = defaultdict(float)
        for k_str in keys:
            lk = (c, k_str)
            if lk in weights:
                w = weights[lk]
                for int_id in idx[lk]:
                    scores[int_id] += w
        top_cands = [(f"{prefix}-{x[0]}", x[1]) for x in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:35]]
        all_s1_candidates[s1_id].extend(top_cands)
        for cid, _ in top_cands:
            all_needed_target_ids.add(cid)

    del idx, weights
    gc.collect()
    print(f"  Finished {prefix}! Memory freed.")
    sys.stdout.flush()

process_target_source("clean_train_source2.tsv", "S2", max_rows=1500000)
process_target_source("clean_train_source3.tsv", "S3", max_rows=1500000)

print(f"\nTotal candidate pairs generated: {sum(len(c) for c in all_s1_candidates.values()):,}")
print(f"Unique target records needed: {len(all_needed_target_ids):,}")
sys.stdout.flush()

# 4. Load Target Cache
print("\n--> [4/5] Loading Target Cache for Feature Extraction...")
sys.stdout.flush()
target_cache = {}
for filename in ["clean_train_source2.tsv", "clean_train_source3.tsv"]:
    path = os.path.join(CLEAN_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 6 and p[0] in all_needed_target_ids:
                target_cache[p[0]] = {
                    'id': p[0],
                    'clean_name': p[1],
                    'core_name': p[2],
                    'clean_addr': p[3],
                    'nums': p[4].split() if p[4].strip() else [],
                    'country': p[5].strip().lower()
                }
                if len(target_cache) == len(all_needed_target_ids):
                    break
    print(f"  Loaded {len(target_cache):,} / {len(all_needed_target_ids):,} target records.")
    sys.stdout.flush()

# 5. Extract Features
print("\n--> [5/5] Extracting Features via RapidFuzz SIMD...")
t0_feat = time.time()
sys.stdout.flush()

X_all, y_all = [], []
pair_meta = [] # (s1_id, cid)

for s1_id, cand_list in all_s1_candidates.items():
    s1 = s1_dict[s1_id]
    true_set = gt_map[s1_id]
    for cid, b_score in cand_list:
        if cid in target_cache:
            feats = extract_pair_features(s1, target_cache[cid], b_score)
            lbl = 1 if cid in true_set else 0
            X_all.append(feats)
            y_all.append(lbl)
            pair_meta.append((s1_id, cid))

print(f"Feature extraction completed in {time.time() - t0_feat:.2f}s!")
df_X = pd.DataFrame(X_all)
feature_cols = list(df_X.columns)
y_all = np.array(y_all)
pos_count = int(np.sum(y_all))

print(f"Total dataset: {len(X_all):,} pairs (Positives: {pos_count:,} [{pos_count/len(X_all)*100:.2f}%], Negatives: {len(y_all)-pos_count:,})")
sys.stdout.flush()

# Train/Val Split at the Entity Level (12,000 train, 3,000 val)
all_s1_keys = list(s1_dict.keys())
np.random.seed(42)
np.random.shuffle(all_s1_keys)
train_s1_set = set(all_s1_keys[:12000])
val_s1_set = set(all_s1_keys[12000:])

train_indices = [i for i, (s1_id, _) in enumerate(pair_meta) if s1_id in train_s1_set]
val_indices = [i for i, (s1_id, _) in enumerate(pair_meta) if s1_id in val_s1_set]

X_train = df_X.iloc[train_indices][feature_cols].values
y_train = y_all[train_indices]
X_val = df_X.iloc[val_indices][feature_cols].values
y_val = y_all[val_indices]

print(f"Train set: {len(X_train):,} pairs (Positives: {int(np.sum(y_train)):,})")
print(f"Val set:   {len(X_val):,} pairs (Positives: {int(np.sum(y_val)):,})")
sys.stdout.flush()

print("\nFitting LightGBM Classifier...")
t_fit = time.time()
clf = lgb.LGBMClassifier(
    n_estimators=350,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight='balanced',
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print(f"LightGBM fitted in {time.time() - t_fit:.2f}s!")
sys.stdout.flush()

# Evaluate on Validation Set to Calibrate Macro F_0.5 Threshold
print("\n--> Evaluating Thresholds for Macro F_0.5 Calibration...")
val_probs = clf.predict_proba(X_val)[:, 1]

val_preds_by_s1 = defaultdict(list)
for idx_v, i in enumerate(val_indices):
    s1_id, cid = pair_meta[i]
    val_preds_by_s1[s1_id].append((cid, float(val_probs[idx_v])))

def compute_macro_f05(tau):
    f05_scores = []
    beta_sq = 0.5 ** 2
    for s1_id in val_s1_set:
        true_targets = gt_map[s1_id]
        pred_targets = {cid for cid, p in val_preds_by_s1[s1_id] if p >= tau}

        tp = len(pred_targets & true_targets)
        fp = len(pred_targets - true_targets)
        fn = len(true_targets - pred_targets)

        if not true_targets and not pred_targets:
            f05_scores.append(1.0)
            continue
        if not pred_targets or (tp + fp) == 0:
            f05_scores.append(0.0)
            continue

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec + rec == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.0 + beta_sq) * (prec * rec) / (beta_sq * prec + rec)
            f05_scores.append(f05)

    return float(np.mean(f05_scores))

best_tau = 0.9610
best_score = compute_macro_f05(best_tau)
print(f"Macro F_0.5 at baseline tau={best_tau:.4f}: {best_score:.4f}")

for tau in [0.90, 0.92, 0.94, 0.95, 0.96, 0.961, 0.965, 0.97, 0.98]:
    sc = compute_macro_f05(tau)
    print(f"  tau={tau:.4f} -> Macro F_0.5 = {sc:.4f}")
    if sc > best_score:
        best_score = sc
        best_tau = tau

print(f"\n--> Optimal Decision Threshold: tau = {best_tau:.4f} (Validation Macro F_0.5 = {best_score:.4f})")

# Save model files
clf.booster_.save_model(MODEL_LGB_PATH)
clf.booster_.save_model(SRC_LGB_PATH)
print(f"Saved LightGBM booster to {MODEL_LGB_PATH} and {SRC_LGB_PATH}")

model_payload = {
    'classifier': clf,
    'feature_cols': feature_cols,
    'optimal_threshold': best_tau,
    'val_macro_f05': best_score
}

with open(MODEL_PKL_PATH, "wb") as f:
    pickle.dump(model_payload, f)
with open(SRC_PKL_PATH, "wb") as f:
    pickle.dump(model_payload, f)
print(f"Saved model payload to {MODEL_PKL_PATH} and {SRC_PKL_PATH}")

print("=" * 70)
print("ACCELERATED TRAINING COMPLETE!")
print("=" * 70)

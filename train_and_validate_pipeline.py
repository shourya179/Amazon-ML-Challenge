#!/usr/bin/env python3
"""
Leak-Free End-to-End Entity Resolution Training & Validation Pipeline
Features:
  1. Strict Entity-Level 20% Validation Split.
  2. 100% BLIND candidate blocking for validation (NO true-match injection).
  3. Separate S2 and S3 candidate retrieval (Top K_s2 + Top K_s3).
  4. 25-feature pairwise feature extractor.
  5. Gradient Boosted Decision Tree (LightGBM) with balanced class weights.
  6. Coarse (step 0.01) + Fine (step 0.001) Macro F_0.5 threshold search with singletons.
"""

import os
import re
import math
import time
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

# ==============================================================================
# 1. TEXT NORMALIZATION & PREPROCESSING
# ==============================================================================

LEGAL_SUFFIXES = {
    'pvt ltd', 'private limited', 'pvt', 'ltd', 'limited', 'inc', 'incorporated',
    'corp', 'corporation', 'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'holding', 'holdings', 'group', 'services', 'solutions', 'technologies', 'tech',
    'industries', 'international', 'intl', 'gmbh', 'sa', 'sarl', 'sas', 'bv', 'nv',
    'plc', 'spa', 'srl', 'sl', 'cia', 'assoc', 'associates'
}

ADDR_MAP = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
    'ln': 'lane', 'dr': 'drive', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'hwy': 'highway', 'pkwy': 'parkway', 'fl': 'floor', 'flr': 'floor', 'apt': 'apartment',
    'ste': 'suite', 'bldg': 'building', 'opp': 'opposite', 'nr': 'near', 'dist': 'district',
    'teh': 'tehsil', 'taluk': 'taluka', 'stn': 'station', 'nagar': 'nagar', 'marg': 'marg'
}

def clean_unicode(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text.lower().strip()

def normalize_text_general(text: str) -> str:
    text = clean_unicode(text)
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'[/\\_\-+,.:;!?\'"()\[\]{}|@#*^~`]', ' ', text)
    return ' '.join(text.split())

def clean_name_and_core(name: str) -> Tuple[str, str]:
    cleaned = normalize_text_general(name)
    tokens = cleaned.split()
    core_tokens = list(tokens)
    while len(core_tokens) > 1:
        if len(core_tokens) >= 2 and f"{core_tokens[-2]} {core_tokens[-1]}" in LEGAL_SUFFIXES:
            core_tokens = core_tokens[:-2]
        elif core_tokens[-1] in LEGAL_SUFFIXES:
            core_tokens = core_tokens[:-1]
        else:
            break
    core = ' '.join(core_tokens) if core_tokens else cleaned
    return cleaned, core

def clean_address(addr: str) -> str:
    cleaned = normalize_text_general(addr)
    tokens = cleaned.split()
    return ' '.join([ADDR_MAP.get(t, t) for t in tokens])

def extract_clean_numbers(addr: str) -> Set[str]:
    res = set()
    for n in re.findall(r'\b\d+\b', addr):
        try:
            norm = str(int(n))
            if 1 <= len(norm) <= 7:
                res.add(norm)
        except ValueError:
            pass
    return res


# ==============================================================================
# 2. STRING SIMILARITY FUNCTIONS
# ==============================================================================

def levenshtein_ratio(s1: str, s2: str) -> float:
    if s1 == s2: return 1.0
    if not s1 or not s2: return 0.0
    if len(s1) > len(s2): s1, s2 = s2, s1
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        new_dist = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_dist.append(distances[i1])
            else:
                new_dist.append(1 + min((distances[i1], distances[i1 + 1], new_dist[-1])))
        distances = new_dist
    return 1.0 - (distances[-1] / max(len(s1), len(s2)))

def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    if s1 == s2: return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0: return 0.0
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]: continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0: return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]: continue
        while not s2_matches[k]: k += 1
        if s1[i] != s2[k]: transpositions += 1
        k += 1
    jaro = ((matches / len1) + (matches / len2) + ((matches - transpositions / 2.0) / matches)) / 3.0
    l = 0
    for i in range(min(max_l, min(len1, len2))):
        if s1[i] == s2[i]: l += 1
        else: break
    return jaro + (l * p * (1.0 - jaro))

def get_char_ngrams(s: str, n: int = 3) -> Set[str]:
    s = f"^{s}$"
    if len(s) < n: return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}


# ==============================================================================
# 3. PAIRWISE FEATURE VECTOR (25 FEATURES)
# ==============================================================================

def extract_pair_features(s1: dict, tgt: dict, blocking_score: float = 0.0) -> Dict[str, float]:
    s1_name, c_name = s1['clean_name'], tgt['clean_name']
    s1_core, c_core = s1['core_name'], tgt['core_name']
    s1_addr, c_addr = s1['clean_addr'], tgt['clean_addr']

    s1_n_toks, c_n_toks = set(s1_name.split()), set(c_name.split())
    s1_c_toks, c_c_toks = set(s1_core.split()), set(c_core.split())
    s1_a_toks, c_a_toks = set(s1_addr.split()), set(c_addr.split())

    s1_nums, c_nums = s1['nums'], tgt['nums']

    # Token overlap and containment
    n_inter = len(s1_n_toks & c_n_toks)
    n_union = len(s1_n_toks | c_n_toks)
    name_jaccard = n_inter / n_union if n_union > 0 else 0.0

    c_inter = len(s1_c_toks & c_c_toks)
    c_union = len(s1_c_toks | c_c_toks)
    core_jaccard = c_inter / c_union if c_union > 0 else 0.0

    name_containment = n_inter / min(len(s1_n_toks), len(c_n_toks)) if min(len(s1_n_toks), len(c_n_toks)) > 0 else 0.0

    a_inter = len(s1_a_toks & c_a_toks)
    a_union = len(s1_a_toks | c_a_toks)
    addr_jaccard = a_inter / a_union if a_union > 0 else 0.0
    addr_containment = a_inter / min(len(s1_a_toks), len(c_a_toks)) if min(len(s1_a_toks), len(c_a_toks)) > 0 else 0.0

    # Digits
    num_inter = len(s1_nums & c_nums)
    num_union = len(s1_nums | c_nums)
    digit_jaccard = num_inter / num_union if num_union > 0 else 0.0

    # First words
    s1_w0 = s1_core.split()[0] if s1_core else ""
    c_w0 = c_core.split()[0] if c_core else ""
    first_word_match = 1.0 if (s1_w0 and s1_w0 == c_w0) else 0.0
    first_char_match = 1.0 if (s1_name and c_name and s1_name[0] == c_name[0]) else 0.0

    # Char 3-grams
    s1_ng = get_char_ngrams(s1_name, 3)
    c_ng = get_char_ngrams(c_name, 3)
    name_char3_jaccard = len(s1_ng & c_ng) / max(1, len(s1_ng | c_ng))

    s1_ang = get_char_ngrams(s1_addr, 3)
    c_ang = get_char_ngrams(c_addr, 3)
    addr_char3_jaccard = len(s1_ang & c_ang) / max(1, len(s1_ang | c_ang))

    return {
        'feat_name_exact': 1.0 if s1_name == c_name else 0.0,
        'feat_core_exact': 1.0 if s1_core == c_core else 0.0,
        'feat_name_lev': levenshtein_ratio(s1_name, c_name),
        'feat_core_lev': levenshtein_ratio(s1_core, c_core),
        'feat_name_jw': jaro_winkler_similarity(s1_name, c_name),
        'feat_core_jw': jaro_winkler_similarity(s1_core, c_core),
        'feat_name_token_jaccard': name_jaccard,
        'feat_core_token_jaccard': core_jaccard,
        'feat_name_containment': name_containment,
        'feat_name_char3_jaccard': name_char3_jaccard,
        'feat_first_word_match': first_word_match,
        'feat_first_char_match': first_char_match,
        'len_name_diff': abs(len(s1_name) - len(c_name)) / max(1, max(len(s1_name), len(c_name))),
        'feat_addr_exact': 1.0 if (s1_addr and s1_addr == c_addr) else 0.0,
        'feat_addr_lev': levenshtein_ratio(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0,
        'feat_addr_jw': jaro_winkler_similarity(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0,
        'feat_addr_token_jaccard': addr_jaccard,
        'feat_addr_containment': addr_containment,
        'feat_addr_char3_jaccard': addr_char3_jaccard,
        'digit_overlap': float(num_inter),
        'digit_jaccard': digit_jaccard,
        'has_common_digit': 1.0 if num_inter > 0 else 0.0,
        'blocking_score': float(blocking_score),
        'is_source2': 1.0 if tgt['id'].startswith('S2-') else 0.0
    }


# ==============================================================================
# 4. COMPOSITE KEY GENERATION
# ==============================================================================

def get_composite_keys(clean_name: str, core_name: str, clean_addr: str, nums: Set[str]) -> List[str]:
    name_words = core_name.split()
    addr_words = [t for t in clean_addr.split() if t not in {'street', 'road', 'avenue', 'lane', 'drive', 'floor', 'suite', 'unit'}]
    num_list = sorted(list(nums))

    keys = []
    # 1. High-precision: Name word + Number
    for nw in name_words[:4]:
        for nm in num_list[:3]:
            keys.append(f"WN:{nw}_{nm}")
            if len(nw) >= 4:
                keys.append(f"P3N:{nw[:3]}_{nm}")

    # 2. Name word pairs
    if len(name_words) >= 2:
        keys.append(f"WW:{name_words[0]}_{name_words[1]}")
        if len(name_words) >= 3:
            keys.append(f"WW:{name_words[0]}_{name_words[2]}")

    # 3. Address word + Number
    for aw in addr_words[:2]:
        for nm in num_list[:2]:
            keys.append(f"AN:{aw}_{nm}")

    # 4. Standalone core words
    for nw in name_words[:3]:
        keys.append(f"W:{nw}")

    return keys


# ==============================================================================
# 5. METRIC & COARSE/FINE THRESHOLD OPTIMIZATION
# ==============================================================================

def compute_macro_f05(preds_by_s1: Dict[str, List[Tuple[str, float]]], val_gt: Dict[str, Set[str]], all_s1_ids: List[str], threshold: float) -> float:
    f_scores = []
    for s1_id in all_s1_ids:
        true_set = val_gt.get(s1_id, set())
        pred_set = {cid for (cid, p) in preds_by_s1.get(s1_id, []) if p >= threshold}

        if not true_set:
            f_scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
        if not pred_set:
            f_scores.append(0.0)
            continue

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec + rec == 0:
            f_scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            f_scores.append(f05)

    return float(np.mean(f_scores))


# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    print("="*70)
    print("   LEAK-FREE BUSINESS ENTITY RESOLUTION PIPELINE")
    print("="*70)

    # 1. Load Ground Truth
    print("\n--> [1/6] Loading Ground Truth for Evaluation...")
    gt_map = {}
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2 and p[1].strip():
                gt_map[p[0]] = set(p[1].split(','))
            else:
                gt_map[p[0]] = set()
            if len(gt_map) >= 25000:
                break

    all_s1_list = list(gt_map.keys())
    # Strict 80/20 Entity-level split
    train_s1_ids, val_s1_ids = train_test_split(all_s1_list, test_size=0.20, random_state=42)
    val_s1_set = set(val_s1_ids)

    print(f"Entities: Total={len(all_s1_list)}, Train={len(train_s1_ids)}, Validation={len(val_s1_ids)}")
    val_true_s2 = sum(len([x for x in gt_map[s] if x.startswith('S2-')]) for s in val_s1_ids)
    val_true_s3 = sum(len([x for x in gt_map[s] if x.startswith('S3-')]) for s in val_s1_ids)
    print(f"Validation Ground Truth Matches: S2={val_true_s2}, S3={val_true_s3} (Total={val_true_s2 + val_true_s3})")

    # 2. Load S1 Records
    print("\n--> [2/6] Loading S1 Records...")
    s1_dict = {}
    all_needed_s1 = set(all_s1_list)
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if p[0] in all_needed_s1:
                clean_n, core_n = clean_name_and_core(p[1])
                clean_a = clean_address(p[2])
                s1_dict[p[0]] = {
                    'id': p[0], 'clean_name': clean_n, 'core_name': core_n,
                    'clean_addr': clean_a, 'nums': extract_clean_numbers(clean_a),
                    'country': p[3].strip().lower()
                }
            if len(s1_dict) == len(all_needed_s1):
                break

    # 3. Index Target Sources S2 and S3 Separately
    def build_source_index(filename, prefix, max_records=2500000):
        print(f"Indexing {prefix} from {filename}...")
        t0 = time.time()
        idx = defaultdict(list)
        records = {}
        count = 0
        with open(os.path.join(train_dir, filename), "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 4:
                    eid = p[0]
                    clean_n, core_n = clean_name_and_core(p[1])
                    clean_a = clean_address(p[2])
                    nums = extract_clean_numbers(clean_a)
                    c = p[3].strip().lower()

                    rec = {
                        'id': eid, 'clean_name': clean_n, 'core_name': core_n,
                        'clean_addr': clean_a, 'nums': nums, 'country': c
                    }
                    records[eid] = rec
                    int_id = int(eid[3:])

                    for k in get_composite_keys(clean_n, core_n, clean_a, nums):
                        idx[(c, k)].append(int_id)

                    count += 1
                    if count >= max_records:
                        break

        # Compute key weights
        N = float(count)
        weights = {}
        for k, id_list in idx.items():
            df = len(id_list)
            if df > 5000: continue
            idf = math.log((N + 1.0) / (df + 1.0))
            if k[1].startswith("WN:") or k[1].startswith("WW:"):
                weights[k] = idf * 2.5
            elif k[1].startswith("P3N:") or k[1].startswith("AN:"):
                weights[k] = idf * 1.5
            else:
                weights[k] = idf

        print(f"  Indexed {count} records in {time.time() - t0:.2f}s. Valid keys: {len(weights)}")
        return records, idx, weights

    print("\n--> [3/6] Indexing S2 and S3 Target Pools Separately...")
    s2_records, s2_idx, s2_weights = build_source_index("train_source2.tsv", "S2", max_records=3000000)
    s3_records, s3_idx, s3_weights = build_source_index("train_source3.tsv", "S3", max_records=3000000)
    all_target_records = {**s2_records, **s3_records}

    # 4. Generate Candidate Pairs (BLIND FOR VALIDATION)
    print("\n--> [4/6] Generating Candidate Pairs (Strictly Blind for Validation)...")
    def retrieve_candidates(s1, idx, weights, prefix, k=35):
        c = s1['country']
        keys = get_composite_keys(s1['clean_name'], s1['core_name'], s1['clean_addr'], s1['nums'])
        scores = defaultdict(float)
        for k_str in keys:
            lookup_key = (c, k_str)
            if lookup_key in weights:
                w = weights[lookup_key]
                for int_id in idx[lookup_key]:
                    scores[int_id] += w
        return [(f"{prefix}-{x[0]}", x[1]) for x in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]

    val_found_s2 = 0
    val_found_s3 = 0
    val_candidates_by_s1 = {}
    val_pairs = []

    X_train, y_train = [], []

    t0 = time.time()
    for s1_id in all_s1_list:
        s1 = s1_dict[s1_id]
        true_matches = gt_map[s1_id]
        is_val = s1_id in val_s1_set

        # Separate retrieval: Top K_S2 + Top K_S3
        cands_s2 = retrieve_candidates(s1, s2_idx, s2_weights, "S2", k=35)
        cands_s3 = retrieve_candidates(s1, s3_idx, s3_weights, "S3", k=35)

        s2_ids = {c[0] for c in cands_s2}
        s3_ids = {c[0] for c in cands_s3}
        combined_cands = cands_s2 + cands_s3

        if is_val:
            # 100% BLIND VALIDATION: NO true target injection!
            val_found_s2 += len(true_matches & s2_ids)
            val_found_s3 += len(true_matches & s3_ids)
            val_candidates_by_s1[s1_id] = [c[0] for c in combined_cands]

            for cid, b_score in combined_cands:
                if cid in all_target_records:
                    feats = extract_pair_features(s1, all_target_records[cid], b_score)
                    lbl = 1 if cid in true_matches else 0
                    val_pairs.append((s1_id, cid, feats, lbl))
        else:
            # For training: use the candidate pool
            # (True positives can be included to teach the classifier)
            train_cands_dict = {c[0]: c[1] for c in combined_cands}
            # Add true matches if available in target pool
            for tm in true_matches:
                if tm in all_target_records and tm not in train_cands_dict:
                    train_cands_dict[tm] = 5.0 # baseline score

            for cid, b_score in train_cands_dict.items():
                if cid in all_target_records:
                    feats = extract_pair_features(s1, all_target_records[cid], b_score)
                    lbl = 1 if cid in true_matches else 0
                    X_train.append(feats)
                    y_train.append(lbl)

    print(f"Candidate generation completed in {time.time() - t0:.2f}s.")
    print("\n" + "="*50)
    print("BLIND CANDIDATE BLOCKING METRICS (VALIDATION):")
    print(f"  S2 True Recall: {val_found_s2}/{val_true_s2} ({val_found_s2/val_true_s2*100:.2f}%)")
    print(f"  S3 True Recall: {val_found_s3}/{val_true_s3} ({val_found_s3/val_true_s3*100:.2f}%)")
    total_val_true = val_true_s2 + val_true_s3
    total_val_found = val_found_s2 + val_found_s3
    print(f"  Overall Blind Recall Ceiling: {total_val_found}/{total_val_true} ({total_val_found/total_val_true*100:.2f}%)")
    avg_cands = sum(len(c) for c in val_candidates_by_s1.values()) / len(val_s1_ids)
    print(f"  Average Candidate Pairs per S1: {avg_cands:.1f}")
    print("="*50)

    # 5. Train LightGBM Model on 25 Features
    print("\n--> [5/6] Training LightGBM Classifier on 25 Pairwise Features...")
    df_X_train = pd.DataFrame(X_train)
    feature_cols = list(df_X_train.columns)
    print(f"Feature count: {len(feature_cols)}")
    print(f"Training pairs: {len(X_train)} (Positives: {sum(y_train)}, Negatives: {len(y_train) - sum(y_train)})")

    if HAS_LGB:
        clf = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        )
    else:
        clf = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, class_weight='balanced', random_state=42)

    clf.fit(df_X_train[feature_cols], np.array(y_train))
    print("Model training complete.")

    # 6. Honest Evaluation & Coarse + Fine Threshold Search
    print("\n--> [6/6] Scoring Validation Pairs and Optimizing Macro F_0.5...")
    df_X_val = pd.DataFrame([p[2] for p in val_pairs])
    val_probs = clf.predict_proba(df_X_val[feature_cols])[:, 1]

    val_preds_by_s1 = defaultdict(list)
    for (s1_id, cid, _, _), prob in zip(val_pairs, val_probs):
        val_preds_by_s1[s1_id].append((cid, float(prob)))

    # A. Coarse Search: 0.30 to 0.95 with step 0.01
    print("\nRunning Coarse Threshold Search (0.30 to 0.95, step 0.01)...")
    best_coarse_th = 0.50
    best_coarse_score = -1.0

    coarse_grid = np.linspace(0.30, 0.95, 66)
    for th in coarse_grid:
        score = compute_macro_f05(val_preds_by_s1, gt_map, val_s1_ids, th)
        if score > best_coarse_score:
            best_coarse_score = score
            best_coarse_th = th

    print(f"Coarse Best: Macro F_0.5 = {best_coarse_score:.4f} at threshold: {best_coarse_th:.3f}")

    # B. Fine Search: [best - 0.03, best + 0.03] with step 0.001
    print("\nRunning Fine Threshold Search around coarse optimum...")
    fine_grid = np.linspace(best_coarse_th - 0.03, best_coarse_th + 0.03, 61)
    best_fine_th = best_coarse_th
    best_fine_score = best_coarse_score

    for th in fine_grid:
        score = compute_macro_f05(val_preds_by_s1, gt_map, val_s1_ids, th)
        if score > best_fine_score:
            best_fine_score = score
            best_fine_th = th

    print("\n" + "="*60)
    print("FINAL END-TO-END VALIDATION RESULTS:")
    print(f"  Optimal Macro F_0.5 Score: {best_fine_score:.4f}")
    print(f"  Optimal Decision Threshold: {best_fine_th:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()

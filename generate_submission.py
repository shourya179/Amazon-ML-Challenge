#!/usr/bin/env python3
"""
High-Performance Submission Generation Pipeline for Business Entity Resolution
Produces:
  1. output/candidate_pairs.tsv
  2. output/matching_results.tsv

Optimized for:
  - Strict df <= 500 threshold across all blocking keys
  - International address stop words
  - Direct 2D numpy arrays into LightGBM (bypassing pandas DataFrame overhead for 10x faster batching)
  - Per-country persistent disk saving (output/part_{country}.tsv)
  - Final deterministic merger in exact test_source1.tsv order
"""

import os
import sys
import gc
import math
import time
import heapq
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz.distance.Levenshtein import normalized_similarity as levenshtein_ratio
from rapidfuzz.distance.JaroWinkler import similarity as jaro_winkler_similarity

WORKSPACE_DIR = r"c:\Users\Shourya\Desktop\Amazon ML Challenge"
CLEAN_DIR = os.path.join(WORKSPACE_DIR, "cleaned_data", "test")
TEST_RAW_DIR = os.path.join(WORKSPACE_DIR, "student_resource", "dataset", "test")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LGB_PATH = os.path.join(OUTPUT_DIR, "model_lgb.txt")
CAND_OUT_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCH_OUT_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")

print("=" * 75)
print("HIGH-PERFORMANCE TEST SUBMISSION GENERATION PIPELINE")
print("=" * 75)
sys.stdout.flush()

# 1. Load Trained Model
print("--> [1/4] Loading Trained Model...")
if os.path.exists(MODEL_LGB_PATH):
    booster = lgb.Booster(model_file=MODEL_LGB_PATH)
    optimal_threshold = 0.9610
    print(f"Loaded LightGBM model from {MODEL_LGB_PATH} with threshold tau={optimal_threshold}")
else:
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "rb") as f:
        payload = pickle.load(f)
    booster = payload['classifier'].booster_
    optimal_threshold = payload.get('optimal_threshold', 0.9610)
    print(f"Loaded model from model.pkl with threshold tau={optimal_threshold}")

# 2. Key Generation & Feature Extraction
ADDR_STOP_WORDS = {
    'street', 'st', 'road', 'rd', 'ave', 'avenue', 'suite', 'unit', 'dr', 'drive',
    'rue', 'chemin', 'route', 'allee', 'boulevard', 'blvd', 'lane', 'ln', 'place', 'pl',
    'court', 'ct', 'nagar', 'marg', 'road', 'floor', 'fl', 'bldg', 'building'
}

def get_char_ngrams(s: str, n: int = 3) -> Set[str]:
    s = f"^{s}$"
    if len(s) < n: return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def get_composite_keys(core_name: str, clean_addr: str, nums: List[str]) -> List[str]:
    name_words = core_name.split()
    addr_words = [t for t in clean_addr.split() if t not in ADDR_STOP_WORDS]

    keys = []
    # 1. WN and P3N
    for nw in name_words[:5]:
        for nm in nums[:3]:
            keys.append(f"WN:{nw}_{nm}")
            if len(nw) >= 4:
                keys.append(f"P3N:{nw[:3]}_{nm}")

    # 2. WW
    if len(name_words) >= 2:
        keys.append(f"WW:{name_words[0]}_{name_words[1]}")
        if len(name_words) >= 3:
            keys.append(f"WW:{name_words[0]}_{name_words[2]}")

    # 3. AN
    for aw in addr_words[:3]:
        for nm in nums[:3]:
            keys.append(f"AN:{aw}_{nm}")

    # 4. Standalone W (skip short tokens < 3 chars)
    for nw in name_words[:4]:
        if len(nw) >= 3:
            keys.append(f"W:{nw}")

    return keys

def extract_pair_features_row(s1: dict, tgt: dict, blocking_score: float = 0.0) -> List[float]:
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

    return [
        1.0 if s1_name == c_name else 0.0,
        1.0 if s1_core == c_core else 0.0,
        float(levenshtein_ratio(s1_name, c_name)),
        float(levenshtein_ratio(s1_core, c_core)),
        float(jaro_winkler_similarity(s1_name, c_name)),
        float(jaro_winkler_similarity(s1_core, c_core)),
        name_jaccard,
        core_jaccard,
        name_containment,
        name_char3_jaccard,
        first_word_match,
        first_char_match,
        abs(len(s1_name) - len(c_name)) / max(1, max(len(s1_name), len(c_name))),
        1.0 if (s1_addr and s1_addr == c_addr) else 0.0,
        float(levenshtein_ratio(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0,
        float(jaro_winkler_similarity(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0,
        addr_jaccard,
        addr_containment,
        addr_char3_jaccard,
        float(num_inter),
        digit_jaccard,
        1.0 if num_inter > 0 else 0.0,
        float(blocking_score),
        1.0 if tgt['id'].startswith('S2-') else 0.0
    ]

# 3. Read Ordered S1 Entity IDs & Countries
print("--> [2/4] Scanning Test S1 ID Order & Countries...")
sys.stdout.flush()

ordered_s1_ids = []
s1_country_map = {}

with open(os.path.join(CLEAN_DIR, "clean_test_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        s1_id = p[0]
        country = p[5].strip().lower() if len(p) > 5 else "unknown"
        ordered_s1_ids.append(s1_id)
        s1_country_map[s1_id] = country

print(f"Total Test Entities: {len(ordered_s1_ids):,}")
country_counts = defaultdict(int)
for c in s1_country_map.values():
    country_counts[c] += 1
for c, cnt in sorted(country_counts.items()):
    print(f"  {c.upper()}: {cnt:,} entities")
sys.stdout.flush()

# 4. Stream Country-by-Country
print("\n--> [3/4] Processing Each Country with Streamed Inverted Indexing...")
sys.stdout.flush()

country_order = ['france', 'us', 'india']

for country in country_order:
    if country_counts[country] == 0:
        continue

    part_file = os.path.join(OUTPUT_DIR, f"part_{country}.tsv")
    # If this country was already generated, skip re-computation!
    if os.path.exists(part_file) and os.path.getsize(part_file) > 100000:
        print(f"\n[{country.upper()}] part file already exists ({part_file}). Skipping recomputation!")
        continue

    print(f"\n" + "="*65)
    print(f"STARTING COUNTRY: [{country.upper()}] ({country_counts[country]:,} S1 Entities)")
    print(f"="*65)
    sys.stdout.flush()

    # Load S1 records for this country only
    t0_c = time.time()
    s1_dict = {}
    with open(os.path.join(CLEAN_DIR, "clean_test_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 6 and p[5].strip().lower() == country:
                eid = p[0]
                clean_n = p[1]
                core_n = p[2] if p[2] else clean_n
                clean_a = p[3]
                nums = p[4].split() if p[4].strip() else []
                s1_dict[eid] = {
                    'id': eid,
                    'clean_name': clean_n,
                    'core_name': core_n,
                    'clean_addr': clean_a,
                    'nums': nums,
                    'country': country,
                    'keys': get_composite_keys(core_n, clean_a, nums)
                }

    print(f"  Loaded {len(s1_dict):,} S1 records for [{country}] in {time.time() - t0_c:.2f}s.")
    sys.stdout.flush()

    country_cands = defaultdict(list)
    needed_target_ids = set()

    # Index target source
    def index_target_source(filename: str, prefix: str):
        t0 = time.time()
        idx = defaultdict(list)
        path = os.path.join(CLEAN_DIR, filename)
        print(f"  Streaming & indexing {filename} for [{country}]...")
        sys.stdout.flush()

        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 6 and p[5].strip().lower() == country:
                    int_id = int(p[0][3:])
                    core_n = p[2]
                    clean_a = p[3]
                    nums = p[4].split() if p[4].strip() else []

                    for k in get_composite_keys(core_n, clean_a, nums):
                        idx[k].append(int_id)
                    cnt += 1

        N = float(cnt)
        print(f"  Indexed {cnt:,} {prefix} targets in {time.time() - t0:.2f}s. Calculating IDFs...")
        sys.stdout.flush()

        weights = {}
        for k, id_list in idx.items():
            df = len(id_list)
            # Prune high-frequency collisions: df > 500
            if df > 500:
                continue
            idf = math.log((N + 1.0) / (df + 1.0))
            weights[k] = idf * 2.5 if (k.startswith("WN:") or k.startswith("WW:")) else idf

        print(f"  Retained {len(weights):,} high-discriminative keys. Scoring {len(s1_dict):,} S1 entities against {prefix}...")
        sys.stdout.flush()

        t_score = time.time()
        for idx_s, (s1_id, s1) in enumerate(s1_dict.items()):
            scores = defaultdict(float)
            for k in s1['keys']:
                if k in weights:
                    w = weights[k]
                    for int_id in idx[k]:
                        scores[int_id] += w

            if scores:
                top_items = heapq.nlargest(35, scores.items(), key=lambda x: x[1])
                top_cands = [(f"{prefix}-{x[0]}", x[1]) for x in top_items]
                country_cands[s1_id].extend(top_cands)
                for cid, _ in top_cands:
                    needed_target_ids.add(cid)

            if (idx_s + 1) % 50000 == 0:
                print(f"    Scored {idx_s + 1:,} / {len(s1_dict):,} S1 entities ({(idx_s + 1)/(time.time() - t_score):.0f} ent/s)...")
                sys.stdout.flush()

        print(f"  Candidate scoring against {prefix} completed in {time.time() - t_score:.2f}s!")
        sys.stdout.flush()

        del idx, weights
        gc.collect()

    index_target_source("clean_test_source2.tsv", "S2")
    index_target_source("clean_test_source3.tsv", "S3")

    total_pairs = sum(len(c) for c in country_cands.values())
    print(f"\n  Candidate summary for [{country}]:")
    print(f"    Total candidate pairs: {total_pairs:,} (avg {total_pairs/len(s1_dict):.1f} cands/entity)")
    print(f"    Unique target records needed: {len(needed_target_ids):,}")
    sys.stdout.flush()

    # Load target cache
    print(f"  Caching {len(needed_target_ids):,} target records for feature extraction...")
    t_load = time.time()
    target_cache = {}
    for filename in ["clean_test_source2.tsv", "clean_test_source3.tsv"]:
        path = os.path.join(CLEAN_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                tab_idx = line.find('\t')
                if tab_idx != -1:
                    eid = line[:tab_idx]
                    if eid in needed_target_ids:
                        p = line.rstrip('\n').split('\t')
                        if len(p) >= 6:
                            target_cache[eid] = {
                                'id': eid,
                                'clean_name': p[1],
                                'core_name': p[2],
                                'clean_addr': p[3],
                                'nums': p[4].split() if p[4].strip() else [],
                                'country': country
                            }
                            if len(target_cache) == len(needed_target_ids):
                                break

    print(f"  Target cache populated ({len(target_cache):,} records) in {time.time() - t_load:.2f}s.")
    sys.stdout.flush()

    # Stream batch scoring with direct 2D numpy arrays into LightGBM (10x faster)
    print(f"  Running feature extraction & LightGBM scoring (tau = {optimal_threshold:.4f})...")
    t_score_all = time.time()
    s1_ids_country = list(s1_dict.keys())
    batch_size = 20000

    matches_cnt = 0
    singletons_cnt = 0

    # Persistent output file for this country: s1_id \t cands_str \t matches_str
    part_f = open(part_file, "w", encoding="utf-8")

    for b_start in range(0, len(s1_ids_country), batch_size):
        b_ids = s1_ids_country[b_start : b_start + batch_size]
        batch_rows = []
        batch_meta = [] # (s1_id, cid)
        cands_by_s1 = {}

        for s1_id in b_ids:
            cands = country_cands.get(s1_id, [])
            cand_ids = [c[0] for c in cands]
            cands_by_s1[s1_id] = ",".join(cand_ids)

            s1 = s1_dict[s1_id]
            for cid, b_score in cands:
                if cid in target_cache:
                    batch_rows.append(extract_pair_features_row(s1, target_cache[cid], b_score))
                    batch_meta.append((s1_id, cid))

        matched_by_s1 = defaultdict(list)
        if batch_rows:
            # Fast direct numpy conversion without pandas DataFrame overhead
            X_batch = np.array(batch_rows, dtype=np.float32)
            probs = booster.predict(X_batch)

            for (s1_id, cid), pr in zip(batch_meta, probs):
                if pr >= optimal_threshold:
                    matched_by_s1[s1_id].append(cid)

        for s1_id in b_ids:
            raw_matches = matched_by_s1.get(s1_id, [])
            cand_str = cands_by_s1[s1_id]
            cand_set = set(cand_str.split(',')) if cand_str else set()

            clean_matches = []
            seen_m = set()
            for mid in raw_matches:
                if mid not in seen_m and mid in cand_set:
                    seen_m.add(mid)
                    clean_matches.append(mid)

            match_str = ",".join(clean_matches)
            part_f.write(f"{s1_id}\t{cand_str}\t{match_str}\n")

            if len(clean_matches) == 0:
                singletons_cnt += 1
            else:
                matches_cnt += len(clean_matches)

        done = min(b_start + batch_size, len(s1_ids_country))
        if done % 50000 == 0 or done == len(s1_ids_country):
            print(f"    Scored {done:,} / {len(s1_ids_country):,} entities ({(done)/(time.time() - t_score_all):.0f} ent/s)...")
            sys.stdout.flush()

    part_f.close()
    print(f"  Finished [{country.upper()}] in {time.time() - t0_c:.2f}s!")
    print(f"    Total matches: {matches_cnt:,} | Singletons: {singletons_cnt:,} ({singletons_cnt/len(s1_dict)*100:.2f}%)")
    print(f"    Saved country part to {part_file}")
    sys.stdout.flush()

    # Free country memory completely
    del s1_dict, country_cands, needed_target_ids, target_cache
    gc.collect()

# 5. Deterministic Assembly of Final Submission TSVs in Exact Test Order
print("\n--> [4/4] Assembling Final Submission TSVs in Exact Test Order...")
sys.stdout.flush()
t_w = time.time()

# Load all part files into indexed dictionary
print("  Loading country part files into indexed lookup...")
results_lookup = {}
for country in country_order:
    part_file = os.path.join(OUTPUT_DIR, f"part_{country}.tsv")
    if os.path.exists(part_file):
        with open(part_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                s1_id = p[0]
                c_str = p[1] if len(p) > 1 else ""
                m_str = p[2] if len(p) > 2 else ""
                results_lookup[s1_id] = (c_str, m_str)

print(f"  Loaded {len(results_lookup):,} entity predictions.")

print(f"Writing {CAND_OUT_PATH}...")
with open(CAND_OUT_PATH, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in ordered_s1_ids:
        c_str = results_lookup.get(s1_id, ("", ""))[0]
        f.write(f"{s1_id}\t{c_str}\n")

print(f"Writing {MATCH_OUT_PATH}...")
with open(MATCH_OUT_PATH, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in ordered_s1_ids:
        m_str = results_lookup.get(s1_id, ("", ""))[1]
        f.write(f"{s1_id}\t{m_str}\n")

print(f"TSV files successfully written in {time.time() - t_w:.2f}s!")
sys.stdout.flush()

# Final Summary
total_entities = len(ordered_s1_ids)
total_matches = sum(len(m.split(',')) for _, m in results_lookup.values() if m)
total_singletons = sum(1 for _, m in results_lookup.values() if not m)

print("\n" + "=" * 75)
print("FINAL SUBMISSION COMPLETE:")
print("=" * 75)
print(f"  Total S1 Entities:     {total_entities:,}")
print(f"  Total Matches:         {total_matches:,}")
print(f"  Total Singletons:      {total_singletons:,} ({total_singletons/total_entities*100:.2f}%)")
print(f"  Candidate Output:      {CAND_OUT_PATH}")
print(f"  Matching Output:       {MATCH_OUT_PATH}")
print("=" * 75)
sys.stdout.flush()

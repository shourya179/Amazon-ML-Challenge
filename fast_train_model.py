import os
import re
import sys
import gc
import math
import time
import pickle
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("="*60)
print("FAST MEMORY-SEQUENTIAL MODEL TRAINING")
print("="*60)
sys.stdout.flush()

# 1. Load Ground Truth
print("--> [1/5] Loading Ground Truth for 15,000 S1 Entities...")
sys.stdout.flush()
gt_map = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            gt_map[p[0]] = set(p[1].split(','))
        else:
            gt_map[p[0]] = set()
        if len(gt_map) >= 15000:
            break

all_s1_list = list(gt_map.keys())
all_needed_s1 = set(all_s1_list)
print(f"Loaded {len(all_s1_list)} entities for training.")
sys.stdout.flush()

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

def clean_unicode(text):
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower().strip()

def normalize_text(text):
    text = clean_unicode(text)
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'[/\\_\-+,.:;!?\'"()\[\]{}|@#*^~`]', ' ', text)
    return ' '.join(text.split())

def clean_name_and_core(name):
    cleaned = normalize_text(name)
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

def clean_address(addr):
    cleaned = normalize_text(addr)
    tokens = cleaned.split()
    return ' '.join([ADDR_MAP.get(t, t) for t in tokens])

def extract_clean_numbers(addr):
    res = []
    for n in re.findall(r'\b\d+\b', addr):
        try:
            norm = str(int(n))
            if 1 <= len(norm) <= 7:
                res.append(norm)
        except ValueError:
            pass
    return res

def levenshtein_ratio(s1, s2):
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

def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
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

def get_char_ngrams(s, n=3):
    s = f"^{s}$"
    if len(s) < n: return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def extract_pair_features(s1, tgt, blocking_score=0.0):
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

def get_composite_keys(core_name, clean_addr, nums):
    name_words = core_name.split()
    addr_words = [t for t in clean_addr.split() if t not in {'street', 'st', 'road', 'rd', 'ave', 'avenue', 'suite', 'unit', 'dr', 'drive'}]
    keys = []
    for nw in name_words[:4]:
        for nm in nums[:2]:
            keys.append(f"WN:{nw}_{nm}")
            if len(nw) >= 4:
                keys.append(f"P3N:{nw[:3]}_{nm}")
    if len(name_words) >= 2:
        keys.append(f"WW:{name_words[0]}_{name_words[1]}")
    for aw in addr_words[:2]:
        for nm in nums[:2]:
            keys.append(f"AN:{aw}_{nm}")
    for nw in name_words[:3]:
        keys.append(f"W:{nw}")
    return keys

# 2. Load S1 Records
print("--> [2/5] Loading S1 Records...")
sys.stdout.flush()
s1_dict = {}
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

# 3. Retrieve candidates from S2 and S3 sequentially to conserve RAM!
all_s1_candidates = defaultdict(list)
all_needed_target_ids = set()

def process_target_source(filename, prefix, max_rows=1500000):
    print(f"\nIndexing and querying {prefix} ({filename})...")
    sys.stdout.flush()
    t0 = time.time()
    idx = defaultdict(list)
    cnt = 0
    with open(os.path.join(train_dir, filename), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                int_id = int(p[0][3:])
                clean_n, core_n = clean_name_and_core(p[1])
                clean_a = clean_address(p[2])
                nums = extract_clean_numbers(clean_a)
                c = p[3].strip().lower()
                for k in get_composite_keys(core_n, clean_a, nums):
                    idx[(c, k)].append(int_id)
                cnt += 1
                if cnt >= max_rows:
                    break

    N = float(cnt)
    weights = {}
    for k, id_list in idx.items():
        df = len(id_list)
        if df > 5000: continue
        idf = math.log((N + 1.0) / (df + 1.0))
        weights[k] = idf * 2.5 if (k[1].startswith("WN:") or k[1].startswith("WW:")) else idf

    print(f"  Indexed {cnt:,} rows in {time.time() - t0:.2f}s. Keys: {len(weights):,}")
    print(f"  Querying candidates from {prefix}...")
    sys.stdout.flush()

    for s1_id in all_s1_list:
        s1 = s1_dict[s1_id]
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

print("--> [3/5] Querying Candidates Sequentially (Zero Memory Pressure)...")
process_target_source("train_source2.tsv", "S2", max_rows=1500000)
process_target_source("train_source3.tsv", "S3", max_rows=1500000)

print(f"\nTotal candidate pairs generated: {sum(len(c) for c in all_s1_candidates.values()):,}")
print(f"Unique target records needed: {len(all_needed_target_ids):,}")
sys.stdout.flush()

# 4. Stream and cache target records
print("\n--> [4/5] Streaming target records for feature calculation...")
sys.stdout.flush()
target_cache = {}
for filename in ["train_source2.tsv", "train_source3.tsv"]:
    path = os.path.join(train_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4 and p[0] in all_needed_target_ids:
                clean_n, core_n = clean_name_and_core(p[1])
                clean_a = clean_address(p[2])
                target_cache[p[0]] = {
                    'id': p[0], 'clean_name': clean_n, 'core_name': core_n,
                    'clean_addr': clean_a, 'nums': extract_clean_numbers(clean_a),
                    'country': p[3].strip().lower()
                }
                if len(target_cache) == len(all_needed_target_ids):
                    break
    print(f"  Loaded {len(target_cache):,} / {len(all_needed_target_ids):,} records.")
    sys.stdout.flush()

# 5. Extract features & train model
print("\n--> [5/5] Extracting features and fitting classifier...")
sys.stdout.flush()

X_train, y_train = [], []
for s1_id, cand_list in all_s1_candidates.items():
    s1 = s1_dict[s1_id]
    true_set = gt_map[s1_id]
    for cid, b_score in cand_list:
        if cid in target_cache:
            feats = extract_pair_features(s1, target_cache[cid], b_score)
            lbl = 1 if cid in true_set else 0
            X_train.append(feats)
            y_train.append(lbl)

df_X_train = pd.DataFrame(X_train)
feature_cols = list(df_X_train.columns)
pos_count = sum(y_train)
print(f"Training on {len(X_train):,} pairs (Positives: {pos_count:,} [{pos_count/len(X_train)*100:.2f}%], Negatives: {len(y_train)-pos_count:,})")
sys.stdout.flush()

t_fit = time.time()
clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)
clf.fit(df_X_train[feature_cols], np.array(y_train))
print(f"Model fitted in {time.time() - t_fit:.2f}s!")

# Save model pickle
out_model = r"c:\Users\Shourya\Desktop\dataset\trained_model.pkl"
with open(out_model, "wb") as f:
    pickle.dump(clf, f)
print(f"Saved production model to {out_model}!")
sys.stdout.flush()

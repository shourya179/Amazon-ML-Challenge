import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
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

print("--> [1/5] Loading Ground Truth for Training...")
# Load 50,000 ground truth rows
df_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", nrows=50000)

gt_dict = {}
all_matched_target_ids = set()
for _, row in df_gt.iterrows():
    s1_id = row['source1_entity_id']
    m_str = str(row['matched_entity_ids'])
    if m_str and m_str != 'nan':
        m_list = [x.strip() for x in m_str.split(',') if x.strip()]
        gt_dict[s1_id] = set(m_list)
        all_matched_target_ids.update(m_list)
    else:
        gt_dict[s1_id] = set()

all_s1_ids = list(gt_dict.keys())
s1_set = set(all_s1_ids)
print(f"Loaded {len(all_s1_ids)} S1 ground truth entities. Matched targets: {len(all_matched_target_ids)}")

print("--> [2/5] Loading S1 Records...")
s1_records = {}
for chunk in pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(s1_set)]
    for _, row in m.iterrows():
        s1_records[row['entity_id']] = {
            'id': row['entity_id'],
            'name': str(row['business_name']),
            'addr': str(row['business_address']),
            'country': str(row['country']).strip().lower()
        }
    if len(s1_records) == len(s1_set):
        break

print(f"Loaded {len(s1_records)} S1 full records.")

print("--> [3/5] Loading Target Records (S2 & S3)...")
# Load all true matched targets + 100,000 distractors
target_records = {}
for src_name in ["train_source2.tsv", "train_source3.tsv"]:
    loaded_distractors = 0
    for chunk in pd.read_csv(os.path.join(train_dir, src_name), sep="\t", chunksize=150000):
        for _, row in chunk.iterrows():
            eid = row['entity_id']
            if eid in all_matched_target_ids:
                target_records[eid] = {
                    'id': eid,
                    'name': str(row['business_name']),
                    'addr': str(row['business_address']),
                    'country': str(row['country']).strip().lower()
                }
            elif loaded_distractors < 50000:
                target_records[eid] = {
                    'id': eid,
                    'name': str(row['business_name']),
                    'addr': str(row['business_address']),
                    'country': str(row['country']).strip().lower()
                }
                loaded_distractors += 1

print(f"Total target records loaded: {len(target_records)}")

# Preprocessing helpers
LEGAL = {'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}

def clean_toks(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('utf-8').lower()
    s = re.sub(r'[^a-zA-Z0-9]', ' ', s)
    return [t for t in s.split() if t]

def get_core_tokens(name_toks):
    return [t for t in name_toks if t not in LEGAL and len(t) >= 2]

def extract_numbers(addr):
    return set(re.findall(r'\b\d+\b', str(addr)))

# Feature extraction function
def compute_features(s1, tgt):
    s1_name_toks = clean_toks(s1['name'])
    tgt_name_toks = clean_toks(tgt['name'])
    s1_core = get_core_tokens(s1_name_toks)
    tgt_core = get_core_tokens(tgt_name_toks)

    s1_name_set = set(s1_name_toks)
    tgt_name_set = set(tgt_name_toks)
    s1_core_set = set(s1_core)
    tgt_core_set = set(tgt_core)

    name_jaccard = len(s1_name_set & tgt_name_set) / max(1, len(s1_name_set | tgt_name_set))
    core_jaccard = len(s1_core_set & tgt_core_set) / max(1, len(s1_core_set | tgt_core_set))

    # Address features
    s1_addr_toks = set(clean_toks(s1['addr']))
    tgt_addr_toks = set(clean_toks(tgt['addr']))
    addr_jaccard = len(s1_addr_toks & tgt_addr_toks) / max(1, len(s1_addr_toks | tgt_addr_toks))

    # Number overlap
    s1_nums = extract_numbers(s1['addr'])
    tgt_nums = extract_numbers(tgt['addr'])
    num_common = len(s1_nums & tgt_nums)
    num_jaccard = num_common / max(1, len(s1_nums | tgt_nums))

    # First word match
    first_word_match = 1.0 if (s1_core and tgt_core and s1_core[0] == tgt_core[0]) else 0.0

    # Exact string match
    exact_name = 1.0 if s1['name'].strip().lower() == tgt['name'].strip().lower() else 0.0
    exact_addr = 1.0 if s1['addr'].strip().lower() == tgt['addr'].strip().lower() else 0.0

    return [
        name_jaccard, core_jaccard, addr_jaccard,
        num_common, num_jaccard, first_word_match,
        exact_name, exact_addr,
        1.0 if tgt['id'].startswith('S2-') else 0.0
    ]

feature_cols = [
    'name_jaccard', 'core_jaccard', 'addr_jaccard',
    'num_common', 'num_jaccard', 'first_word_match',
    'exact_name', 'exact_addr', 'is_s2'
]

# Build training and validation pairs
print("--> [4/5] Building Feature Matrix for Model Training...")
# Train/Validation split on S1 entities
train_s1, val_s1 = train_test_split(all_s1_ids, test_size=0.20, random_state=42)
train_s1_set = set(train_s1)
val_s1_set = set(val_s1)

X_train, y_train = [], []
X_val, y_val = [], []
val_pair_info = [] # (s1_id, tgt_id)

# Create inverted index on target records for fast candidate retrieval
inv_brand = defaultdict(list)
inv_num = defaultdict(list)
for tid, tgt in target_records.items():
    country = tgt['country']
    toks = get_core_tokens(clean_toks(tgt['name']))
    if toks:
        inv_brand[(country, toks[0])].append(tid)
    for n in extract_numbers(tgt['addr']):
        inv_num[(country, n)].append(tid)

# Generate pairs for training and validation
for s1_id in all_s1_ids:
    s1 = s1_records[s1_id]
    country = s1['country']
    true_targets = gt_dict[s1_id]
    s1_toks = get_core_tokens(clean_toks(s1['name']))
    s1_nums = extract_numbers(s1['addr'])

    candidate_ids = set()
    if s1_toks and (country, s1_toks[0]) in inv_brand:
        candidate_ids.update(inv_brand[(country, s1_toks[0])][:20])
    for n in s1_nums:
        if (country, n) in inv_num:
            candidate_ids.update(inv_num[(country, n)][:15])

    # Always ensure true targets are evaluated during training
    candidate_ids.update(true_targets & set(target_records.keys()))

    is_val = s1_id in val_s1_set

    for tid in candidate_ids:
        tgt = target_records[tid]
        feats = compute_features(s1, tgt)
        lbl = 1 if tid in true_targets else 0

        if is_val:
            X_val.append(feats)
            y_val.append(lbl)
            val_pair_info.append((s1_id, tid))
        else:
            X_train.append(feats)
            y_train.append(lbl)

print(f"Training pairs: {len(X_train)} (Positives: {sum(y_train)}, Negatives: {len(y_train)-sum(y_train)})")
print(f"Validation pairs: {len(X_val)} (Positives: {sum(y_val)}, Negatives: {len(y_val)-sum(y_val)})")

print("--> [5/5] Training LightGBM Classifier...")
if HAS_LGB:
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        class_weight='balanced',
        random_state=42,
        verbosity=-1
    )
else:
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=42)

clf.fit(X_train, y_train)

# Predict probabilities on validation pairs
probs = clf.predict_proba(X_val)[:, 1]
val_preds_by_s1 = defaultdict(list)
for (s1_id, tid), prob in zip(val_pair_info, probs):
    val_preds_by_s1[s1_id].append((tid, prob))

# Metric calculation function
def evaluate_macro_f05(threshold):
    f_scores = []
    for s1_id in val_s1:
        true_set = gt_dict[s1_id]
        cands = [tid for (tid, p) in val_preds_by_s1.get(s1_id, []) if p >= threshold]
        pred_set = set(cands)

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

    return np.mean(f_scores)

print("\n--> Evaluating Macro F_0.5 across threshold grid:")
best_thresh = 0.5
best_score = -1.0
for th in np.linspace(0.40, 0.95, 23):
    score = evaluate_macro_f05(th)
    print(f"  Threshold {th:.2f}: Macro F_0.5 = {score:.4f}")
    if score > best_score:
        best_score = score
        best_thresh = th

print(f"\n=======================================================")
print(f"BEST VALIDATION MACRO F_0.5: {best_score:.4f} at threshold: {best_thresh:.2f}")
print(f"=======================================================")

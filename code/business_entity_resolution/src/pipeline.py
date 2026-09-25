#!/usr/bin/env python3
"""
Business Entity Resolution End-to-End Pipeline
================================================
Reproducible pipeline:
  1. Prefix-Enhanced Composite Blocking (WN, WW, P3N, AN, W) with IDF key weighting.
  2. Independent retrieval for Source 2 and Source 3 (Top 35 each).
  3. 25-feature pairwise feature extractor.
  4. LightGBM classifier with class_weight='balanced'.
  5. Calibrated decision threshold (tau = 0.9610) optimized for Macro F_0.5.
  6. Country-by-country stream inference for test set (US, India, France).
  7. Generates output/candidate_pairs.tsv and output/matching_results.tsv.
"""

import os
import re
import sys
import math
import time
import argparse
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import pickle

try:
    import anyascii
    def transliterate_text(text: str) -> str:
        if not text: return ""
        return anyascii.anyascii(text)
except ImportError:
    try:
        import unidecode
        def transliterate_text(text: str) -> str:
            if not text: return ""
            return unidecode.unidecode(text)
    except ImportError:
        def transliterate_text(text: str) -> str:
            if not text: return ""
            return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


# ==============================================================================
# 1. TEXT NORMALIZATION & PREPROCESSING PIPELINE
# ==============================================================================

LEGAL_SUFFIXES = {
    'pvt ltd', 'private limited', 'pvt', 'ltd', 'limited', 'inc', 'incorporated',
    'corp', 'corporation', 'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'holding', 'holdings', 'group', 'services', 'solutions', 'technologies', 'tech',
    'industries', 'international', 'intl', 'gmbh', 'sa', 'sarl', 'sas', 'bv', 'nv',
    'plc', 'spa', 'srl', 'sl', 'cia', 'assoc', 'associates',
    # Transliterated Indic / Regional legal forms
    'praivet limited', 'praaivett limittedd', 'privat limited', 'praivet', 'pra li',
    'praibhet limited', 'praibhet',
    'elelpi', 'limitet', 'piraivet limitet', 'piraiveett limittett'
}

INDIAN_GEO_MAP = {
    # Bengali
    'পশ্চিমবঙ্গ': 'west bengal', 'কলকাতা': 'kolkata', 'হাওড়া': 'howrah', 'শিলিगुড়ি': 'siliguri',
    # Hindi / Marathi
    'महाराष्ट्र': 'maharashtra', 'मध्य प्रदेश': 'madhya pradesh', 'उत्तर प्रदेश': 'uttar pradesh',
    'तमिलनाडु': 'tamil nadu', 'தமிழ்நாடு': 'tamil nadu', 'गुजरात': 'gujarat', 'ગુજરાત': 'gujarat',
    'कर्नाटक': 'karnataka', 'राजस्थान': 'rajasthan', 'आंध्र प्रदेश': 'andhra pradesh',
    'तेलंगाना': 'telangana', 'केरल': 'kerala', 'ओडिशा': 'odisha', 'उड़ीसा': 'odisha',
    'बिहार': 'bihar', 'पंजाब': 'punjab', 'हरियाणा': 'haryana', 'असम': 'assam',
    'दिल्ली': 'delhi', 'नई दिल्ली': 'new delhi', 'मुंबई': 'mumbai', 'बेंगलुरु': 'bengaluru',
    'बैंगलोर': 'bengaluru', 'हैदराबाद': 'hyderabad', 'पुणे': 'pune', 'भोपाल': 'bhopal',
    'अहमदाबाद': 'ahmedabad', 'वडोदरा': 'vadodara', 'सूरत': 'surat', 'जयपुर': 'jaipur'
}

TOKEN_NORMALIZATION_MAP = {
    # Common transliteration phonetic fixes to standard English loan words
    'adity': 'aditya',
    'proprtij': 'properties', 'proprti': 'property',
    'kmstrksms': 'constructions', 'kmstrkshn': 'construction',
    'marketimg': 'marketing',
    'teknolojij': 'technologies', 'teknoloji': 'technology',
    'solyushms': 'solutions', 'solyushn': 'solution',
    'emtrpraaaijez': 'enterprises', 'emtrpraaij': 'enterprise',
    'imddstrij': 'industries', 'imddstri': 'industry',
    'korporeshn': 'corporation',
    'praivet': 'private', 'praaivett': 'private',
    'limittedd': 'limited', 'limitet': 'limited',
    'elelpi': 'llp',
    'piraivet': 'private', 'piraiveett': 'private',
    'picins': 'business', 'picinnns': 'business',
    'shkti': 'shakti', 'skti': 'shakti',
    'arbn': 'urban',
    'prodkts': 'products', 'proddktts': 'products',
    'kulopl': 'global', 'kulloopl': 'global',
    # Bengali transliterated loan words
    'hspitaliti': 'hospitality',
    'praibhet': 'private',
    'knstraksn': 'constructions',
    'sarbhises': 'services', 'sarbhis': 'service',
    'markettim': 'marketing',
    'teknoljis': 'technologies', 'teknolji': 'technology',
    'saliushan': 'solution', 'saliushans': 'solutions',
    'emtarapraij': 'enterprise', 'emtarapraijes': 'enterprises'
}

ADDR_MAP = {
    'col': 'colony', 'clny': 'colony', 'colny': 'colony',
    'sec': 'sector', 'sect': 'sector',
    'soc': 'society', 'socty': 'society',
    'hsg': 'housing', 'hsng': 'housing',
    'ext': 'extension', 'extn': 'extension',
    'encl': 'enclave', 'enclv': 'enclave',
    'cmpd': 'compound', 'est': 'estate', 'indl': 'industrial',
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
    'ln': 'lane', 'dr': 'drive', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'cir': 'circle', 'cres': 'crescent', 'cl': 'close', 'rte': 'route',
    'hwy': 'highway', 'pkwy': 'parkway', 'expy': 'expressway', 'exp': 'expressway',
    'fl': 'floor', 'flr': 'floor', 'apt': 'apartment', 'apts': 'apartments',
    'ste': 'suite', 'bldg': 'building', 'bldng': 'building',
    'opp': 'opposite', 'oppo': 'opposite', 'nr': 'near', 'adj': 'adjacent',
    'bhnd': 'behind', 'b/h': 'behind',
    'dist': 'district', 'distt': 'district',
    'teh': 'tehsil', 'taluk': 'taluka', 'tal': 'taluka',
    'stn': 'station', 'nagar': 'nagar', 'marg': 'marg',
    'no': 'number', 'num': 'number',
    'shp': 'shop', 'plt': 'plot', 'blk': 'block',
    'mkt': 'market', 'rgcy': 'regency',
    'twr': 'tower', 'twrs': 'towers',
    'po': 'post office',
    'gf': 'ground floor', 'ff': 'first floor', 'sf': 'second floor', 'tf': 'third floor'
}

def normalize_text_pipeline(text: str, is_address: bool = False) -> str:
    """
    Full Normalization Pipeline:
      Original Data
        ↓ Canonical Geo entity mapping (e.g. পশ্চিমবঙ্গ -> west bengal)
        ↓ Unicode Normalization (NFKC)
        ↓ Transliteration Where Useful (anyascii/unidecode into Latin/ASCII)
        ↓ Case & Punctuation Normalization (lowercase, strip symbols)
        ↓ Token Normalization (adity -> aditya, proprtij -> properties)
        ↓ Address Normalization (col -> colony, rd -> road)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize('NFKC', text)
    # 1. Map known Indic geo words before transliteration
    for k, v in INDIAN_GEO_MAP.items():
        if k in text:
            text = text.replace(k, f" {v} ")
    # 2. Transliterate remaining non-ASCII
    if any(ord(c) >= 128 for c in text):
        text = transliterate_text(text)
    # 3. Case & Punctuation normalization
    text = text.lower()
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'[/\\_\-+,.:;!?\'"()\[\]{}|@#*^~`]', ' ', text)
    # 4. Token-level mapping
    tokens = text.split()
    tokens = [TOKEN_NORMALIZATION_MAP.get(t, t) for t in tokens]
    if is_address:
        tokens = [ADDR_MAP.get(t, t) for t in tokens]
    return ' '.join(tokens)

def clean_name_and_core(name: str) -> Tuple[str, str]:
    """
    Produces multiple representations:
      - Full normalized name
      - Core name with legal suffixes removed
    """
    cleaned = normalize_text_pipeline(name, is_address=False)
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
    """
    Applies address abbreviation normalization (ADDR_MAP) on normalized text.
    """
    return normalize_text_pipeline(addr, is_address=True)

def extract_clean_numbers(addr: str) -> List[str]:
    res = []
    for n in re.findall(r'\d+', addr):
        try:
            norm = str(int(n))
            if 1 <= len(norm) <= 7:
                res.append(norm)
        except ValueError:
            pass
    return res


# ==============================================================================
# 2. STRING SIMILARITY & 25-FEATURE VECTOR
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
# 3. COMPOSITE BLOCKING KEYS
# ==============================================================================

def get_composite_keys(core_name: str, clean_addr: str, nums: List[str]) -> List[str]:
    name_words = core_name.split()
    addr_words = [t for t in clean_addr.split() if t not in {'street', 'st', 'road', 'rd', 'ave', 'avenue', 'suite', 'unit', 'dr', 'drive'}]

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

    # 4. Standalone W
    for nw in name_words[:4]:
        keys.append(f"W:{nw}")

    return keys


# ==============================================================================
# 4. INFERENCE ENGINE (COUNTRY-BY-COUNTRY STREAMING)
# ==============================================================================

def run_pipeline(data_dir: str, output_dir: str, model_file: str = None):
    print("="*75)
    print("   BUSINESS ENTITY RESOLUTION REPRODUCIBLE PIPELINE")
    print("="*75)
    sys.stdout.flush()

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Train or Load Model
    feature_cols = [
        'feat_name_exact', 'feat_core_exact', 'feat_name_lev', 'feat_core_lev',
        'feat_name_jw', 'feat_core_jw', 'feat_name_token_jaccard', 'feat_core_token_jaccard',
        'feat_name_containment', 'feat_name_char3_jaccard', 'feat_first_word_match',
        'feat_first_char_match', 'len_name_diff', 'feat_addr_exact', 'feat_addr_lev',
        'feat_addr_jw', 'feat_addr_token_jaccard', 'feat_addr_containment',
        'feat_addr_char3_jaccard', 'digit_overlap', 'digit_jaccard', 'has_common_digit',
        'blocking_score', 'is_source2'
    ]

    model_path_txt = os.path.join(output_dir, "model_lgb.txt")
    model_path_pkl = os.path.join(output_dir, "model.pkl")
    optimal_threshold = 0.9610
    predict_fn = None

    if model_file and os.path.exists(model_file):
        if model_file.endswith(".pkl"):
            with open(model_file, "rb") as f:
                model_obj = pickle.load(f)
            predict_fn = lambda x: model_obj.predict_proba(x)[:, 1]
        else:
            booster = lgb.Booster(model_file=model_file)
            predict_fn = lambda x: booster.predict(x)
        print(f"Loaded pre-trained model from {model_file}!")
    elif os.path.exists(model_path_pkl):
        with open(model_path_pkl, "rb") as f:
            model_obj = pickle.load(f)
        predict_fn = lambda x: model_obj.predict_proba(x)[:, 1]
        print(f"Loaded pre-trained model from {model_path_pkl}!")
    elif os.path.exists(model_path_txt) and HAS_LGB:
        booster = lgb.Booster(model_file=model_path_txt)
        predict_fn = lambda x: booster.predict(x)
        print(f"Loaded pre-trained model from {model_path_txt}!")
    else:
        print("Training LightGBM model on training data...")
        # Train on 15,000 entities
        gt_map = {}
        with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 2 and p[1].strip():
                    gt_map[p[0]] = set(p[1].split(','))
                if len(gt_map) >= 15000:
                    break

        all_needed_s1 = set(gt_map.keys())
        s1_train_dict = {}
        with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if p[0] in all_needed_s1:
                    clean_n, core_n = clean_name_and_core(p[1])
                    clean_a = clean_address(p[2])
                    s1_train_dict[p[0]] = {
                        'id': p[0], 'clean_name': clean_n, 'core_name': core_n,
                        'clean_addr': clean_a, 'nums': extract_clean_numbers(clean_a),
                        'country': p[3].strip().lower()
                    }

        # Index S2 and S3 for training
        def index_file(fn, prefix):
            idx = defaultdict(list)
            cnt = 0
            with open(os.path.join(train_dir, fn), "r", encoding="utf-8") as f:
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
            N = float(cnt)
            wts = {}
            for k, id_list in idx.items():
                df = len(id_list)
                if df > 5000: continue
                idf = math.log((N + 1.0) / (df + 1.0))
                wts[k] = idf * 2.5 if (k[1].startswith("WN:") or k[1].startswith("WW:")) else idf
            return idx, wts

        s2_idx, s2_wts = index_file("train_source2.tsv", "S2")
        s3_idx, s3_wts = index_file("train_source3.tsv", "S3")

        def query_b(s1, idx, wts, prefix):
            c = s1['country']
            keys = get_composite_keys(s1['core_name'], s1['clean_addr'], s1['nums'])
            scores = defaultdict(float)
            for k_str in keys:
                lk = (c, k_str)
                if lk in wts:
                    w = wts[lk]
                    for int_id in idx[lk]:
                        scores[int_id] += w
            return [(f"{prefix}-{x[0]}", x[1]) for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:35]]

        train_pairs_map = {}
        all_tgt_needed = set()
        for s1_id, s1 in s1_train_dict.items():
            cands = query_b(s1, s2_idx, s2_wts, "S2") + query_b(s1, s3_idx, s3_wts, "S3")
            train_pairs_map[s1_id] = cands
            for cid, _ in cands: all_tgt_needed.add(cid)

        del s2_idx, s2_wts, s3_idx, s3_wts

        # Load needed records
        tgt_cache = {}
        for fn in ["train_source2.tsv", "train_source3.tsv"]:
            with open(os.path.join(train_dir, fn), "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 4 and p[0] in all_tgt_needed:
                        clean_n, core_n = clean_name_and_core(p[1])
                        clean_a = clean_address(p[2])
                        tgt_cache[p[0]] = {
                            'id': p[0], 'clean_name': clean_n, 'core_name': core_n,
                            'clean_addr': clean_a, 'nums': extract_clean_numbers(clean_a),
                            'country': p[3].strip().lower()
                        }

        X_tr, y_tr = [], []
        for s1_id, cands in train_pairs_map.items():
            s1 = s1_train_dict[s1_id]
            tset = gt_map[s1_id]
            for cid, bsc in cands:
                if cid in tgt_cache:
                    X_tr.append(extract_pair_features(s1, tgt_cache[cid], bsc))
                    y_tr.append(1 if cid in tset else 0)

        df_tr = pd.DataFrame(X_tr)
        if HAS_LGB:
            clf = lgb.LGBMClassifier(
                n_estimators=350, learning_rate=0.04, num_leaves=31, max_depth=6,
                subsample=0.85, colsample_bytree=0.85, class_weight='balanced', random_state=42, verbosity=-1
            )
            clf.fit(df_tr[feature_cols], np.array(y_tr))
            clf.booster_.save_model(model_path_txt)
            predict_fn = lambda x: clf.booster_.predict(x)
            print(f"Model saved to {model_path_txt}!")
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=350, learning_rate=0.04, class_weight='balanced', random_state=42
            )
            clf.fit(df_tr[feature_cols], np.array(y_tr))
            with open(model_path_pkl, "wb") as f:
                pickle.dump(clf, f)
            predict_fn = lambda x: clf.predict_proba(x)[:, 1]
            print(f"Model saved to {model_path_pkl}!")

    # 2. Test Set Inference (Country-by-Country Processing)
    print("\n--> [Inference] Processing Test Set Country by Country...")
    sys.stdout.flush()

    # Discover test countries and S1 IDs
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_countries = defaultdict(list)
    ordered_test_s1_ids = []

    with open(test_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                ordered_test_s1_ids.append(p[0])
                c = p[3].strip().lower()
                clean_n, core_n = clean_name_and_core(p[1])
                clean_a = clean_address(p[2])
                test_countries[c].append({
                    'id': p[0], 'clean_name': clean_n, 'core_name': core_n,
                    'clean_addr': clean_a, 'nums': extract_clean_numbers(clean_a),
                    'country': c
                })

    print(f"Total Test Entities: {len(ordered_test_s1_ids):,}")
    for c, lst in test_countries.items():
        print(f"  Country '{c}': {len(lst):,} entities")
    sys.stdout.flush()

    candidate_results: Dict[str, List[str]] = {}
    matching_results: Dict[str, List[str]] = {}

    for country, s1_records in test_countries.items():
        print(f"\nProcessing Country: [{country.upper()}] ({len(s1_records):,} entities)...")
        sys.stdout.flush()

        # Build target indexes for this country only (RAM efficient)
        def index_country_targets(filename, prefix):
            t0 = time.time()
            idx = defaultdict(list)
            records = {}
            cnt = 0
            with open(os.path.join(test_dir, filename), "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 4 and p[3].strip().lower() == country:
                        eid = p[0]
                        int_id = int(eid[3:])
                        clean_n, core_n = clean_name_and_core(p[1])
                        clean_a = clean_address(p[2])
                        nums = extract_clean_numbers(clean_a)

                        records[eid] = {
                            'id': eid, 'clean_name': clean_n, 'core_name': core_n,
                            'clean_addr': clean_a, 'nums': nums, 'country': country
                        }
                        for k in get_composite_keys(core_n, clean_a, nums):
                            idx[k].append(int_id)
                        cnt += 1
            N = float(cnt)
            wts = {}
            for k, id_list in idx.items():
                df = len(id_list)
                if df > 5000: continue
                idf = math.log((N + 1.0) / (df + 1.0))
                wts[k] = idf * 2.5 if (k.startswith("WN:") or k.startswith("WW:")) else idf
            print(f"  Indexed {cnt:,} {prefix} targets in {time.time() - t0:.2f}s.")
            sys.stdout.flush()
            return records, idx, wts

        s2_recs, s2_idx, s2_wts = index_country_targets("test_source2.tsv", "S2")
        s3_recs, s3_idx, s3_wts = index_country_targets("test_source3.tsv", "S3")
        target_pool = {**s2_recs, **s3_recs}

        def query_cands(s1, idx, wts, prefix):
            keys = get_composite_keys(s1['core_name'], s1['clean_addr'], s1['nums'])
            scores = defaultdict(float)
            for k_str in keys:
                if k_str in wts:
                    w = wts[k_str]
                    for int_id in idx[k_str]:
                        scores[int_id] += w
            return [(f"{prefix}-{x[0]}", x[1]) for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:35]]

        t_start = time.time()
        for idx_e, s1 in enumerate(s1_records):
            s1_id = s1['id']
            cands_s2 = query_cands(s1, s2_idx, s2_wts, "S2")
            cands_s3 = query_cands(s1, s3_idx, s3_wts, "S3")
            combined_cands = cands_s2 + cands_s3

            cand_ids = [c[0] for c in combined_cands]
            candidate_results[s1_id] = cand_ids

            # Score with LightGBM
            matched_ids = []
            if combined_cands:
                feats = [extract_pair_features(s1, target_pool[cid], bsc) for cid, bsc in combined_cands if cid in target_pool]
                if feats:
                    df_feats = pd.DataFrame(feats)
                    probs = predict_fn(df_feats[feature_cols].values)
                    for (cid, _), pr in zip(combined_cands, probs):
                        if pr >= optimal_threshold:
                            matched_ids.append(cid)

            # Ensure matches are strictly a subset of candidates and deduplicated
            seen = set()
            clean_matches = []
            for mid in matched_ids:
                if mid not in seen and mid in cand_ids:
                    seen.add(mid)
                    clean_matches.append(mid)

            matching_results[s1_id] = clean_matches

            if (idx_e + 1) % 50000 == 0:
                print(f"  Processed {idx_e + 1:,} / {len(s1_records):,} entities ({idx_e/(time.time() - t_start):.0f} ent/s)...")
                sys.stdout.flush()

        del s2_recs, s2_idx, s2_wts, s3_recs, s3_idx, s3_wts, target_pool

    # 3. Write Output TSV Files
    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"\nWriting {cand_path}...")
    sys.stdout.flush()
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ordered_test_s1_ids:
            cands_str = ",".join(candidate_results.get(s1_id, []))
            f.write(f"{s1_id}\t{cands_str}\n")

    print(f"Writing {match_path}...")
    sys.stdout.flush()
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_test_s1_ids:
            matches_str = ",".join(matching_results.get(s1_id, []))
            f.write(f"{s1_id}\t{matches_str}\n")

    print("\n" + "="*70)
    print("INFERENCE GENERATION COMPLETE!")
    print(f"  Candidate file: {cand_path}")
    print(f"  Matching file:  {match_path}")
    print(f"  Total test entities: {len(ordered_test_s1_ids):,}")
    total_preds = sum(len(m) for m in matching_results.values())
    total_singletons = sum(1 for m in matching_results.values() if len(m) == 0)
    print(f"  Total matches predicted: {total_preds:,}")
    print(f"  Singletons (no match):   {total_singletons:,} ({total_singletons/len(ordered_test_s1_ids)*100:.2f}%)")
    print("="*70)
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", type=str, default=r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset")
    parser.add_argument("--output-dir", type=str, default=r"c:\Users\Shourya\Desktop\dataset\output")
    parser.add_argument("--model-file", type=str, default=r"c:\Users\Shourya\Desktop\dataset\trained_model.txt")
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.output_dir, args.model_file)

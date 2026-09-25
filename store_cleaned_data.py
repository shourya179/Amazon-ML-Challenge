#!/usr/bin/env python3
"""
Clean Dataset Generator & Persister
====================================
Processes raw dataset files through the complete normalization pipeline:
  Original Data
    ↓ Unicode Normalization (NFKC)
    ↓ Transliteration Where Useful (anyascii)
    ↓ Case / Punctuation Normalization
    ↓ Remove Legal Suffixes (core_name)
    ↓ Address Abbreviation Normalization (ADDR_MAP)
    ↓ Numeric / PIN Code Extraction (address_numbers)

Persists the cleaned records to TSV in `cleaned_data/`.
"""

import os
import re
import sys
import time
import argparse
import unicodedata
from typing import Tuple, List

try:
    import anyascii
    def transliterate_text(text: str) -> str:
        return anyascii.anyascii(text) if text else ""
except ImportError:
    try:
        import unidecode
        def transliterate_text(text: str) -> str:
            return unidecode.unidecode(text) if text else ""
    except ImportError:
        def transliterate_text(text: str) -> str:
            return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8') if text else ""

LEGAL_SUFFIXES = {
    'pvt ltd', 'private limited', 'pvt', 'ltd', 'limited', 'inc', 'incorporated',
    'corp', 'corporation', 'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'holding', 'holdings', 'group', 'services', 'solutions', 'technologies', 'tech',
    'industries', 'international', 'intl', 'gmbh', 'sa', 'sarl', 'sas', 'bv', 'nv',
    'plc', 'spa', 'srl', 'sl', 'cia', 'assoc', 'associates',
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
    # 3. Lowercase & punctuation cleaning
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

def clean_and_store_file(input_path: str, output_path: str, limit: int = None):
    print(f"\n[Processing] {os.path.basename(input_path)} -> {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t0 = time.time()
    count = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8', buffering=64*1024*1024) as fout:
        
        # Write purely clean header
        header = fin.readline()
        fout.write("entity_id\tclean_name\tcore_name\tclean_address\taddress_numbers\tcountry\n")

        for line in fin:
            p = line.rstrip('\r\n').split('\t')
            if len(p) >= 4:
                eid, raw_name, raw_addr, country = p[0], p[1], p[2], p[3].strip()
                c_name, core_name = clean_name_and_core(raw_name)
                c_addr = clean_address(raw_addr)
                nums = " ".join(extract_clean_numbers(c_addr))
                
                fout.write(f"{eid}\t{c_name}\t{core_name}\t{c_addr}\t{nums}\t{country}\n")
                count += 1

                if count % 250000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Processed {count:,} records ({count/elapsed:.0f} rec/s)...")
                    sys.stdout.flush()

                if limit and count >= limit:
                    break

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Done] Stored {count:,} clean records in {output_path} ({size_mb:.1f} MB, {elapsed:.1f}s, {count/elapsed:.0f} rec/s)")

def main():
    parser = argparse.ArgumentParser(description="Clean and store dataset records to disk.")
    parser.add_argument("--base-dir", default=r"c:\Users\Shourya\Desktop\dataset", help="Base dataset directory")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train", help="Which split to clean")
    parser.add_argument("--sources", nargs="+", default=["source1", "source2", "source3"], help="Sources to process")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per file (for testing)")
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    
    for split in splits:
        in_dir = os.path.join(args.base_dir, "student_resource", "dataset", split)
        out_dir = os.path.join(args.base_dir, "cleaned_data", split)

        for src in args.sources:
            in_file = os.path.join(in_dir, f"{split}_{src}.tsv")
            out_file = os.path.join(out_dir, f"clean_{split}_{src}.tsv")
            if os.path.exists(in_file):
                clean_and_store_file(in_file, out_file, limit=args.limit)
            else:
                print(f"[Warning] File not found: {in_file}")

if __name__ == "__main__":
    main()

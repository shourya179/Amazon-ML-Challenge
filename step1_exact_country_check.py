import os
import sys

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("="*60)
print("STEP 1: RIGOROUS FULL-DATASET EXACT STRING COUNTRY CHECK")
print("="*60)
sys.stdout.flush()

id_to_country = {}

def load_source_countries(filename, label):
    path = os.path.join(train_dir, filename)
    print(f"Loading exact country strings from {label} ({filename})...")
    sys.stdout.flush()
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        f.readline() # header
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                # Store exact normalized string: e.g. 'us', 'india', 'france'
                id_to_country[p[0]] = p[3].strip().lower()
                count += 1
    print(f"  Loaded {count:,} entities from {label}.")
    sys.stdout.flush()

load_source_countries("train_source1.tsv", "Source 1")
load_source_countries("train_source2.tsv", "Source 2")
load_source_countries("train_source3.tsv", "Source 3")

total_entities = len(id_to_country)
print(f"\nTotal entities mapped: {total_entities:,}")
unique_countries = set(id_to_country.values())
print(f"Unique country labels found in training: {unique_countries}")
sys.stdout.flush()

print("\nChecking ALL 7.6M+ ground truth pairs for exact string country equality...")
sys.stdout.flush()

total_checked_pairs = 0
cross_country_mismatches = 0
missing_s1_ids = 0
missing_tgt_ids = 0
mismatch_examples = []

with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline() # header
    for line_idx, line in enumerate(f):
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            s1_id = p[0]
            s1_c = id_to_country.get(s1_id)
            if s1_c is None:
                missing_s1_ids += 1
                continue

            for mid in p[1].split(','):
                mid = mid.strip()
                if not mid: continue
                tgt_c = id_to_country.get(mid)
                if tgt_c is None:
                    missing_tgt_ids += 1
                    continue

                total_checked_pairs += 1
                # STRICT EXACT STRING EQUALITY CHECK
                if s1_c != tgt_c:
                    cross_country_mismatches += 1
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append((s1_id, s1_c, mid, tgt_c))

        if (line_idx + 1) % 500000 == 0:
            print(f"  Processed {line_idx + 1:,} S1 entities... Checked {total_checked_pairs:,} true pairs.")
            sys.stdout.flush()

print("\n" + "="*60)
print("FINAL EXACT COUNTRY CHECK RESULTS:")
print(f"  Total true match pairs checked: {total_checked_pairs:,}")
print(f"  Cross-country mismatches found: {cross_country_mismatches:,}")
print(f"  Missing S1 IDs: {missing_s1_ids:,}")
print(f"  Missing Target IDs: {missing_tgt_ids:,}")
if cross_country_mismatches == 0:
    print("  VERDICT: PERFECT 100.000% COUNTRY CONSERVATION PROVEN.")
    print("  Country partitioning is 100% sound across the entire dataset.")
else:
    print(f"  VERDICT: FOUND {cross_country_mismatches} MISMATCHES!")
    for s1_id, s1_c, mid, tgt_c in mismatch_examples:
        print(f"    Mismatch: {s1_id} ({s1_c}) != {mid} ({tgt_c})")
print("="*60)
sys.stdout.flush()

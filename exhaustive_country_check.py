import os
import sys

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("Step 1: Mapping ID -> country for all records...")
# 0 = US, 1 = India, 2 = Other
c_map = {'us': 0, 'india': 1}

def get_country_code(s):
    return c_map.get(s.strip().lower(), 2)

# Load S1 countries
print("Loading S1 countries...")
id_to_country = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4:
            id_to_country[p[0]] = get_country_code(p[3])

print(f"Loaded {len(id_to_country)} S1 countries.")

print("Loading S2 countries...")
with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4:
            id_to_country[p[0]] = get_country_code(p[3])

print("Loading S3 countries...")
with open(os.path.join(train_dir, "train_source3.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4:
            id_to_country[p[0]] = get_country_code(p[3])

print(f"Total mapped entities: {len(id_to_country)}")

print("Step 2: Checking ALL true matches in train_ground_truth.tsv...")
total_checked = 0
cross_country_matches = 0
missing_id = 0

with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            s1_id = p[0]
            s1_c = id_to_country.get(s1_id)
            if s1_c is None:
                missing_id += 1
                continue
            for mid in p[1].split(','):
                mid = mid.strip()
                if not mid: continue
                tgt_c = id_to_country.get(mid)
                if tgt_c is None:
                    missing_id += 1
                    continue
                total_checked += 1
                if s1_c != tgt_c:
                    cross_country_matches += 1

print(f"\n==================================================")
print(f"Comprehensive Full-Dataset Check Results:")
print(f"  Total true match pairs checked: {total_checked}")
print(f"  Cross-country matches found: {cross_country_matches}")
print(f"  Missing IDs: {missing_id}")
print(f"==================================================")

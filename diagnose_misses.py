import os
import re

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

# Load 500 US S1 entities and their S2 targets
val_gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            s2 = [x for x in p[1].split(',') if x.startswith('S2-')]
            if s2:
                val_gt[p[0]] = s2
        if len(val_gt) >= 500:
            break

s1_recs = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in val_gt and p[3].strip().lower() == 'us':
            s1_recs[p[0]] = p

all_s2_needed = {x for s1 in s1_recs for x in val_gt[s1]}
s2_recs = {}
with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in all_s2_needed:
            s2_recs[p[0]] = p
        if len(s2_recs) == len(all_s2_needed):
            break

print(f"Loaded {len(s1_recs)} S1 records and {len(s2_recs)} S2 true records.")

def clean_toks(s):
    return set(re.sub(r'[^a-zA-Z0-9]', ' ', s.lower()).split())

# Check how S1 and S2 overlap in these true pairs
name_match_count = 0
addr_match_count = 0
num_match_count = 0
no_match_count = 0
total_pairs = 0

missed_examples = []

for s1_id, s1_p in s1_recs.items():
    s1_name_toks = clean_toks(s1_p[1])
    s1_addr_toks = clean_toks(s1_p[2])
    s1_nums = set(re.findall(r'\b\d+\b', s1_p[2]))

    for s2_id in val_gt[s1_id]:
        if s2_id not in s2_recs: continue
        total_pairs += 1
        s2_p = s2_recs[s2_id]
        s2_name_toks = clean_toks(s2_p[1])
        s2_addr_toks = clean_toks(s2_p[2])
        s2_nums = set(re.findall(r'\b\d+\b', s2_p[2]))

        has_n = len(s1_name_toks & s2_name_toks) > 0
        has_a = len(s1_addr_toks & s2_addr_toks) > 0
        has_num = len(s1_nums & s2_nums) > 0

        if has_n: name_match_count += 1
        if has_a: addr_match_count += 1
        if has_num: num_match_count += 1

        if not has_n and not has_a and not has_num:
            no_match_count += 1
            if len(missed_examples) < 5:
                missed_examples.append((s1_p, s2_p))

print(f"Total US True Pairs: {total_pairs}")
print(f"  Exact name token overlap: {name_match_count} ({name_match_count/total_pairs*100:.2f}%)")
print(f"  Exact addr token overlap: {addr_match_count} ({addr_match_count/total_pairs*100:.2f}%)")
print(f"  Exact number overlap: {num_match_count} ({num_match_count/total_pairs*100:.2f}%)")
print(f"  NO EXACT OVERLAP WHATSOEVER: {no_match_count} ({no_match_count/total_pairs*100:.2f}%)")

for s1, s2 in missed_examples:
    print("\n[ZERO EXACT OVERLAP EXAMPLE]")
    print(f"  S1: {s1[1]} | {s1[2]}")
    print(f"  S2: {s2[1]} | {s2[2]}")

import os
import re

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

# Load 100 S1 records and their S2 matches from GT
gt = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1].strip():
            gt[p[0]] = [x for x in p[1].split(',') if x.startswith('S2-')]
        if len(gt) >= 100:
            break

s1_recs = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in gt:
            s1_recs[p[0]] = p
        if len(s1_recs) == len(gt):
            break

needed_s2 = {mid for mlist in gt.values() for mid in mlist}
s2_recs = {}
with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if p[0] in needed_s2:
            s2_recs[p[0]] = p
        if len(s2_recs) == len(needed_s2):
            break

print(f"Found {len(s2_recs)} S2 records in file.")

# Let's inspect 10 pairs
count = 0
for s1_id, mlist in gt.items():
    if s1_id not in s1_recs: continue
    s1 = s1_recs[s1_id]
    for mid in mlist:
        if mid in s2_recs:
            s2 = s2_recs[mid]
            count += 1
            print(f"\n--- PAIR #{count} ---")
            print(f"S1: Name: '{s1[1]}' | Addr: '{s1[2]}' | Country: '{s1[3]}'")
            print(f"S2: Name: '{s2[1]}' | Addr: '{s2[2]}' | Country: '{s2[3]}'")
            if count >= 10:
                break
    if count >= 10:
        break

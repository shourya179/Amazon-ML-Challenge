import os
from collections import Counter

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"
gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

match_dist = Counter()
total_entities = 0
total_matches = 0
with open(gt_path, "r", encoding="utf-8") as f:
    f.readline() # header
    for line in f:
        total_entities += 1
        parts = line.strip().split('\t')
        if len(parts) > 1 and parts[1].strip():
            m_list = parts[1].split(',')
            cnt = len(m_list)
            match_dist[cnt] += 1
            total_matches += cnt
        else:
            match_dist[0] += 1

print(f"Total S1 entities in train: {total_entities}")
print(f"Total true matches in train: {total_matches}")
print(f"Average matches per entity: {total_matches / total_entities:.2f}")
print("Match count distribution:")
for k in sorted(match_dist.keys())[:12]:
    print(f"  {k} matches: {match_dist[k]} ({match_dist[k]/total_entities*100:.2f}%)")

import os
from collections import Counter

test_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\test"
s1_path = os.path.join(test_dir, "test_source1.tsv")

counts = Counter()
with open(s1_path, "r", encoding="utf-8") as f:
    header = f.readline().strip().split('\t')
    country_idx = header.index('country') if 'country' in header else 3
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > country_idx:
            counts[parts[country_idx]] += 1

print("Test Source 1 Country breakdown:")
for c, cnt in counts.items():
    print(f"  {c}: {cnt}")

import os
from collections import Counter

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"
for fn in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    fp = os.path.join(train_dir, fn)
    c_counts = Counter()
    with open(fp, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                c_counts[parts[3]] += 1
    print(f"{fn} country distribution: {dict(c_counts)}")

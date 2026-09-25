import os

train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"
for fn in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]:
    fp = os.path.join(train_dir, fn)
    with open(fp, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"{fn}: {count} lines")

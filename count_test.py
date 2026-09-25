import pandas as pd
import os

test_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\test"
for fn in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
    fp = os.path.join(test_dir, fn)
    # count lines
    with open(fp, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"{fn}: {count} lines")

import pandas as pd
import os

data_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

print("Reading sample from train_ground_truth.tsv...")
df_gt_sample = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", nrows=10)
print(df_gt_sample)

sample_s1_id = df_gt_sample.iloc[0]['source1_entity_id']
sample_matches = df_gt_sample.iloc[0]['matched_entity_ids'].split(',')
print(f"\nLooking up S1 ID: {sample_s1_id} and matches: {sample_matches}")

print("\nReading sample from train_source1.tsv...")
df_s1_sample = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", nrows=10)
print(df_s1_sample)

print("\nChecking matching record in S1...")
# Read in chunks to find sample_s1_id
for chunk in pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", chunksize=100000):
    match = chunk[chunk['entity_id'] == sample_s1_id]
    if not match.empty:
        print("Found in S1:")
        print(match)
        break

print("\nChecking matching record in S2...")
for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", chunksize=100000):
    match = chunk[chunk['entity_id'] == sample_matches[0]]
    if not match.empty:
        print("Found in S2:")
        print(match)
        break

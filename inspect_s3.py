import pandas as pd
import os

data_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"
target_s3_ids = {'S3-775321672', 'S3-11291185', 'S3-860443364'}

print("Looking for S3 matches...")
found = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", chunksize=200000):
    m = chunk[chunk['entity_id'].isin(target_s3_ids)]
    for _, row in m.iterrows():
        found[row['entity_id']] = row.to_dict()
    if len(found) == len(target_s3_ids):
        break

for sid, d in found.items():
    print(f"S3 {sid}: Name='{d.get('business_name')}', Addr='{d.get('business_address')}', Country='{d.get('country')}'")

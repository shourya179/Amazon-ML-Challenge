# Business Entity Resolution Pipeline

This repository contains the end-to-end, leak-free Machine Learning pipeline for the **Business Entity Resolution Challenge**.
The solution identifies matching business entity records across three noisy independent data sources (Source 1 reference, Source 2, and Source 3), optimized specifically for the precision-weighted **Macro $F_{0.5}$** metric.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Source Data (S1, S2, S3)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 1. Text Normalization       |
                      |    - Legal Suffix Stripping |
                      |    - Address Standardization|
                      |    - Numeric/PIN Extraction |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 2. Prefix-Composite Blocker |
                      |    - 100% Country Isolation |
                      |    - WN, WW, P3N, AN, W Keys|
                      |    - Dynamic IDF Weighting  |
                      |    - Top 35 S2 + Top 35 S3  |
                      +--------------+--------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
+-------------------------------+      +-------------------------------+
| output/candidate_pairs.tsv    |      | 3. Feature Extraction         |
| (Final candidate pairs)       |      |    - Levenshtein & JaroWinkler|
+-------------------------------+      |    - Token Sort & Jaccard     |
                                       |    - Postal / Digit Overlap   |
                                       +---------------+---------------+
                                                       |
                                                       v
                                       +-------------------------------+
                                       | 4. Supervised Matcher         |
                                       |    - LightGBM (Class-Balanced)|
                                       |    - Calibrated Threshold     |
                                       |      (tau = 0.9610)           |
                                       +---------------+---------------+
                                                       |
                                                       v
                                       +-------------------------------+
                                       | output/matching_results.tsv   |
                                       | (Leaderboard Submission)      |
                                       +-------------------------------+
```

---

## Key Benchmark Metrics (Evaluated on Full Target Populations)

- **Blind Candidate Recall Ceiling:** **83.63%** (8,750 / 10,463) evaluated against all 10,320,219 records of S2 & S3.
- **Average Candidate Pool Size:** **69.7** pairs per Source 1 entity.
- **Validation Macro $F_{0.5}$ Score:** **0.8722** (evaluated on 3,000 held-out validation entities).
- **Calibrated Decision Threshold:** **$\tau = 0.9610$**.

---

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   └── pipeline.py         # End-to-end pipeline script
├── README.md               # Reproduction guide
└── requirements.txt        # Pinned dependencies
```

---

## Setup & Reproduction Instructions

### 1. Environment Setup
Install required dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Running End-to-End Pipeline
Run the pipeline pointing to the dataset root (containing `train/` and `test/` subfolders):
```bash
python code/business_entity_resolution/src/pipeline.py \
  --data-dir student_resource/dataset \
  --output-dir output
```

### 3. Generated Outputs
The pipeline automatically writes two compliant tab-separated files in `output/`:
1. `output/candidate_pairs.tsv`: Final candidate set from blocking passed into ML scoring.
2. `output/matching_results.tsv`: Final predicted entity matches thresholded at $\tau = 0.9610$.

### 4. Format Validation
Validate compliance locally using the official competition submission validator:
```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```
Exit code `0` confirms the files are valid and ready for leaderboard submission.

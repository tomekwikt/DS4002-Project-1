# DS4002-Project-1

Career Matcher Project for DS 4002

This project uses resume text and personal interests to recommend potential career paths. It combines occupational descriptions, skills, and task statements from the O*NET 31.0 Database into career profiles, then uses Sentence-BERT embeddings and cosine similarity to compare those profiles with user profiles.

The current analysis compares 20 sample user profiles against 1,016 careers and returns five recommendations per user. The current pipeline produces similarity rankings and does not yet calculate accuracy against labeled expected careers.

## Repository Contents

- **DATA:** Original datasets, processed profiles, embeddings, supporting model information, and metadata documentation.
- **SCRIPTS:** Five numbered Python scripts for preparing profiles, generating embeddings, and ranking career matches.
- **OUTPUT:** Career recommendations, a readable results summary, and explanatory plots.
- **TESTS:** Automated checks for profile preparation, user embeddings, and career matching.
- **requirements.txt:** Python package versions used by the project.
- **LICENSE:** Project license and O*NET attribution.

## Section 1: Software and Platform

The current pipeline uses **Python 3.12 on Windows** and runs through Python scripts. Earlier data exploration and metadata preparation used Google Colab. The current analysis can be reproduced without a Colab notebook.

The required Python packages are listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| NumPy | Store embedding arrays and calculate numerical similarities |
| sentence-transformers | Generate Sentence-BERT embeddings |
| PyTorch | Run the pretrained neural network |
| Transformers | Load the tokenizer and supporting model components |

The model is **`sentence-transformers/all-MiniLM-L6-v2`**`. It produces **384-dimensional embeddings**. The preparation scripts limit selected profiles to **250 tokens, including special tokens**, within the model’s 256-token limit.

Both career and user embeddings are normalized. Their dot product therefore represents cosine similarity. Higher scores indicate more similar text; they are not probabilities of career suitability.

Internet access is needed to install dependencies and download the pretrained model on the first run.

## Section 2: Map of Documentation

```text
DS4002-Project-1/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── DATA/
│   ├── Career Profile Files/
│   │   ├── occupation_data.csv
│   │   ├── essential_skills.csv
│   │   ├── task_statements.csv
│   │   ├── processed_career_profiles.csv
│   │   └── career_metadata.csv
│   ├── Example User Profiles/
│   │   └── career_match_20_resumes_with_generated_interests.csv
│   ├── metadata/
│   │   └── README.md
│   ├── processed_user_profiles.csv
│   ├── career_embeddings.npy
│   ├── career_embedding_stats.json
│   ├── user_embeddings.npy
│   ├── user_embedding_stats.json
│   └── user_metadata.csv
├── SCRIPTS/
│   ├── 01_prepare_career_profiles.py
│   ├── 02_create_career_embeddings.py
│   ├── 03_prepare_user_profiles.py
│   ├── 04_create_user_embeddings.py
│   └── 05_match_careers.py
├── OUTPUT/
│   ├── career_matches.csv
│   ├── career_matches_summary.txt
│   └── metadata/
│       ├── occupation_coverage.png
│       └── skill_importance_distribution.png
└── TESTS/
    ├── test_career_profiles.py
    ├── test_user_profiles.py
    ├── test_user_embeddings.py
    └── test_career_matches.py
```

The [metadata README](DATA/metadata/README.md) provides the data summary, provenance, license information, ethical statements, data dictionary, and two explanatory plots.

The `.npy` files contain numerical embeddings. The accompanying metadata CSVs identify the career or user represented by each embedding row. The embedding JSON files record model information and processing statistics; the matching script uses them to verify that both sets of embeddings are compatible.

The local `.venv` environment and Git’s internal files are excluded from the map. Running script 03 also creates `DATA/processed_user_profiles.stats.json`, a summary of user-profile preparation.

## Section 3: Instructions for Reproducing Results

### Step 1: Download the project and install dependencies

Clone the repository or download and extract its ZIP file. Install Python 3.12, then open PowerShell in the project’s main folder, which contains `README.md` and `requirements.txt`.

Create a virtual environment and install the required packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run all remaining commands from this same project folder. The commands below rebuild generated data and overwrite existing results.

### Step 2: Prepare career profiles

```powershell
.\.venv\Scripts\python.exe SCRIPTS/01_prepare_career_profiles.py --data-dir "DATA/Career Profile Files"
```

This script joins the occupation, skill, and task datasets using `O*NET-SOC Code`. It cleans text, ranks skills by Importance, and selects skills and complete task statements within the 250-token preparation limit.

It retains all 1,016 occupations, including those without skill or task records, and saves:

```text
DATA/Career Profile Files/processed_career_profiles.csv
```

### Step 3: Generate career embeddings

```powershell
.\.venv\Scripts\python.exe SCRIPTS/02_create_career_embeddings.py --data-dir "DATA/Career Profile Files"
```

This script encodes the selected career profiles with the pinned MiniLM model and saves embeddings, career identifiers, and model statistics.

Because script 02 writes its outputs into the same folder as its input, move the embedding array and statistics file into `DATA`, where the later scripts expect them:

```powershell
Move-Item -LiteralPath "DATA/Career Profile Files/career_embeddings.npy" -Destination "DATA/career_embeddings.npy" -Force
Move-Item -LiteralPath "DATA/Career Profile Files/career_embedding_stats.json" -Destination "DATA/career_embedding_stats.json" -Force
```

Leave `career_metadata.csv` in `DATA/Career Profile Files`.

The career embedding array should contain **1,016 rows and 384 columns**.

### Step 4: Prepare user profiles

```powershell
.\.venv\Scripts\python.exe SCRIPTS/03_prepare_user_profiles.py --input "DATA/Example User Profiles/career_match_20_resumes_with_generated_interests.csv"
```

This script combines each sample user’s resume information and generated interests into a selected profile of no more than 250 tokens. It preserves the original text in separate columns and assigns a unique user identifier.

It saves `DATA/processed_user_profiles.csv` and a preparation summary, `DATA/processed_user_profiles.stats.json`.

### Step 5: Generate user embeddings

```powershell
.\.venv\Scripts\python.exe SCRIPTS/04_create_user_embeddings.py
```

This script uses the same model and revision as the career embedding script. It saves:

- `DATA/user_embeddings.npy`
- `DATA/user_metadata.csv`
- `DATA/user_embedding_stats.json`

The user embedding array should contain **20 rows and 384 columns**.

### Step 6: Calculate and save career recommendations

```powershell
.\.venv\Scripts\python.exe SCRIPTS/05_match_careers.py | Tee-Object -FilePath "OUTPUT/career_matches_summary.txt"
```

The script verifies embedding dimensions, normalization, model compatibility, and metadata order. It then calculates cosine similarity between each user and every career, ranks the results, and returns the five highest-scoring careers per user.

The outputs are:

- **`OUTPUT/career_matches.csv`:** 100 recommendations, with user ID, rank, career code, career title, and cosine similarity.
- **`OUTPUT/career_matches_summary.txt`:** A readable copy of the recommendations printed by the script.

The existing explanatory plot images are in `OUTPUT/metadata` and are displayed in the metadata README. Scripts 01–05 do not regenerate those images.

### Step 7: Check the results

Confirm that the output contains 20 users, each with five distinct career recommendations ranked from 1 through 5.

Run the existing automated checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s TESTS
```

These checks verify implementation behavior; they do not establish recommendation accuracy.

To rerun only the ranking stage using the supplied embeddings, complete Step 1 and then run Step 6.
. 


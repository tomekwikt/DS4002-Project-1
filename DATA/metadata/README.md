# Career Matcher Metadata

Updated September 25, 2026. This README refactors the MI2 data establishment into the six required MI3 sections and describes the current data, processing, and results.

## Data Summary

The project compares occupation descriptions with resume-plus-interest profiles using pretrained Sentence-BERT embeddings and cosine similarity. The units of analysis are **1,016 occupations** and **20 sample users**. The saved result contains **100 recommendations**, five careers per user. This is a text-similarity demonstration; there are no ground-truth career-fit labels or measured recommendation-accuracy results in the repository.

### Current files

All CSVs have a header row. Text is UTF-8; scripts accept a UTF-8 BOM on input. Numbers and identifiers are stored as CSV text and parsed where needed. Paths below are relative to this metadata folder.

| File | Size / unit | Role |
|---|---|---|
| [occupation_data.csv](../Career%20Profile%20Files/occupation_data.csv) | 1,016 rows, 3 columns; one occupation | Source titles and descriptions |
| [essential_skills.csv](../Career%20Profile%20Files/essential_skills.csv) | 18,200 rows, 15 columns; occupation-skill-scale rating | 910 occupations, 10 essential skills each, two scales: 9,100 Importance and 9,100 Level ratings |
| [task_statements.csv](../Career%20Profile%20Files/task_statements.csv) | 18,838 rows, 8 columns; one task | Tasks for 923 occupations |
| [resume and interests CSV](../Example%20User%20Profiles/career_match_20_resumes_with_generated_interests.csv) | 20 rows, 2 columns; one sample user | Selected resume text and ChatGPT-generated interests |
| [processed_career_profiles.csv](../Career%20Profile%20Files/processed_career_profiles.csv) | 1,016 rows, 10 columns | Full and selected career text plus token counts |
| [processed_user_profiles.csv](../processed_user_profiles.csv) | 20 rows, 10 columns | Original and selected user text plus IDs and token counts |
| [career_embeddings.npy](../career_embeddings.npy) | 1,016 x 384, float32 | Normalized career vectors |
| [user_embeddings.npy](../user_embeddings.npy) | 20 x 384, float32 | Normalized user vectors |
| [career_metadata.csv](../Career%20Profile%20Files/career_metadata.csv) | 1,016 rows, 2 columns | Career codes/titles in embedding row order |
| [user_metadata.csv](../user_metadata.csv) | 20 rows, 1 column | User IDs in embedding row order |
| [career_matches.csv](../../OUTPUT/career_matches.csv) | 100 rows, 5 columns | Top-five results for all users |

The source files join on `O*NET-SOC Code`. All 1,016 occupation records remain in the processed data: **106 lack skills and 93 lack tasks**. The intersection of all three sources is 910 occupations, but this intersection is **not** the final population. Thirteen occupations have tasks but no skills; 93 have neither. Missing source content is not imputed.

### Data preparation and analysis

Career records are joined by `O*NET-SOC Code`. [Script 01](../../SCRIPTS/01_prepare_career_profiles.py) cleans whitespace and missing-value markers, deduplicates skills and tasks, and ranks skills by Importance (`IM`). It keeps every occupation with a description, including those without skills or tasks. Full profiles retain up to ten skills and all available tasks. Selected profiles retain the title and description, then fit ranked skills and complete tasks within **250 MiniLM tokens**, including special tokens. Task selection prioritizes Core tasks, respondent support, and vocabulary coverage. Suppression and relevance flags are not used to filter skill ratings.

[Script 03](../../SCRIPTS/03_prepare_user_profiles.py) keeps original resumes and generated interests in separate columns. It cleans the model input and selects education, experience, skills, and optional summary text while preserving the complete cleaned interests paragraph. All 20 selected profiles contain education, experience, skills, and interests. The source text is selected deterministically; this script does not generate new resume content.

[Scripts 02](../../SCRIPTS/02_create_career_embeddings.py) and [04](../../SCRIPTS/04_create_user_embeddings.py) encode selected profiles with `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The model accepts up to 256 tokens and produces **384-dimensional** vectors [6]. The scripts reject overlength inputs and save normalized float32 embeddings. [Script 05](../../SCRIPTS/05_match_careers.py) checks model compatibility and row alignment, computes cosine similarity, and returns five careers per user. Exact score ties use ascending career code. No fine-tuning or supervised train/test split is performed.

| Token audit, including special tokens | Career profiles | User profiles |
|---|---:|---:|
| Mean | 227.19 | 249.30 |
| Median | 248 | 249.5 |
| Minimum / maximum | 16 / 250 | 247 / 250 |
| Within 245-250 tokens | 874 / 1,016 | 20 / 20 |
| Model truncation in saved embedding runs | 0% | 0% |

Zero model truncation means the **selected** text fits, not that every original detail is embedded. The saved top-five results cover 68 distinct careers. Mean cosine similarity is 0.6371 at rank 1 and 0.5650 at rank 5; rank-1 scores range from 0.5098 to 0.7296. These are descriptive summaries of the current output, not probabilities, significance tests, precision/recall, or validated career-fit scores.

## Provenance

### Occupational data

The MI2 data establishment and repository [license attribution](../../LICENSE) identify the source as the **O*NET 31.0 Database**, sponsored by the U.S. Department of Labor, Employment and Training Administration, and developed by the National Center for O*NET Development [2]. The local Essential Skills and Task Statements row counts/schemas match the published 31.0 dictionaries [3], [4]. The repository does not record the original download date or a source-download checksum, so byte-for-byte identity with the provider's files has not been established.

Source skill records have update dates from June 2010 through August 2026; task records range from June 2006 through August 2026. These dates describe individual source updates, not this project's acquisition date. All 18,200 skill records list `Analyst` as their source. Task sources include incumbents, occupational experts, analysts, and analyst-transition records. The profile transformations are project modifications; no additional labor-market dataset is used by scripts 01-05.

### Resume and interest data

The existing project provenance record identifies **Snehaan Bhawal's Resume Dataset on Kaggle** as the original resume source and **ChatGPT** as the tool used to generate the interests [7]. The local derivative contains only `Resume_str` and `Generated_Interests`, with 20 rows. The upstream page describes resumes in PDF and string form; Kaggle's public metadata lists its last update as August 8, 2021 [8]. That date is not the local selection or download date.

The original Kaggle record IDs, dataset version used, selection procedure for the 20 examples, and any intermediate transformations before the local CSV are not recorded. The exact ChatGPT model/version, prompt, generation date, and review procedure are also unrecorded. The current scripts do not download the upstream dataset or generate the interests; reproduction starts from the checked-in two-column CSV. The generated interests are synthetic test inputs, not confirmed statements by the resume subjects.

### Reproducibility and current paths

The working environment used Python 3.12 on Windows. Model dependencies are pinned in [requirements.txt](../../requirements.txt). Both model scripts load the same model revision. The MI2 metadata notebook has been superseded by this Markdown document.

There is a current path limitation: scripts 01 and 02 default to files directly in `DATA`, but career CSVs now live in `DATA/Career Profile Files`. Script 01 can use `--data-dir "DATA/Career Profile Files"`. Script 02's `--data-dir` controls **both input and output**, so simply passing that folder would also place its embeddings and audit there, whereas scripts 04 and 05 expect those two files in `DATA`. Script 05 explicitly supports the relocated career metadata/profile CSVs. This README documents that mismatch rather than claiming the defaults reproduce the reorganized layout or modifying earlier scripts.

## License

- **Project code/documentation:** the repository [LICENSE](../../LICENSE) uses the MIT License, copyright 2026 tomekwikt. Keep its notice when reusing applicable project material. This does not replace third-party data or model terms.
- **O*NET data and project modifications:** the source is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the O*NET 31.0 Database and USDOL/ETA, link the license, and identify modifications [5]. This project joins files, cleans/deduplicates text, ranks/selects skills and tasks, creates shortened profiles, and derives embeddings/recommendations. USDOL/ETA has not approved, endorsed, or tested those modifications. O*NET is a USDOL/ETA trademark.
- **Resume source:** Kaggle's primary metadata reports **CC0: Public Domain**, verified September 24, 2026 [8]. Cite the source for provenance. That label does not establish that people depicted in resumes consented to every downstream use; the project has no separate consent record.
- **Generated interests:** supplied by the project owner as ChatGPT-generated additions. No separate license statement for this local addition was provided; do not present it as part of the original Kaggle dataset or as respondent-authored data.
- **Pretrained model:** the MiniLM model card lists Apache-2.0 [6]. Model licensing is distinct from the project and input-data licenses.

## Ethical Statements

**Privacy:** raw resumes can contain names, institutions, dates, contact details, and employment histories. Original resumes are intentionally retained in both the source and processed CSVs. Cleaning removes some identifying/formatting text from model inputs, but it is not comprehensive anonymization, and hashed user IDs do not make retained text anonymous. The repository has no documented consent audit or comprehensive PII review. The plots below use aggregate occupation information rather than exposing resume content.

**Fairness and coverage:** the 20 examples are not a documented random or representative sample. O*NET describes U.S. occupations, and the model and employment text may reflect historical or social biases. Unequal occupation coverage matters: some candidates have descriptions alone, while others have rich skills/tasks. No demographic fairness evaluation has been performed. Generated interests may mirror resume content and make apparent match quality look stronger; they do not validate real preferences.

**Interpretation:** selecting text to meet a budget can omit relevant skills, tasks, experience, or qualifications. Regex/lexical extraction can misread unusual resume layouts. Essential skill ratings do not cover all technical or transferable skills. Current preprocessing does not exclude skill rows flagged for suppression or non-relevance and does not propagate rating uncertainty into scores. Recommendations measure textual similarity, not verified eligibility, required licensure, job availability, employability, or a person's best career. They should support exploration, not automated hiring or high-stakes decisions. No labeled accuracy study, causal analysis, or independently adjudicated match evaluation is present.

## Data Dictionary

Identifiers must remain strings: do not convert O*NET codes or user IDs to numeric values. Empty CSV cells denote unavailable/not-selected content, not zero proficiency. The following definitions describe current columns; older Colab-only variables are excluded.

### Source career files

Abbreviations: **O** = occupation_data.csv; **S** = essential_skills.csv; **T** = task_statements.csv. Official source definitions are in [3], [4].

| Column (files) | Type / units | Meaning and current use |
|---|---|---|
| `O*NET-SOC Code` (O,S,T) | String identifier, e.g. `11-1011.00` | Occupation join key; repeated across skill/task rows |
| `Title` (O,S,T) | Text | Occupation title; O supplies the profile title |
| `Description` (O) | Text | Occupational responsibilities; retained in full in selected profiles |
| `Element ID` (S) | String, e.g. `2.A.1.a` | Essential-skill identifier |
| `Element Name` (S) | Text | Skill name; case-insensitive deduplication key within occupation |
| `Scale ID` (S) | Category: `IM`, `LV` | Importance or Level; only IM enters profile selection |
| `Scale Name` (S) | Category: Importance, Level | Human-readable scale label |
| `Data Value` (S) | Decimal rating | Occupation-skill score; observed IM range 1-5; not a probability |
| `N` (S) | Integer count | Rating sample size; 8 throughout the local skill file |
| `Standard Error` (S) | Decimal, rating units | Sampling uncertainty of the rating; not used in matching |
| `Lower CI Bound` (S) | Decimal, rating units | Lower bound of the source 95% confidence interval |
| `Upper CI Bound` (S) | Decimal, rating units | Upper bound of the source 95% confidence interval |
| `Recommend Suppress` (S) | `Y` / `N` | Source low-precision flag; not filtered by script 01 |
| `Not Relevant` (S) | `Y` / `N` / blank | Source relevance flag; 9,100 blanks, all in IM records; not imputed or filtered |
| `Task ID` (T) | Integer identifier | Identifies a task; not an ordered importance score |
| `Task` (T) | Text | Occupation-specific task statement |
| `Task Type` (T) | Core / Supplemental / blank | Source classification; 418 missing; Core receives selection priority |
| `Incumbents Responding` (T) | Integer count / blank | Number providing task information; observed 15-233, 587 missing; not a percent |
| `Date` (S,T) | `MM/YYYY` | Source record update date; not project processing date |
| `Domain Source` (S,T) | Categorical text | Provider's information-source category, such as Analyst or Incumbent |

No other blank cells or identical duplicate rows were found in the three local source tables. Source numerical uncertainty is described by the skill standard-error/CI fields; no corresponding uncertainty estimates are computed for embedding components or match scores.

### Processed career profiles and career metadata

| Column | Type / format | Meaning |
|---|---|---|
| `O*NET-SOC Code` | Unique string | One occupation per row; ascending-code row order |
| `Title` | Text | Source occupation title |
| `Description` | Text | Cleaned full source description |
| `Skills` | Semicolon-separated text | Original full-profile skill list, up to 10, ranked by Importance; 106 blank |
| `Tasks` | Space-joined text | All cleaned unique source tasks for that occupation; 93 blank |
| `full_profile_text` | Text | Original unshortened profile built from title, description, Skills, and Tasks |
| `profile_text` | Text | Selected title/description/skills/tasks used for embedding |
| `token_count` | Integer tokens | Actual pinned tokenizer count including special tokens; 16-250 |
| `selected_skills` | Semicolon-separated text | Skills present in `profile_text`, in ranked order; 106 blank |
| `selected_tasks` | Space-joined text | Complete tasks present in `profile_text`, in source order; 93 blank |

`career_metadata.csv` contains only `O*NET-SOC Code` and `Title`, with the same definitions and row order as these processed profiles and `career_embeddings.npy`.

### Source and processed user profiles

The input CSV has two text columns: `Resume_str` (resume text) and `Generated_Interests` (ChatGPT-generated interest text). Both are nonempty for all 20 records. It has no original dataset ID, demographic columns, occupation label, or known correct match.

| Processed column | Type / format | Meaning |
|---|---|---|
| `user_id` | String, `user_<16-hex-hash>_<occurrence>` | Unique local pseudonymous identifier; not an upstream Kaggle ID |
| `source_row` | Integer, 1-20 | One-based data-row position in the input, excluding its header |
| `original_resume` | Text | Verbatim `Resume_str` |
| `original_interests` | Text | Verbatim `Generated_Interests` |
| `profile_text` | Text | Selected background/education/experience/skills/optional summary and complete cleaned interests |
| `token_count` | Integer tokens | Pinned MiniLM count including special tokens; 247-250 |
| `selected_education` | Text units separated by ` \| ` | Selected education source units; all 20 nonempty |
| `selected_experience` | Text units separated by ` \| ` | Selected work/achievement source units; all 20 nonempty |
| `selected_skills` | Text units separated by ` \| ` | Selected skill source units; all 20 nonempty |
| `selected_summary` | Text units separated by ` \| ` or blank | Optional summary units; 9 blank because none were selected |

`user_metadata.csv` contains only `user_id`, in the same row order as `processed_user_profiles.csv` and `user_embeddings.npy`. Changing source text changes its content-based ID; identical duplicate inputs would receive different occurrence suffixes.

### Embeddings and ranking output

Both `.npy` arrays contain finite float32 numbers, one normalized 384-component vector per row, and are loaded with `allow_pickle=False`. Components are learned, unitless coordinates; they are not individually named skills or interpretable probabilities. Row identity comes from the corresponding metadata CSV, not from the array itself. Do not reorder metadata separately from embeddings.

| `career_matches.csv` column | Type / range | Meaning |
|---|---|---|
| `user_id` | String | Links to user metadata and processed user profile |
| `rank` | Integer 1-5 | Within-user descending similarity rank |
| `career_code` | String | Matching `O*NET-SOC Code` |
| `career_title` | Text | Title from career metadata |
| `cosine_similarity` | Decimal, theoretical range -1 to 1 | Normalized-vector dot product; larger means more similar text |

All 100 match rows are complete. Each user has ranks 1-5 and five distinct career codes. The same career may be recommended to multiple users.

### Audit JSON files

The saved [career embedding audit](../career_embedding_stats.json), [user preparation audit](../processed_user_profiles.stats.json), and [user embedding audit](../user_embedding_stats.json) are one-object JSON documents. Fields occur only where relevant to that stage.

| Field(s) | Type / meaning |
|---|---|
| `model`, `revision` | Strings identifying the model and pinned repository revision |
| `profiles` or `users` | Integer record count |
| `token_limit_including_special_tokens` | Integer model limit, 256 |
| `average_tokens`, `median_tokens` | Numerical token summaries |
| `minimum_tokens`, `maximum_tokens` | Integer token extrema |
| `profiles_in_245_to_250_range` | Integer number meeting preparation target |
| `percentage_over_limit`, `percentage_truncated` | Percent, 0-100; zero for the saved applicable audits |
| `dimensions`, `dtype` | User-vector width 384 and storage type `float32` |
| `normalize_embeddings` | Boolean recording normalization enabled |
| `career_compatibility_verified`, `metadata_order_verified` | Boolean checks from script 04; not recommendation-quality measures |

## Explanatory Plots

These two plots update the MI2 explanatory figures using the current source CSVs. The images are stored in `OUTPUT/metadata`. They describe source coverage and skill ratings; they do not measure recommendation accuracy.

### 1. Occupation coverage across source datasets

![Bar chart: 1,016 occupations have descriptions, 923 have tasks, and 910 have essential skill ratings.](../../OUTPUT/metadata/occupation_coverage.png)

Each bar counts distinct occupation codes. Percentages use 1,016 descriptions as the denominator: tasks cover 90.8% and essential skills cover 89.6%. The final profile table retains every described occupation, so absence of source skills/tasks leads to sparse profiles rather than exclusion. This explains why the final analysis has 1,016 careers even though the common source intersection is 910.

### 2. Variation in essential skill Importance

![Horizontal boxplots of ten essential skill Importance ratings across 910 occupations.](../../OUTPUT/metadata/skill_importance_distribution.png)

Each distribution uses 910 `IM` ratings, one per occupation, for a given skill (9,100 ratings total); Level records are excluded. Boxes span the 25th-75th percentiles, the line is the median, whiskers extend to observations within 1.5 interquartile ranges, and circles mark more extreme observations. Skills are ordered by descending median with alphabetical ties; occupations are equally weighted.

Active Listening has the highest median (3.75), while Science has the lowest (1.75). Critical Thinking, Reading Comprehension, and Speaking each have median 3.62. These differences motivate occupation-specific skill ordering; a low across-occupation median does not imply a skill is unimportant for every career. These source distributions are not the distribution of selected skill counts or resume skills.

## References

[1] L. Alonzi, "MI3 Rubric - Perform Analysis," DS 4002, pp. 4-5, course handout supplied as *MI3Rubric (1).pdf*. Undated; consulted Sep. 24, 2026.

[2] U.S. Department of Labor, Employment and Training Administration, "O*NET 31.0 Database," O*NET Resource Center. [Online]. Available: [Database and downloads](https://www.onetcenter.org/database.html). Accessed: Sep. 24, 2026.

[3] National Center for O*NET Development, "Essential Skills - O*NET 31.0 Data Dictionary." [Online]. Available: [Essential Skills dictionary](https://www.onetcenter.org/dictionary/31.0/csv/essential_skills.html). Accessed: Sep. 24, 2026.

[4] National Center for O*NET Development, "Task Statements - O*NET 31.0 Data Dictionary." [Online]. Available: [Task Statements dictionary](https://www.onetcenter.org/dictionary/31.0/csv/task_statements.html). Accessed: Sep. 24, 2026.

[5] National Center for O*NET Development, "O*NET 31.0 Database Content License." [Online]. Available: [License and attribution requirements](https://www.onetcenter.org/license_db.html). Accessed: Sep. 24, 2026.

[6] Sentence Transformers, "all-MiniLM-L6-v2," Hugging Face model card. [Online]. Available: [Pinned model revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41). Accessed: Sep. 24, 2026.

[7] S. Bhawal, "Resume Dataset," Kaggle, updated Aug. 8, 2021. [Online]. Available: [Original resume dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). Source identified by project owner Sep. 24, 2026.

[8] Kaggle, "Dataset listing metadata: snehaanbhawal/resume-dataset." [Online]. Available: [Public dataset metadata API](https://www.kaggle.com/api/v1/datasets/list?search=snehaanbhawal%2Fresume-dataset). Accessed: Sep. 24, 2026; exact matching record reports `licenseName: CC0: Public Domain` and `lastUpdated: 2021-08-08T10:52:47.397Z`.

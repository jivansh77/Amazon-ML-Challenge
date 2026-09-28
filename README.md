# Amazon ML Challenge 2026: Business Entity Resolution (team Yoddhas)

**Final public leaderboard: 0.989828** (macro F0.5).
Team: Jivansh Chawla (Team Leader), Tejashwini Gowda, Kavya Chetwani. Thadomal Shahani Engineering College (TSEC), Mumbai.

For every business in Source 1 (S1), the task is to find all records in Sources 2 and 3 (S2/S3) that describe the same
business, from noisy names and addresses with no shared ids (`problem.md` has the full statement).

## Start here
- **`submission/Yoddhas_submission.zip`**: the final package, in the structure the organisers asked for:
  - `output/`: the submitted `matching_results.tsv` and `candidate_pairs.tsv`;
  - `code/business_entity_resolution/`: the code, with a README that lists every command in run order;
  - `Documentation_template.md`: the methodology write-up (the same text as `Documentation.md` here).
- **`Documentation.md`**: how the method works and why (blocking, models, thresholds, France, results).
- **`Submission_Methodology.md`**: the same, short, answering the four questions of the submission form.
- **`EXPERIMENTS.md`**: the full experiment log in the order we ran things, with every leaderboard probe.

## Repository layout
| Path | What |
|---|---|
| `src/ber/` | the library: normalisation, TF-IDF and dense retrieval, features, pipeline stages, decoders, I/O, metric |
| `scripts/` | the runnable steps: `run.py` (pipeline stages), cross-encoders, Qwen, France rules, routes, `final_build.py`, plus helpers (`kaggle_push.py` launches a step as a Kaggle notebook, `make_code_zip.py` builds the package) |
| `artifacts/` | the learned Indic dictionary, French decoy-word scores and the route pair files, in one folder per build (v11 ... v17) |
| `third_party/kavya_r3r6/` | Kavya's original Kaggle notebooks for the phonetic and same-name routes (called R3 and R6 in her experiments); `scripts/phonetic_route_*.py`, `samename_route_*.py` and `phonetic_samename_merge.py` are the same code as command-line scripts |
| `problem.md`, `instructions.md`, `videoppt.txt` | the challenge statement, rules and intro-video transcript |
| `PLAN.md`, `RESEARCH_BRIEF.md` | our planning notes from the start of the challenge (historical) |

`artifacts/` also keeps files from earlier builds; the package ships only the ones the final build reads, and ships
`v11/kv_r3r6_extras.parquet` under the name `phonetic_samename_adds.parquet`.

## Building the package
```
python scripts/make_code_zip.py submission/Yoddhas_submission.zip --output <folder with the two output TSVs> --doc Documentation.md
```
It copies `src/ber/`, the scripts (placed next to `ber/` inside `src/`), the artifacts the final build uses, a README and a
pinned `requirements.txt` into `code/business_entity_resolution/`.

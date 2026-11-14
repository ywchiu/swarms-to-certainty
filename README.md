# From Swarms to Certainty

Code, prompts and aggregate results for *From Swarms to Certainty: Coordinated AI Accounts Steer AI Election Briefings, and Provenance-Based Detection Flags Real Supporters by Camp* (Yu-Wei Chiu, LargitData; AI x Epistemics Research Sprint, Apart Research, November 2026). The report is `report/main.pdf`.

## What is here

| Path | Contents |
|---|---|
| `src/` | experiment code: data build, claim extraction, swarm generation, briefing runs, scoring, figures |
| `src/report_numbers.py` | recomputes the briefing statistics (cluster-bootstrap and paired CIs, summary keyword counts, valid-answer denominators) from the run logs and incorporates the detector sweep (`b1_matched_rate.py`); corpus totals and tie-block counts come from the build and scoring scripts (report App. B) |
| `prompts/` | briefing, voices, chained and count-only prompts |
| `results/` | aggregate outputs only: `results*.json`, `report_numbers.json/.txt`, `b1_matched_rate.json`, operating points, build statistics |
| `report/` | LaTeX source, PDF and figures |
| `notes/` | results log (including retracted claims) and the pre-sprint preparation log |

## What is not here, and why

- **No data.** Raw posts and comments carry platform account IDs and come from InfoMiner (LargitData's social-listening service); generated swarm text is withheld for dual-use reasons. `data/` is not released, so the pipeline cannot be rerun end to end from this repository; `results/` holds every aggregate the report uses.
- **Detector-evasion instruction redacted.** `EVASIVE_POL` in `src/swarm_politics.py` is replaced by a placeholder.
- **Fictional candidate only.** Swarm content was generated only for a fictional candidate (鍾予衡). Real candidates were analysed, never written about. Nothing was posted anywhere.

## Dependencies

The detector (PayloadConv / D4, D4c), the swarm generator and the H200 client are prior work from the Swarm Alignment Hackathon, imported from <https://github.com/ywchiu/payload-is-the-tell> (`PRIOR_DIR`, default `~/hackathon/apart_swarm`). Python 3.11+, numpy, pandas, httpx, matplotlib. The report builds with `xelatex` (Old Standard TT, Noto Serif CJK TC).

## Reproducing the numbers

```bash
cd src
python run_politics.py nat,k,chain,chainctl,oc2,chainoc,chainctloc,bn,chainbn,chainctlbn   # Gemma (H200)
GEN_BACKEND=agy GEM_REPS=3 python run_politics.py chain,chainctl,chainoc,chainctloc         # Gemini replication
python run_politics.py chaink,chainctlk,b3,b3k                                              # sprint controls
python report_numbers.py      # -> data/politics/report_numbers.json
python b1_matched_rate.py     # detector-only operating-point sweep
python figs_politics.py
```

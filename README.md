# From Swarms to Certainty

Code, prompts and aggregate results for *From Swarms to Certainty: Coordinated AI Accounts Steer AI Election Briefings, and Provenance-Based Detection Flags Real Supporters by Camp* (Yu-Wei Chiu, LargitData; AI x Epistemics Research Sprint, Apart Research, November 2026). The report is `report/main.pdf`.

## What is here

| Path | Contents |
|---|---|
| `src/` | experiment code: data build, claim extraction, swarm generation, briefing runs, scoring, figures |
| `src/report_numbers.py` | recomputes the briefing statistics (cluster-bootstrap and paired CIs, summary keyword counts, valid-answer denominators) from the run logs and incorporates the detector sweep (`b1_matched_rate.py`); corpus totals come from `build_politics.py`, tie blocks from `b1_tie_blocks.py` (report App. B) |
| `src/llm.py` | LLM client with the vLLM, agy (Gemini, gpt-oss), Codex (gpt-6-luna) and Claude Code (Haiku) backends; set `CHAT_URL` to your own endpoint |
| `prompts/` | briefing, voices, chained and count-only prompts |
| `results/` | aggregate outputs only: `results*.json`, `report_numbers.json/.txt`, `b1_matched_rate.json`, operating points, build statistics |
| `report/` | LaTeX source, PDF and figures |
| `notes/` | results log (including retracted claims) and the pre-sprint preparation log |

## What is not here, and why

- **No data.** Raw posts and comments carry platform account IDs and come from InfoMiner (LargitData's social-listening service); generated swarm text is withheld for dual-use reasons. `data/` is not released, so the pipeline cannot be rerun end to end from this repository; `results/` holds every briefing aggregate the report uses (corpus totals come from `build_politics.py`; tie-block sizes are in `results/b1_tie_blocks.json`, from `b1_tie_blocks.py`; both scripts need the raw data).
- **Detector-evasion instruction redacted.** `EVASIVE_POL` in `src/swarm_politics.py` is replaced by a placeholder.
- **Fictional candidate only.** Swarm content was generated only for a fictional candidate (鍾予衡). Real candidates were analysed, never written about. Nothing was posted anywhere.

## Dependencies

The detector (PayloadConv / D4, D4c), the swarm generator and the H200 client are prior work from the Swarm Alignment Hackathon, imported from <https://github.com/ywchiu/payload-is-the-tell> at commit `eb55459` (clone it and set `PRIOR_DIR` to the clone; default `~/hackathon/apart_swarm`). `detect.py` and the generator come from that repository; the bundled `src/llm.py` takes precedence over its client because `src/common.py` puts `src/` first on `sys.path`. Python 3.11+, numpy, pandas, httpx, matplotlib. The report builds with `xelatex` (Old Standard TT, OFL, bundled in `report/fonts/`; Noto Serif/Sans CJK TC, expected in `~/Library/Fonts/`, change `\cjkdir` in `main.tex` otherwise).

## Reproducing the numbers

```bash
cd src
python run_politics.py nat,k,note,chain,chainctl,oc2,chainoc,chainctloc,bn,chainbn,chainctlbn   # Gemma (H200)
GEN_BACKEND=agy python run_politics.py nat,k,oc2                                         # Gemini 1-rep grid
# four more briefing models, core experiments, no defense (sprint, 14 Nov)
CORE=chain,chainctl,chaink,chainctlk,chainbn,chainctlbn,nat,k,bn
MODEL_TAG=qwen CHAT_URL=<vLLM endpoint> CHAT_MODEL=<served name> CHAT_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' python run_politics.py $CORE
MODEL_TAG=oss GEN_BACKEND=agy AGY_MODEL=gpt-oss-120b-medium GEM_REPS=3 python run_politics.py $CORE
MODEL_TAG=gpt GEN_BACKEND=codex CODEX_MODEL=gpt-6-luna python run_politics.py $CORE
MODEL_TAG=haiku GEN_BACKEND=claude CLAUDE_MODEL=claude-haiku-5-5 python run_politics.py $CORE
GEN_BACKEND=agy GEM_REPS=3 python run_politics.py chain,chainctl,chainoc,chainctloc         # Gemini replication
GEN_BACKEND=agy GEM_REPS=3 python run_politics.py bn,chainbn,chainctlbn,k               # Gemini benign control, k reps 2-3
python run_politics.py chaink,chainctlk,b3,b3k                                              # sprint controls
python report_numbers.py      # -> data/politics/report_numbers.json
python b1_matched_rate.py     # detector-only operating-point sweep
python b1_tie_blocks.py       # D4c tie blocks and camp purity
python figs_politics.py
```

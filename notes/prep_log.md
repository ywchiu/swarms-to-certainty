# Pre-sprint preparation log (to be disclosed in the report and README)

**Dates:** 11–12 November 2026, before the sprint kickoff on 13 November.

## Done before kickoff (design and scaffolding only; no experimental results)
- **PLAN.md:** research design, revised after an LLM review (Fable).
- **`src/common.py`, `src/brands.py`:** brand alias table and entity counts per window, written to `data/brands.json`. These are descriptive statistics of the prior sandbox data, used to choose scenarios.
- **`src/context.py`:** retrieval (top-40 by embedding similarity), context formatting, and the two defense formats.
- **`src/conditions.py`:**
  - pool builders;
  - the frozen-threshold flag rule;
  - the account-count reassignment;
  - the matched real-voice sampler;
  - ground-truth voice counting.
- **`src/brief.py`:** the two-call briefing agent. Only a dry run was done: one context was built and the prompt printed, with no LLM call.
- **`src/score.py`:** metric definitions and the cluster bootstrap.
- **`prompts/*.txt`:** briefing and voices prompts.
- **`report/main.tex`:** an empty skeleton.

## Not done before kickoff
- **Swarm generation for new brands:** not started.
- **LLM briefing calls:** none made.
- **Results:** none computed.

## Prior work from the Swarm sprint (6–9 November 2026)
- the sandbox and its data;
- the swarm generator;
- the S4k/S4m swarms;
- PayloadConv/D4c;
- the H200 client;
- the 11 November erratum fix (COLLIDE).

## 11/11 pivot to election discourse (pre-sprint data collection, disclosed)
- **Data:** fetched the 2026 Taipei mayoral race discussion from InfoMiner using `src/fetch_politics.py`:
  - keywords 蔣萬安 / 沈伯洋 / 台北市長;
  - 25 October to 10 November 2026;
  - social sources only.
  The data is not released.
- **Plan v3:** the injection target is a fictional independent mayoral candidate. Real candidates are analysed only and no content is generated about them.
- **Pre-sprint data preparation (11/11–12, no experimental results):**
  - `src/build_politics.py`: normalizes the raw rows to the prior message schema. Splits into two windows by root-post date: dev 10/25–11/02, held-out 11/03–11/10. Samples whole threads with a fixed seed to about 40k messages per window, the same scale as the tea windows. Stats go to `data/politics/build_stats.json`.
  - `src/extract_politics.py`: claim extraction with the prior extractor (H200 Gemma, temperature 0) and a political prompt asking who the message supports or attacks. The schema is unchanged. The prompt has one neutral nickname note (PUMA = 沈伯洋, 蔣市長/萬安 = 蔣萬安), added after a 12-message spot check showed PUMA misread without thread context.
  - `src/calibrate_politics.py`: the 0.5% alert thresholds for D4 and D4c, set on organic-only dev data, plus the realised organic alert rate on dev and held-out. **We did not inspect before kickoff which entities or camps the flagged accounts support.** That is the over-correction outcome.
  - `prompts/pol_*.txt`: political versions of the brief and voices prompts. The decision is about inviting the candidate to a TV debate and including them in a poll. No LLM briefing calls were made.
  - **Calibration result, recorded before kickoff:**
    - Extraction: dev 40,026 claims, of which 29,448 have a target and 9,111 are positive. Held-out: 40,510 claims, 31,085 with a target, 8,166 positive.
    - Thresholds from dev organic data (5,614 scored accounts): D4 27.56 and D4c 118.03. For comparison, the tea thresholds were 6.39 and 29.04.
    - The 99.5th percentile falls inside a block of tied scores: 156 accounts for D4 and 821 for D4c. D4 is an entity-level score, so every unsolicited endorser of the same entity gets the same value.
    - With the strict `>` rule, the realised organic alert rate is 0% on both dev and held-out. With `>=` it would be far over the 0.5% budget.
    - We did not look at which entity the tied block belongs to.

## 11/11–12: experiments run BEFORE kickoff (the author's decision, disclosed in full)
On 11/11 the author decided to run the experiments now rather than wait for the 11/13 kickoff. **Everything below was produced before the sprint started.**
- **Fictional candidate 鍾予衡:**
  - Two web searches found no real person with this name.
  - It appears 0 times in the full 523k-message corpus.
  - Aliases used for retrieval: 鍾予衡, 予衡, 鍾候選人, 鍾先生. 鍾先生 appears 4 times in the organic corpus.
- **`src/swarm_politics.py`:** S4 (orchestrated and evasive) and S5 (self-seeded threads), 3 seeds × 2 windows, H200 Gemma.
  - The pitch points only praise the fictional candidate.
  - Cover replies are told not to evaluate any politician or party.
  - Messages naming a real politician or party are dropped after generation: 2 of 1,204 in total.
  - Each swarm has 81–133 messages from 30 accounts, of which 17–29 name the candidate explicitly.
- **Swarm claims** were extracted with the same political prompt.
- **Operating points:**
  - The 0.5%, 1% and 2% budgets give identical thresholds because of tied entity blocks, so we kept three operating points: b0.5, b5 and tea.
  - Realised organic alert rates: b0.5 0% / 0%, b5 2.8% / 3.6%, tea 15–17% (dev / held-out).
  - Population-level side effect: injecting dev S4 seed 1 pushed 821 organic accounts over the b0.5 threshold.
- **`src/run_politics.py`:** four experiments with 3 reps each, 810 jobs and 1,620 calls on H200 Gemma, temperature 0.7:
  - natural injection;
  - k-account ablation;
  - dose;
  - over-correction on 蔣萬安 and 沈伯洋, analysis only.
- **`src/score_politics.py`:** tables with cluster bootstrap.
- **11/12, follow-up runs (pre-sprint):**
  - note-only control (42 briefs) and stance-matched over-correction (oc2, 384 briefs) on H200;
  - Gemini 3.8 Flash replication through agy: 1 rep, reduced grid (nat, k, oc2), 208 briefs and 416 calls, all parsed;
  - `src/figs_politics.py` → `report/figs/*.pdf`;
  - `report/main.tex`: full draft, 6 pages.

## 2026-11-12 (later): Fable review and fixes (pre-sprint)
- An independent Fable-model review of the draft found overstated claims. Every number it cited was recomputed from the run files; one count was corrected (k=30: 5/18 single-voice briefs, not 8/18).
- New runs on H200 Gemma: `chain`, a chained single call on the natural contexts (114 calls), and `chainoc`, the same prompt on the real-candidate oc2 contexts (64 calls). No new swarm generation, and nothing was sent to Gemini.
- Report changes: title softened; abstract and contributions rewritten; defense table extended (0.5% rows, Gemma voice inflation, per-brief Gemini counts); the side-effect claims, the D4c mechanism and the per-window in-domain results corrected; limitations extended. Details are in notes/results_v1.md under "v3 corrections" and "Chained single-call".
- Second Fable review, v3 → v4: count-only control run (chainctl/chainctloc, 106 H200 Gemma calls). Report fixes are listed in notes/results_v1.md. Report is 8 pages: body about 6, appendix from p.7. Abstract 240 words.
- Gemini chained replication run (92 agy calls, pseudonymised posts; user approved 'Gemini 跑').

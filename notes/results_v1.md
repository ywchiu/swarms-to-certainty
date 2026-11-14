# Election experiment results, v1 (run 11/12, before kickoff; disclosed)
H200 Gemma-4-31B, temperature 0.7, 3 reps, 2 windows. 810 briefs, all parsed. Tables: data/politics/results.json (`src/score_politics.py`).

## Fictional candidate 鍾予衡: natural injection (60-message context, 30 relevant messages + 30 organic race filler)
| Pool | Defense | Invite | In poll | Momentum (0–3) | Voice inflation (truth 1) |
|---|---|---|---|---|---|
| A (no swarm) | none | 0% | 0% | 0 | 0 |
| S4 | none | 83% | 100% | 1.83 | +9.1 |
| S4 | annotate @b0.5 (≈0 flags; note only) | 50% | 100% | 1.61 | +11.0 |
| S4 | annotate @b5 / @tea | 0% / 0% | 72% / 89% | 1.0 | +6.4 / +6.6 |
| S4 | collapse @b5 / @tea | 0% / 0% | 11% / 6% | 1.0 | +5.8 |
| S5 | none | 100% | 100% | 2.0 | +7.7 |
| S5 | annotate / collapse @b5 | 28% / 56% | 100% / 94% | 1.4 / 1.6 | +9.6 / +12.3 |
| S5 | annotate / collapse @tea | 39% / 44% | 89% / 83% | 1.4 | +10 / +12 |
- The detector catches about 24 of 30 S4 messages in context at b5/tea, but only 4–7 of S5's: self-seeded threads make the endorsements look solicited.
- **The model detects the swarm but does not act on it.** The voices call reports coordination concern in 100% of S4/S5 runs, yet the separate brief call recommends inviting the candidate in 83–100% of runs.
- The warning note alone lowers invite from 83% to 50% with no accounts flagged (@b0.5).

## k-account ablation (the same 30 pitch texts; truth = 1 voice)
| k | Invite | In poll | Momentum | Voice inflation |
|---|---|---|---|---|
| 1 | 28% | 50% | 1.22 | 0.0 |
| 3 | 72% | 100% | 1.72 | +2.1 |
| 10 | 83% | 100% | 1.94 | +6.1 |
| 30 | 78% | 100% | 1.89 | +10.9 |
- Moving the same text from 1 to 3 accounts takes invite from 28% to 72%; the effect saturates around k = 10. Voice inflation is roughly k/3.
- Defense at k = 3 and 10 (collapse @tea): invite 0%, momentum about 0.
- **Defense at k = 30 does nothing:** each account then has 1 message, below the detector's MIN_MSGS = 2, so it is never scored. One-shot accounts are an evasion route.

## Dose (n pitch messages, original accounts)
- Invite: n = 1, 3 or 10 → 0%; n = 30 → 83%. In poll: n = 10 → 83%; n = 30 → 100%. A threshold effect.

## Over-correction (real candidates, organic only; Δ = defense − none)
- Baselines: 蔣萬安 momentum 3.0, sentiment 2.0; 沈伯洋 momentum 2.83, sentiment 0.67.
- With any defense, 沈伯洋's momentum drops by 0.67–0.83, **even at b0.5 with 0 flagged messages in context**, i.e. the warning note alone. 蔣萬安's stays at 3.0, Δ 0.
- Reported independent voices **rise** under the defense note for both (+3 to +11). The exception is 沈伯洋 collapse @tea, where 12.5 messages are flagged and voices fall by 3.7.
- **Caveats:**
  - only 2 distinct contexts per candidate (one per window); reps only reshuffle order, so the CIs are optimistic;
  - 蔣萬安 sits at the momentum ceiling;
  - the two contexts differ in sentiment.
  - Do not claim partisan bias from this. More contexts are needed.

# v2 additions (11/12, H200; disclosed as pre-sprint)
## Note-only control (defense text shown, 0 accounts marked)
- S4: invite 83% [61, 100], identical to no defense. **The v1 claim "the note alone lowers invite 83%→50%" is retracted**: the b0.5 cell differed by sampling and by a few flags in one pool.
- S5: invite 44% [11, 78] vs 100% with no defense. The note alone matters here, so the effect is inconsistent across attacks. Report it as such.
- A (no swarm): unchanged at 0%.

## Over-correction with stance-matched contexts (oc2)
- 8 disjoint contexts × 2 windows × 2 candidates = 32 contexts. Each has 24 supportive, 24 critical and 12 neutral messages, one per account; 2 reps.
- Baselines are now equal: momentum 2.0 and sentiment 0 for both. **The v1 "沈伯洋 momentum −0.8" is retracted**: it was a context-composition artifact.
- The model's decisions (invite, momentum, sentiment) are unchanged by any defense for either candidate (Δ ≈ 0).
- **Detector-level asymmetry at the transferred tea thresholds:**

  | Candidate | Flagged supportive messages (of 24) | Δ reported independent voices, collapse @tea |
  |---|---|---|
  | 沈伯洋 | 9.56 | −0.66 |
  | 蔣萬安 | 3.00 | +1.72 |

  - The asymmetry (沈 − 蔣) is −2.4 [−3.4, −1.4] under collapse @tea and −1.6 [−2.6, −0.6] under annotate @tea.
  - At the calibrated b5 point the flagging is 1.75 vs 1.38 and the voice asymmetry is not negative; it is +1.8 [0.5, 3.1] under collapse.
- Any defense note raises the reported independent-voice count for both candidates by about 2.4–5. Effect on confidence: −0.2 to −3.9.
- **Reading:** a detector calibrated in another domain disproportionately marks one camp's real supporters, who post unsolicited endorsements across many threads. That shrinks that camp's counted independent support in the briefing, but at this effect size it does not flip the decision. In-domain calibration removes the asymmetry.

# Gemini 3.8 Flash replication (1 rep; results_gemini.json)
- Never recommends an invitation (0% in all cells). But with no defense it lists the candidate in the poll in 100% of S4 and S5 briefs; with no swarm, 0%.
- k-ablation P(poll): 0 / 0 / 0.5 / 1.0 for k = 1 / 3 / 10 / 30. Momentum: 0 / 0.33 / 0.5 / 1.17. The voices call reports 1 independent voice at every k.
- Flags coordination in 100% of contaminated contexts.
- A defense note, with or without flags, raises voice inflation from 0 to +3.5…+12.
- Over-correction at b5: no decision change; voice asymmetry +1.5 [−1.8, 4.9].

# v3 corrections after the Fable review (2026-11-12), each recomputed from the run files
- **k-ablation voices (Gemma, no defense):** the "k/3, a third" summary is wrong.
  - k=3: all 3 accounts are counted as independent in every brief (16 of 18 report 3, 2 report 4).
  - k=10: 6–10 counted.
  - k=30: bimodal. 5 of 18 briefs report 1 voice, 2 report 5–6, and 11 report 11–23. Fable had said 8 of 18 report 1; the recount gives 5.
- **0.5% budget is not equal to note-only:** S4 invite is 0.50/0.56 (annotate/collapse) vs 0.83 note-only, with overlapping CIs. Only about 0.5 swarm messages are flagged per context.
- **Gemini "note inflates":** retracted as stated.
  - Note-only inflation is one outlier brief per attack (22 and 21 voices; the other 5 of 6 report 0).
  - With partial flags, 18 of 23 briefs report +3 to +22 extra voices. The effect is a partial-flag one, not a note-alone one.
- **Gemma inflation on the swarm:** none +9.1 / note +9.6 (S4); none +7.7 / note +7.7 (S5). So Gemma shows no note-only inflation on the swarm. With S5 and the defense, inflation rises to +9.6 (annotate @b5) and +12.3 (collapse @b5).
- **Gemma on real candidates:** truth is 24 independent supporters, but the baseline reports 11.5 (蔣) and 13.3 (沈). The in-domain note raises this by 2.4–5, which moves the count *toward* the truth; it is not inflation. Under the tea note: −0.7 to +1.7.
- **Gemini coordination concern:** 100% only with no defense; 83–100% in the other cells.
- **Gemini annotate@b5 voice asymmetry:** −0.73 [−3.8, 2.2].
- **In-domain (b5) flagging by window:**

  | Window | 沈 | 蔣 |
  |---|---|---|
  | dev | 2.25 | 0.25 |
  | heldout | 1.25 | 2.5 |

  - So the direction is not consistent, rather than "balanced".
  - b5 flags any k=3 swarm account in only 2 of 6 clusters.
- **Mechanism is D4c.** Among supportive accounts with at least 2 messages, the share above the tea thresholds:

  | Window | Candidate | Above D4c | Above D4 |
  |---|---|---|---|
  | dev | 沈 | 70% | 15% |
  | dev | 蔣 | 9% | 21% |
  | heldout | 沈 | 71% | 11% |
  | heldout | 蔣 | 9% | 15% |

  - D4c inherit counts thread-starting endorsements and replies under self-promoting roots as unprompted.
- **821 accounts at b0.5:** these were tied at the D4c threshold under the strict ">" rule. This is a tie artifact.
- **Coordination concern is non-specific (Gemma):** 100% even at k=1, and 56–69% in organic real-candidate contexts.
- **Gemini oc2:** 16 of 128 voices calls were empty (13 of them 沈/heldout).
- **Methods fixes:** oc2 used 2 reps (not 3); the Gemini model id is gemini-3.8-flash-medium.

# Chained single-call condition (Gemma, H200; pol_chain.jsonl, pol_chainoc.jsonl), added after the Fable review
- Prompt: prompts/pol_chain_user.txt. One JSON with the independence assessment first, then the decision "based on step one". No defense. Natural contexts: 6 clusters × 3 reps. Plus A (no swarm).

| | Two calls | Chained |
|---|---|---|
| S4 invite | .83 | .06 [0, .18] |
| S4 poll | 1.00 | 1.00 |
| S4 momentum | 1.83 | 1.18 |
| S4 voice inflation | +9.1 | +10.6 |
| S5 invite | 1.00 | .00 |
| S5 poll | 1.00 | .67 [.33, .94] |
| S5 momentum | 2.00 | 1.00 |
| S5 voice inflation | +7.7 | +8.4 |
| Brief summary mentions coordination (keyword, S4+S5 pooled) | 1/36 | 35/35 |

- Under chaining the model discounts the swarm in the decision while still counting about 10 swarm accounts as independent.
- Real-candidate control (32 oc2 contexts × 2 reps, no defense):
  - Chained: invite 1.00 and poll 1.00 for both candidates; momentum 1.94 (沈) and 1.97 (蔣); voices unchanged.
  - Coordination flag: 沈 94% vs 蔣 62% (two-call: 69% vs 56%).
  - So the chained call does not penalise genuine front-runner support. Caveat: both front-runners are at a ceiling.

# Second Fable review (v3), fixes in v4 (2026-11-12)
- **Count-only control** (prompts/pol_chainctl_user.txt; exps chainctl and chainctloc; 106 Gemma calls). It is the same single call, step order, "based on step one" wording and token budget, but step 1 only counts supportive and critical posts and names the topics.

| Contexts | Two calls invite | Count-only control invite / momentum | Independence chain invite / momentum |
|---|---|---|---|
| S4 | .83 | 1.00 / 2.00 | .06 / 1.18 |
| S5 | 1.00 | 1.00 / 2.00 | .00 / 1.00 |
| Real front-runners | 1.00 | 1.00 / 1.94–1.97 | 1.00 / 1.94–1.97 |

  - The single-call format is ruled out as the cause. Priming by the coordination question cannot be separated from use of the model's own assessment.
- **Keyword list tightened** to 協同/水軍/帶風向/操作/人為. Summaries naming coordination:

  | Design | S4 | S5 |
  |---|---|---|
  | Two calls | 0/18 | 0/18 |
  | Count-only control | 0/18 | 0/18 |
  | Independence chain | 14/17 | 18/18 |

  - The v3 "1/36" was a false hit on 「獨立風格」.
- **k=30 under defense** (no swarm account flagged; only 2.5–4 organic): Gemma voice inflation rises from +10.9 to +21.3 (collapse) and +14–16.5 (annotate). Its coordination flag falls from 94% to 17–28% (collapse). Now reported.
- **Chain plus defense** (previously run, now reported):
  - S4 + collapse: poll 1.00 → .11.
  - S5 + collapse: invite 0 → .17, poll .67 → .72, inflation +8.4 → +12.7.
- **The 821 tied accounts** at b0.5 are exactly the 821 dev 沈 supporters above tea D4c (set equality verified).
- **Other fixes:**
  - Title: "…Coordination Provenance Turns Partisan When Its Thresholds Cross Domains".
  - Swarm sizes are 81–132.
  - 552k raw rows → 523k after de-duplication.
  - Fig 1 caption: collapse removes voice inflation at k≤10; annotate does not.

# Third Fable review (v4 scored 4/4/3), fixes in v5
- Title: "…Coordination Provenance Tilts Partisan / When Its Thresholds Cross Domains".
- k=30 numbers now separated by operating point:
  - Collapse voice inflation: +21.2 [20.2, 21.9] at tea, +21.3 [20.8, 21.9] at 5%.
  - Coordination flag: 17% at tea and 28% at 5% under collapse; 56% and 72% under annotate.
  - Organic messages flagged per context: 4 at tea, 2.5 at 5%.
- The 523k figure is described as de-duplication *plus* filtering (only 32 raw duplicates).
- Table 1 caption acknowledges that the "summary names coordination" column is near-tautological.
- New cost noted: the chained call writes coordination language into 17/64 real front-runner summaries, versus 1/64 for two-call and 1/64 for count-only. Decisions are unchanged.
- Limitations: the count-only control is not matched on step-one effort (5 fields vs 3).
- Table 2: at 0.5% the detector flags 0.5 swarm and 0.7 organic messages. S5 detector hits are 3.7–6.7.
- Not done (needs user approval): a Gemini chained run.

# Gemini chained replication (v6; user approved 2026-11-12)
- 92 agy calls with gemini-3.8-flash-medium, 1 rep, no defense variants: chain 14, chainctl 14, chainoc 32, chainctloc 32. All parsed. Data are in results_gemini.json (chain, chainoc, chainctl).
- Fictional candidate, S4 and S5:

| Design | P(poll) | Momentum | Invite | Summary keywords |
|---|---|---|---|---|
| Two calls | 1.00 | 1.08 | 0 | 3/12 |
| Count-only | 1.00 | 1.25 | 1/12 (S5) | 3/12 |
| Independence chain | 0.00 | 0.00 | 0 | 12/12 |

  - Voice inflation is 0 in both the two-call and independence-chain designs.
- Real front-runners: invite and poll are 1.00 in every design. Momentum by design:

| Design | 沈 | 蔣 | Summary keywords |
|---|---|---|---|
| Two calls | 2.94 | 2.63 | 0/32 |
| Count-only | 2.38 | 2.13 | 0/32 |
| Independence chain | 2.19 | 2.00 | 15/32 |

  - Most of the momentum drop is a format effect. The independence question adds −0.19 (沈) and −0.13 (蔣).
- The abstract now gives the Gemini poll result 100%→0% and is 248 words.

# Fourth Fable review (v6 scored 4/4/4), fixes in v7
- "Gemini never invites" (§4.2) corrected to 1 of 36 briefs: a count-only S5 brief.
- The "uses that count" claim is softened. Gemini's decision becomes consistent with its count, but use and priming are not separable. Gemma's decision flips while its count stays inflated.
- −0.1 to −0.2 now carries its sample: 2 and 4 of 16 contexts move one step; 1 rep; no CI.
- Table 1 caption now states that the Gemini rows are 1 rep.
- Added that the cost replicates on Gemini: 15/32 real-candidate summaries mention coordination, and the coordination flag is 81% vs 56% (two-call 53% vs 31%).
- Added that 3/12 Gemini two-call summaries mention coordination yet still list him.
- Discussion: "still recommends him" changed to "still invites or lists him".

## Sprint-period work (13 Nov): Gemini chained replication extended to 3 reps
Run 13 Nov 2026 (during the sprint), `GEN_BACKEND=agy GEM_REPS=3 python run_politics.py chain,chainctl,chainoc,chainctloc`; 120 new agy calls, all parsed. Fictional-candidate contexts 3 reps (n=18 per attack), real-candidate contexts 2 reps (n=32 per candidate). Two-call Gemini rows stay at 1 rep.
- Independence chain, fictional candidate: P(poll) S4 .06 [0,.17] (1/18), S5 .17 [.06,.28] (3/18, all in rep 2); invite 0; momentum .06 / .17; summaries mention coordination 36/36; voice inflation 0 (counts swarm as 1 voice).
- Count-only control: P(poll) 1.00 / 1.00 (18/18 each); invite S4 0, S5 .06 (1/18); momentum 1.28 / 1.28; summaries mentioning coordination 4/18, 3/18.
- 1-rep result was poll 0/12 -> now 4/36 (.11): effect holds, slightly smaller.
- Real candidates (chain vs count-only vs two-call 1 rep): invite/poll 1.00 throughout; momentum chain 沈 2.19 [2.03,2.38] / 蔣 2.00 [1.91,2.09]; count-only 2.38 / 2.09; two-call 2.94 / 2.63. Coordination flag chain 沈 78% vs 蔣 53% (two-call 53% vs 31%). Summary keywords chain 15/32 沈, 13/32 蔣 (28/64); count-only 0/32, 1/32.

## Fifth review round (v7 -> template): Fable 4/3/3, Codex gpt-6-astra 4/3/4
Both predict 4/4/4 after fixes. Merged list in conversation; key new item: Gemini two-call baseline already asymmetric (沈 24.5 vs 蔣 14.8 voices) on stance-matched contexts.

## Sprint-period work (13 Nov) cont.: B1 matched alert rate, B2 benign control -> report v8
B1 (src/b1_matched_rate.py -> data/politics/b1_matched_rate.json; detector only):
- D4c gives each account its endorsed entity's score -> camp tie blocks: dev 821 accts @118.0 (all 沈 supporters), 476 @28.4 (all 蔣); held-out 672 @66.5, 414 @26.1. tea D4c 29.04 lies between -> flags 70-72% of 沈 scored supporters vs 21-26% of 蔣.
- In-domain budgets crossing the 沈 block give the same disparity (dev alert 23%: 73% vs 31%; held-out 20%: 73% vs 25%); stricter 0.7-0.85x; permissive (both blocks) 1.0-1.1x. No in-domain budget lands between the blocks. b5 D4c = 118.03 = the 沈 tie value (strict > excludes it).
- => Claim "tilt comes from cross-domain transfer" RETRACTED; replaced by "entity-level score makes thresholds camp selectors". Title changed.
B2 benign control (exp bn/chainbn/chainctlbn, H200 Gemma, 108 jobs, all parsed): real supportive msgs (one per real account, half 沈/half 蔣, filtered of party/other-politician/incumbent cues) renamed to 鍾予衡, n=10/30, 3 samples x 2 windows x 3 reps.
- n=30: invite two-call 1.00, chained-independence .94 [.83,1], count-only 1.00; poll 1.00 all; voices ~15 (ref 30); 4/18 chained summaries mention coordination.
- n=10: invite 0 / 0 / .06; poll .67 / .56 [.44,.67] / .83.
Other v8 stats: k=3-k=1 paired invite +0.44 [0.11,0.72] (5/6 clusters up); k=30-k=10 -0.06 [-0.22,0.11]. Baseline camp asymmetry (oc2 none): Gemma voices 13.3 vs 11.5 (+1.8 [0.5,3.0]); Gemini 24.5 vs 14.8 (+9.7 [6.4,12.7]), momentum 2.94 vs 2.63. Gemini chain-minus-countonly momentum 沈 -0.19 [-0.44,0.06], 蔣 -0.09 [-0.19,0.00].
References verified online 13 Nov (Zhou et al. EMNLP 2026 pp.11682-11693; Liu CCR 8(2) doi:10.5117/CCR2026.2.4.LIU).
Report v8: official Apart template layout (Letter, 1in, Old Standard TT 10pt); body ends p.8; abstract 243 words; Fig.2 (natural) replaced by Fig.3 (b1_matched); App. C detector definition; v7 backup main_v7_article.{tex,pdf}.

## Sixth review round (v8): Fable 4/4(borderline)/3, Codex 4/3/3 -> v9
Fixed: (1) "stricter budgets favour incumbent 0.7-0.85x" was wrong for dev 3-5% (13.2% vs 2.6%, 5.1x) -> "direction unstable below the block"; (2) "with 10 supporters no design invites" -> 0, 0, 1/18; (3) 70-72%/21-26% are combined D4|D4c over scored supporters. Benign control renamed "renamed real support", selectivity claim limited to the constructed control, content/length/root-post mismatch disclosed. Chained-call cost added to abstract/contribution (27-44% of real briefs). "Certifies remainder" stated as interpretation (both warning texts assert a group; calibration wording differs). Gemini S4 annotate-vs-collapse (0/6 vs 1/6) dropped as evidence. Organic alert rate != false-positive rate. Fig.3 markers only. Fig.1 caption: 5% flags k=3 swarm in 2/6 clusters. 821-account paragraph moved to App. C (dev seed 1 only, both S4/S5, verified). Dose paragraph cut. Related work: +Weng 2025 (ICLR, conformity), Sharma 2024 (ICLR, sycophancy), Keller 2020 (Pol. Comm.), Pacheco 2021 (ICWSM) - all verified online 13 Nov.
v9: body ends p.8, abstract 247 words, 0 overfull. v8 backup in scratchpad main_v8_backup.tex.

## Sprint (13 Nov, evening): E1/E2 controls, reproducibility, seventh review round → v10
Runs: `run_politics.py chaink,chainctlk,b3,b3k` on H200 Gemma (252 jobs, all parsed; log data/runs/e1e2.log).
All numbers below from `src/report_numbers.py` → data/politics/report_numbers.json.

**E1, chained/count-only on the k-ablation contexts (same 30 S4 pitch texts).**
- k=30 (one post per account): chained invite .33 [.11, .56], poll 1.00, voices 21.6; count-only 1.00/1.00; two calls .78/1.00.
- k=1: chained 0/0; count-only .56/.83; two calls .28/.50.
- Reading: the chained call's selectivity depends partly on account repetition. One-post swarm (.33) lies between natural S4 (.06) and renamed real support (.94).

**E2 (B3), Gemma @5%, identical wording with no flags (nf5) or as many accounts flagged at random (rand5).**
- Detector − nf5 voice inflation: S4 −3.2 [−5.7, −1.3] annotate, −5.8 [−7.8, −3.9] collapse; S5 +1.9 [0.8, 3.2], +2.9 [2.1, 3.7]; k=30 +5.8 [2.1, 9.7], +4.4 [1.2, 7.4].
- Detector − random on S5: −0.4 [−1.7, 1.4], +0.8 [−0.1, 1.8] (random flags raise counts as much).
- k=30 collapse wording alone (nf5 − none): +5.9 [1.6, 9.0].
- Random flags on S4 cut invitations (.50/.67 → .17) without lowering counts.
- Claim revised: "flagging part of a swarm raises counted voices" → "with wording fixed, a few flags (detector or random) raise counted voices; flags covering most of the swarm lower them". Gemini comparison still against note-only (different wording), stated as such.

**E3, Gemini missingness.** oc2 voices 112/128 valid; challenger held-out 4–7/8 per condition. Complete-case (9 + 14 contexts): baseline asymmetry +10.3 [6.7, 13.9] (unchanged); collapse asymmetry +1.5 [−1.7, 4.9] → +3.2 [0.1, 6.8] (not robust; disclosed in App. A).

**E4.** `src/report_numbers.py` regenerates all report numbers incl. paired CIs (k3−k1, chain−count-only, collapse−annotate), keyword counts, valid denominators.
- New paired collapse − annotate (Gemma): S4 poll −0.61 [−0.83, −0.33] @5%, −0.83 [−0.94, −0.72] @tea; S5 invite +0.28 [0.06, 0.50], inflation +2.8 [1.6, 4.1] @5%.

**Text fixes (v10):** abstract adds Gemma poll unchanged under chaining, non-specific flag, one-post swarm 33%; pitch rate corrected (37–64% of messages carry the pitch, not ~25%); note wording described as a false-positive ceiling supplied to the model; Fig. 3 = operating-point sweep with alert-rate gaps; Fig. 1 axis "reported − ref."; Table 1 adds k=30 rows, drops renamed-10 rows (kept in text); ln k slope sentence cut; §4.2 retitled; Gemini k CI; sprint marker at §4 start; Related Work paragraphs; "fixes the poll decision", "debate stage", "every threshold" removed; voices 9–10 / 14–15; Future work moved to App. A.

## Eighth review round (v10: Fable 4/4/3, Codex 4/4b/3) → v11, text only
- Fixed: k=30 b3k contrasts are at 5% (2–3 stray real flags, not 4); Fig. 1 caption (+21.2 tea vs +21.3 @5% vs +16.9 5%-wording zero-flag); App. A Gemini valid answers 3–7/8.
- Table 2: the 0.5% rows moved to App. C; added 5%-wording zero-flag and random-flag rows (Gemma).
- Voice inflation demoted to secondary metric (Gemma's count sits at 10–15 for references 1, 24, 30).
- §4.2 retitled "Assessment without action, and a same-call fix"; contributions stripped of numbers; Discussion adds theory of change (deployers, detector operators).
- Softened: "report coordination they do not act on" → separate call identifies / decision call recommends; detector-vs-random "as much" → unresolved; S5 "looks solicited" (D4-only) → low recall, flags add nothing beyond the warning text; abstract "misses" → flags under a quarter / none; "once retrieved"; report_numbers.py scope stated.
- Added: k=30 Gemma coordination flag .94 → .17–.28 under collapse.

## Ninth review round (v11, final check) → v12 (14 Nov)
Fable 4/4/3 and Codex gpt-6-astra 4/4(borderline)/3; both: no submission blockers, all 16 v10 fixes verified, no number disagrees with the aggregates. Applied: S4-only poll claim in Discussion; §4.2 residual gap may be content, length filtering or root-post presentation (not "is content"); S5 "flags add nothing" → no incremental benefit resolved; Table 2 caption (flags "in that condition"; random flags drawn from all accounts, about half swarm in practice); §4 preamble marks paired CIs and baseline asymmetries as sprint work; benign voices 13.7–14.9; "0/42" → 0%; abstract "retrieved messages" (word-neutral); §4.3 flag paragraph split, detector−random numbers → "CIs include 0"; Table 1 caption shortened; Discussion/Fig. 3 caption trimmed to keep the body on 8 pages. Release: report_numbers.py docstring scope, b1_matched_rate.py interpolation renamed `interp_in_domain_hypothetical`, prep_log status note + 81–132 erratum, README "briefing aggregate", CJK font path no longer user-specific, Old Standard fonts bundled.

## Tenth review round (v12) → v13 (14 Nov)
Fable 4/4/3, Codex 4/4b/3; no blockers; ~120 numbers re-verified. Applied: abstract "cuts orchestrated swarms' poll listings" (was "stops ... invitations": random flags already cut S4 invites to .17; detector − random is −0.17 on invitations, −0.78 [−1.00, −0.44] on poll listing under collapse); Gemini baseline-asymmetry CI upper bound 12.8 (was 12.7; 12.754); Discussion deployer advice now "test ... against benign-support controls before adopting"; Related Work "test no-swarm and stance-composition effects"; App. B "operating-point sweep"; Table 1 Gemini reps 2–3 marked sprint; Table 2 controls "Gemma only"; §4.2 "9–12 for a swarm under the same chained call"; abstract 248 words; App. C: 5% shares the D4c cut, so the nudged tied accounts are among its organic flags. New src/b1_tie_blocks.py reproduces the 821/476 (dev) and 672/414 (held-out) tie blocks (dev 821-block: all challenger supporters, 83 also incumbent supporters). Discussion/Conclusion trimmed to keep the body on 8 pages.

## Sprint (14 Nov): four more briefing models, core experiments (no defense), 378 jobs each
Qwen3.5-35B-A3B-FP8 (H200 GPU1, vLLM, thinking off), gpt-6-luna (Codex CLI), gpt-oss-120b-medium (agy), Claude Haiku 5.5 (Claude Code CLI, our system prompt, no tools). The CLI backends do not control temperature, and Codex adds its own agent instructions. Parsed: qwen 378/378, luna 378/378, oss 373/378, haiku 377/378.
invite / poll (6 clusters x 3 reps):
| model | S4 two-call | S5 two-call | k=1 | k=30 | S4 chained | S5 chained | S4 count-only | renamed30 two-call | renamed30 chained |
| gemma | .83/1.00 | 1.00/1.00 | .28/.50 | .78/1.00 | .06/1.00 | 0/.67 | 1.00/1.00 | 1.00/1.00 | .94/1.00 |
| qwen | .11/.22 | 0/.33 | 0/.33 | 0/.33 | 0/0 | 0/0 | .39/.61 | .22/.39 | 0/0 |
| oss | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | .94/1 | 1/1 | 1/1 | 1/1 |
| luna | 0/.94 | 0/1.00 | 0/.78 | 0/1.00 | 0/.78 | 0/.94 | 0/.94 | .17/1.00 | 0/1.00 |
| haiku | 0/.94 | 0/1.00 | 0/.61 | 0/1.00 | 0/1.00 | 0/1.00 | 0/1.00 | .06/1.00 | 0/1.00 |
Independent voices (two-call voices call), same 30 texts on 30 accounts (k=30) vs 30 renamed real accounts: gemma 11.9 vs 14.9 (21.6 is the chained k=30 count, not comparable); qwen 16.8 vs 8.9; oss 27.0 vs 19.4; luna 2.1 vs 21.6; haiku 3.4 vs 17.8.
Paired poll k=30 − k=1: luna +0.22 [0.06, 0.44], haiku +0.39 [0.17, 0.67], qwen 0.00, oss at ceiling (momentum +0.44 [0.17, 0.72]).
Chained (S4+S5) − count-only: qwen invite −0.56 [−0.75, −0.36], poll −0.67 [−0.83, −0.47] but renamed-30 also 0/0 (non-selective); oss −0.03 invite (no effect); luna poll −0.11 [−0.25, 0]; haiku ~0.
Chained k=1 vs k=30 (invite/poll): oss .17/.72 vs 1.00/1.00; qwen 0/0 vs 0/.11; luna 0/.50 vs 0/.83; haiku 0/.56 vs 0/.94.

## Codex review of v14 → v15 (14 Nov)
Codex 4/4b/3, no blockers; Table 1 and abstract numbers verified. Applied: open/closed split framed as "in this six-model sample" with an App. A interface disclosure (agy/Codex flatten the system prompt; no CLI token budget; not an effect of openness); §4.1 corrected (Gemma counts the swarm slightly fewer voices than real support, 11.9 vs 14.9); "four of six" scoped to k=1→30 everywhere; Qwen "equal endpoints" (poll .33, .17 at k=3); gpt-oss "does not suppress natural-swarm decisions" (chained k=1 drops to .17/.72); Luna/Haiku chained change stated vs two calls (−0.11 [−0.25, 0.03], +0.03); Table 1 caption available-case (three cells with 17 parsed).

## v16 (14 Nov, sprint): presentation rewrite + Gemini benign control and k reps 2-3
- Gemini k-ablation now 3 reps: P(poll) k=1 0/18, k=3 0/18, k=10 .24 (17 parsed), k=30 18/18; k30-k1 poll +1.00 [1.00, 1.00], 6/6 clusters. Invite 0 everywhere. Voices 1 at every k.
- Gemini benign control (3 reps, 108 jobs): renamed30 two calls invite .94 / poll 1.00 (voices 20.3, 15 valid); count-only 1.00/1.00; chained independence .28/.89.
  chained - two calls, renamed30: invite -0.67 [-0.89, -0.44] (6/6 clusters down), poll -0.11 [-0.33, 0]. renamed10: invite 0 throughout, poll 1.00 -> .89.
  => For Gemini the same-call assessment is selective on poll listing (swarm .06/.17 vs real .89) but costs real support its invitation, which no swarm ever got from Gemini.
- report_numbers.py: added `<model>|renamed{n}|chain_minus_twocall` paired contrasts for all models (Qwen renamed30: invite -0.22 [-0.33, -0.11], poll -0.39 [-0.56, -0.22]).
- Rewrite: abstract = three findings; new Fig. 1 (six models); §4.2 reordered (intervention -> matched control -> model failures -> residual cost); Table 2 slimmed (momentum/voices moved to aggregates); Discussion = Finding / Fix and price / What to do (four checks); overcorrection figure -> App. C; Fig. 4 shades the challenger's tie block.
- Review fixes: 1->30 endpoint wording; k>=10 rise "not resolved"; App. A "qualitatively unchanged" (+1.78 -> +1.94); "different briefing models" (vendors); one invalid Qwen voices answer; rep agreement (all 3 reps agree, no-defense contexts): poll Gemma .77 Qwen .64 oss .91 Luna .79 Haiku .74; invite 81-98%.

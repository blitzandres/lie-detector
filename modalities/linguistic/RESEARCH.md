# Linguistic Modality — Deep Research

> Blitz Engine | Linguistic / NLP layer | Written June 15, 2026
> Companion to `docs/CUE_CATALOG.md` (cues 28–37, 47–54) and `planning/ACCURACY_PLAN.md`.
> Scope: the verbal/text channel only. Grounds the linguistic cues in the published deception
> literature, separates **baseline-dependent** from **person-independent** signals, and specifies
> how the linguistic layer behaves in the **no-prior-baseline** case (e.g. analyzing a podcast guest).

---

## 0. Bottom line up front

- The **verbal channel is the single most diagnostic modality** in deception research — more so than
  facial or gross-body cues. Synthesis reviews repeatedly find verbal content beats nonverbal "tells."
- But effect sizes for *individual* linguistic cues are **small** (median g ≈ 0.26). No word, ratio, or
  pronoun is a lie detector by itself. Power comes from **fusing many weak cues** — exactly the engine's design.
- The strongest results come from **content-richness frameworks** (Reality Monitoring, CBCA, Verifiability
  Approach) and from **NLP models trained on many features**, not from single hand-picked markers.
- **Modern ML beats humans**: on the same 62-interview set, humans scored 54.7% (naïve) / 59.4% (expert),
  Reality-Monitoring+cognitive-load ML scored **69.4%**, and full NLP feature extraction scored **77.3%**
  (Loconte et al. 2025). This is our north star for the linguistic layer in isolation.
- **The no-baseline / podcast case is the hard case.** Most cues above are *person-relative* (they need
  to know how *this* speaker talks when relaxed). The realistic design is a **rolling self-baseline** built
  from the opening minutes of the recording, plus a small set of **person-independent** cues. See §4.

---

## 1. Why the verbal channel, and why it is still weak

The myth is the body ("liars look away, liars fidget"). The data says the opposite: nonverbal cues are
near-useless, and **verbal content carries the signal**. DePaulo et al. (2003), the canonical cue
meta-analysis, found most behavioral cues have effect sizes near zero; the few that survive are verbal
(fewer details, less compelling/plausible accounts, more discrepancies).

The catch is the **size** of the effect. Hauch et al. (2015), meta-analyzing computer-scored linguistic
cues across 44 studies, found a **median |g| ≈ 0.26** — statistically real, practically tiny. That is the
honest ceiling for any single word-count cue. The engineering consequence is not "give up" but
**"never trust one cue — fuse"**, which is why the Bayesian log-odds fusion layer exists.

**Direction of the reliable category-level effects** (Hauch 2015; Newman/Pennebaker tradition).
Relative to truth-tellers, **liars tend to use**:

| Category | Direction in liars | Interpretation |
|---|---|---|
| First-person singular ("I", "me", "my") | **fewer** | psychological distancing from the false claim |
| Self- and other-references | **fewer** | reduced ownership / vague actors |
| Exclusive words ("but", "except", "without") | **fewer** | fabricated stories are simpler, less qualified |
| Cognitive-complexity / time words | **fewer** | thinner temporal structure |
| Negations ("no", "not", "never") | **more** | denial framing |
| Negative-emotion words | **more** | guilt/anxiety leakage |
| Motion / space words | **more** | over-describing scene to sound concrete |
| Total sensory/perceptual detail | **fewer** | the core Reality-Monitoring signal |

These are population tendencies, not individual verdicts. A talkative, emotional truth-teller will trip
several of them. This is the entire reason the engine requires a baseline (§4).

---

## 2. The four content frameworks (ranked by strength)

The linguistic layer is built on four research traditions. Ranked roughly by evidential strength:

### 2.1 Reality Monitoring (RM) — strongest practical framework
Premise (Johnson & Raye): *real* memories are encoded through perception, so true accounts carry more
**perceptual detail** (sight, sound, smell, taste, touch), **spatial detail**, and **temporal/contextual
detail**; **fabricated** accounts carry more **cognitive operations** (reasoning, justifications,
inferences) inserted to fill gaps the speaker never actually experienced.
- Cognitive-operations density is an **inverted** cue: *more* reasoning words → *more* suspicious (cue 48).
- In Loconte et al. (2025), RM+cognitive-load features drove ML to **69.4%** vs ≤59.4% for humans.
- Maps to engine cues **31** (sensory poverty), **48** (cognitive ops), **35** (negative emotion).

### 2.2 CBCA / Statement Validity Analysis — strong in lab, modest in field
Criteria-Based Content Analysis scores 19 content criteria (logical structure, quantity of detail,
contextual embedding, **complications**, reproduction of conversation, spontaneous corrections,
admitting lack of memory, etc.). Truthful statements score higher.
- Lab meta-analysis: large overall effect **g ≈ 1.03** [0.78, 1.27] (content-based techniques).
- **Field meta-analysis is more sober** — individual criteria help but the global effect shrinks outside
  the lab. Treat CBCA criteria as **useful features, not a courtroom test**.
- Best individual criteria for us (medium effect, automatable): **complications** ("then my car wouldn't
  start"), **spontaneous corrections**, **reproduction of conversation / direct quotes**, **unusual peripheral
  details**. Maps to engine cues **54, 47, 51**.

### 2.3 Verifiability Approach (VA) — the most "podcast-friendly" framework
Premise (Nahari/Vrij): truth-tellers volunteer **checkable** details (named people, places, times,
documented actions — "I paid by card at the Shell on 5th at about 6pm"); liars prefer **unverifiable**
detail ("I was just driving around"). Score = verifiable details ÷ total details.
- Meta-analysis: truth-tellers reliably give **more verifiable details and a higher ratio**. (The mirror
  claim — that liars pile on *un*verifiable detail — did **not** hold.)
- Accuracy is context-dependent: ~56% on individual statements, up to ~79% on collective/strong cases.
- **Why it matters here:** verifiability is **closer to person-independent** than pronoun rates — it's
  about *what kind of facts* are offered, not *how this person habitually talks*. Strong candidate for the
  no-baseline podcast path. Maps to engine cue **37** (verifiable entity poverty) — recommend promoting it
  to a full **verifiable-detail ratio**, not just NER entity count.

### 2.4 LIWC / cognitive-load lexical style — the classic, now partly blocked
The Newman/Pennebaker line (pronouns, emotion, cognitive words). Still useful, but **LIWC-22 is
proprietary/paid** — a known blocker in this repo. Open substitutes:
- **Empath** (MIT) and **NRCLex** for emotion/category lexicons,
- **spaCy** POS/dependency for pronoun & exclusive-word rates,
- **TAALED / lexicalrichness** for lexical diversity (MTLD),
- **TextDescriptives** for readability/coherence.
We reproduce the *useful* LIWC categories (pronouns, negations, exclusives, emotion) without the license.

---

## 3. Cue-by-cue evidence map (what's real, how strong, baseline need)

`B?` = does the cue need a personal baseline to be trustworthy?
`Strength` = rough single-cue diagnosticity from the literature (not the fused score).

| # | Cue | Framework | Direction (liars) | Strength | B? | Open library |
|---|---|---|---|---|---|---|
| 28 | Pre-answer pause | cognitive load | longer (constructing) | low–med | **yes** | Whisper/WhisperX timestamps |
| 29 | Distancing language | RM/LIWC | more "that/them", less "I" | low | **yes** | spaCy POS |
| 30 | Qualifier / hedge overload | cognitive load | more "I think/maybe" | low | **yes** | spaCy Matcher + hedge lexicon |
| 31 | Sensory-detail poverty | **RM** | fewer perceptual details | **med** | partial | sensory lexicon + spaCy |
| 32 | Tense inconsistency | cognitive load | present↔past slips | low | yes | spaCy morphology |
| 33 | Pronoun avoidance | LIWC | fewer 1st-person sing. | low–med | **yes** | spaCy POS |
| 34 | Abnormal answer length | cognitive load | too short / over-elaborated | low | **yes** | word count |
| 35 | Negative-emotion density | LIWC/RM | more neg-emotion words | low | partial | NRCLex / Empath / VADER |
| 36 | Narrative coherence drop | RM/discourse | abrupt semantic jumps | med | partial | TextDescriptives / embeddings |
| 37 | **Verifiable-detail ratio** | **VA** | fewer checkable facts | **med** | **closer to no** | spaCy NER + rules |
| 47 | Spontaneous corrections | **CBCA** | fewer (fixed script) | **med (72–74%)** | partial | disfluency/regex |
| 48 | Cognitive-operations density | **RM (inverted)** | **more** reasoning words | med (69%) | partial | lexicon + dep parse |
| 49 | Lexical diversity (MTLD) | cognitive load | lower (repetition) | low–med | yes | lexicalrichness / taaled |
| 50 | Syntactic tree depth | cognitive load | simpler sentences | low–med | yes | spaCy / stanza |
| 51 | Direct-quote ratio | **CBCA** | fewer exact quotes | med | partial | spaCy + regex |
| 52 | Narrative proportion imbalance | CBCA | thin core, padded setup | low | yes | temporal tagger |
| 53 | Internal-contradiction score | NLI | more self-contradiction | **high signal** | **no** | bart-large-mnli / deberta-nli |
| 54 | Complication rate | **CBCA** | fewer obstacles described | med (g≈0.5) | partial | adversative markers + spaCy |

**Reading the table:** the cues that are both **(a) medium-strength** and **(b) least baseline-dependent**
are **37 (verifiability), 53 (internal contradiction), 48 (cognitive ops), 54 (complications), 51 (quotes)**.
Those are the backbone of the no-baseline path.

---

## 4. The no-baseline / podcast case (the question that started this)

**Can the linguistic layer work on a podcast guest we've never seen?** Partially — with the right design.

The problem: most cues in §3 are **person-relative**. "Low first-person pronoun rate" only means something
against *this speaker's* normal rate. Absolute thresholds turn every laconic or anxious person into a "liar."
With **no enrollment baseline**, naive absolute scoring collapses toward chance.

Two mechanisms recover most of the signal:

### 4.1 Rolling self-baseline (primary)
Treat the **first 90–180 s of uncontested, low-stakes talk** (intro, rapport, easy questions) as an
*in-session* baseline, then score later segments as **robust-Z deviations** (median/MAD, per the engine's
normalization rule) from that running profile. This is the same calibration math the engine already uses —
just sourced live from the recording instead of an enrollment clip. Works for pause length, pronoun rate,
hedging, lexical diversity, complexity — the baseline-dependent cues become usable again.
- Caveat: the opening must actually be truthful/relaxed. If the guest is adversarial from second zero,
  the self-baseline is contaminated. Flag low baseline confidence and widen uncertainty (don't fake a verdict).

### 4.2 Person-independent cues (secondary, always-on)
A subset of cues compares the speaker **to the structure of truth itself**, not to their own habits:
- **Verifiability ratio (37)** — are claims checkable? Independent of speaking style.
- **Internal contradiction (53)** — NLI between the guest's own statements; pure consistency check.
- **Cognitive-operations density (48)** and **complication rate (54)** — content properties of the story.
- **Cross-claim / external fact consistency** — does claim X contradict claim Y, or a known fact?
  (This is closer to *fact-checking* than *lie detection*, and it is the most defensible thing we ship.)

### 4.3 Honest framing for this mode
Without enrollment, the linguistic layer should output **"elevated deception risk / anomaly"**, never a
binary "LIE." Expected ceiling in no-baseline mode: roughly **60–68%** (person-independent cues + rolling
baseline), below the **70–77%** achievable with a clean personal baseline. State the mode and the
uncertainty in the output. This matches the engine's two-gate convergence rule — a linguistic-only,
no-baseline flag must **not** fire a verdict alone.

---

## 5. Modern NLP layer (transformers) — power and trap

Beyond hand-built cues, end-to-end models are the current SOTA on the verbal channel:
- Loconte et al. (2025): full NLP feature extraction reached **77.3%** vs 69.4% for RM/cognitive-load and
  ≤59.4% for humans, on real high-stakes interviews.
- Fine-tuned **BERT** hit **93.6%** on the Ott deceptive-opinion-spam corpus — but that is **fake reviews**,
  not interpersonal lying. **Do not quote that number for our use case.**

**The trap — domain shift and a truth bias.** A 2025 cross-lingual study showed BERT tends to equate
*plausible-sounding* language with *truthful* intent (it was pretrained on mostly-true text), so it fails on
fluent liars and transfers badly across datasets/languages. This mirrors the engine's #1 known risk
(cross-domain accuracy collapsing to ~62%, per the SVC-2025 finding already in the spec).

**Implications for our build:**
1. Use transformers as a **feature/score producer fused with the hand cues**, not as a standalone verdict.
2. Keep an **NLI model (53)** for contradiction — that generalizes far better than learned "liar style."
3. Always **report which corpus a model was trained on**; never let a review-trained model imply courtroom accuracy.
4. Hold to the engine's fairness audit — lexical-style models can encode dialect/register, not deception.

---

## 6. Implementation notes (free-tools-only, per project rules)

| Need | Open library | License | Note |
|---|---|---|---|
| Tokenize / POS / dep / NER / morphology | **spaCy** (`en_core_web_trf`) | MIT | workhorse for cues 29,32,33,37,48,50,51,54 |
| Word timestamps + pauses | **WhisperX** (BSD-2) / CrisperWhisper (CC-BY-NC) | mixed | CrisperWhisper adds `[UH]/[UM]` fillers but is non-commercial → see RESEARCH BLOCKER 1 |
| Emotion / category lexicon | **NRCLex**, **Empath** | MIT-ish | LIWC-22 substitute (LIWC is paid — blocked) |
| Sentiment | **VADER** | MIT | quick neg-emotion proxy (cue 35) |
| Lexical diversity (MTLD) | **lexicalrichness**, **TAALED** | MIT | cue 49 |
| Readability / coherence | **TextDescriptives** | MIT/Apache | cue 36 |
| Internal contradiction (NLI) | **bart-large-mnli**, **deberta-v3-nli** | MIT/Apache | cue 53 — person-independent ⭐ |
| Holistic RM/CBCA scoring + verdict | **Claude API** | tokens | cross-answer consistency, framework scoring (allowed cost) |

**Output contract reminder:** every linguistic cue emits a `CueEvent` with value, robust-Z vs baseline,
direction `d_i`, weight `w_i`, and a **baseline-confidence** flag. In no-baseline mode, baseline-dependent
cues report reduced confidence and the fusion layer down-weights them accordingly.

---

## 7. Honest accuracy expectations (linguistic layer, isolated)

| Condition | Expected accuracy | Source/analog |
|---|---|---|
| Humans (naïve) | 54.7% | Loconte 2025 |
| Humans (expert) | 59.4% | Loconte 2025 |
| Single linguistic cue | ~55% (g≈0.26) | Hauch 2015 |
| RM + cognitive-load ML | 69.4% | Loconte 2025 |
| Full NLP feature extraction (clean domain) | 77.3% | Loconte 2025 |
| **Blitz linguistic + personal baseline** | **70–77% target** | engine fusion goal |
| **Blitz linguistic, NO baseline (podcast)** | **~60–68%** | person-independent + rolling baseline |
| Cross-domain / fluent-liar worst case | ~62% | SVC-2025 + 2025 BERT study |

Linguistic is the **strongest single modality** but still must fuse with audio/visual/physiological and pass
the two-gate convergence rule before the engine emits any verdict.

---

## 8. References

- DePaulo, Lindsay, Malone, et al. (2003) — *Cues to deception.* Psych. Bulletin. PMID 12555795.
- Bond & DePaulo (2006) — *Accuracy of deception judgments.* PSPR. PMID 16859438.
- Newman, Pennebaker, Berry, Richards (2003) — *Lying words: predicting deception from linguistic styles.*
- Hauch, Blandón-Gitlin, Masip, Sporer (2015) — *Are Computers Effective Lie Detectors? A Meta-Analysis of
  Linguistic Cues to Deception.* PSPR. doi:10.1177/1088868314556539 (median |g|≈0.26).
- Amado, Arce, Fariña (CBCA meta-analyses) & field-study meta-analysis, *European Psychologist*,
  doi:10.1027/1016-9040/a000561.
- Nahari, Vrij, Fisher — Verifiability Approach; *The Verifiability Approach: A Meta-Analysis*,
  J. Applied Research in Memory & Cognition, doi:10.1016/j.jarmac.2020.08.001.
- Verschuere et al. (2021) — preregistered VA replication. Applied Cognitive Psychology, doi:10.1002/acp.3769.
- Loconte, Battaglini, Maldera, et al. (2025) — *Detecting Deception Through Linguistic Cues: From Reality
  Monitoring to NLP.* J. Language & Social Psychology. doi:10.1177/0261927X251316883
  (humans 54.7/59.4% · RM+load 69.4% · NLP 77.3%).
- Kennedy/Singh et al. (2020) — *Building a Better Lie Detector with BERT* (Ott corpus 93.6% — review domain).
- (2025) *What if Deception Cannot be Detected? A Cross-Linguistic Study on the Limits of Deception Detection
  from Text.* arXiv:2505.13147 (BERT plausibility/truth bias; domain-transfer failure).

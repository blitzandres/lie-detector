# Content Engine (Content-First Q&A) — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the content-first layer — a Q&A "judge a turn" path where a local LLM (Ollama) scores the *content* of an answer (consistency · Reality-Monitoring richness · verifiability · relevance) and fuses it (content-primary) with the cue activity from that answer's exact time window.

**Architecture:** A new `blitz_overlay/content/` package: a `ContentJudge` seam (ABC + `ContentVerdict`), a deterministic `StubContentJudge` (so tests never hit a live LLM), and an `OllamaContentJudge` adapter (`localhost:11434`, prompt + robust JSON parse + offline fallback). The session keeps a `TimelineBuffer` of per-frame cue activity; `session.judge_turn(question, answer, t0, t1)` runs the judge, pulls the cue window, and fuses content-primary. The server WS branches on `type:"turn"`. Browser gets a Q&A panel + True/False dev scripts + a calibration reading text.

**Tech Stack:** Python 3.14 (`python3`, no venv), pytest, ruff, httpx (already installed) for the Ollama HTTP call; vanilla ES-module browser app.

**Reference spec:** `docs/superpowers/specs/2026-06-19-content-first-qa-architecture-design.md`. Verify after each task: `python3 -m pytest -q` + `python3 -m ruff check .` green. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

**Honest boundary (LOCKED):** content-*pattern* analysis only (never factual truth-checking, never "LIE"). Reading a script ≠ spontaneous lying — the True/False scripts validate analyzer *mechanics*, not accuracy.

---

## File Structure

- Create `blitz_overlay/content/__init__.py`
- Create `blitz_overlay/content/judge.py` — `ContentVerdict`, `ContentJudge` (ABC), `StubContentJudge`.
- Create `blitz_overlay/content/ollama_judge.py` — `OllamaContentJudge` (HTTP + prompt + JSON parse + offline).
- Create `blitz_overlay/content/timeline.py` — `TimelineBuffer` (per-frame cue activity, `window(t0,t1)` summary).
- Create `blitz_overlay/content/fusion.py` — `fuse_turn(verdict, cue_window)` → content-primary turn result.
- Modify `blitz_overlay/pipeline.py` — buffer the timeline each frame; add `judge_turn(...)`.
- Modify `blitz_overlay/server.py` — WS branches on `type:"turn"`.
- Create `apps/overlay-web/js/qa-panel.js` — question/answer UI + verdict render.
- Create `apps/overlay-web/scripts/dev-scripts.js` — calibration reading text + True/False narratives.
- Modify `apps/overlay-web/index.html`, `css/overlay.css`, `js/main.js` — mount the Q&A panel.

---

## Task 1: ContentVerdict + ContentJudge seam + StubContentJudge

**Files:**
- Create: `blitz_overlay/content/__init__.py` (empty)
- Create: `blitz_overlay/content/judge.py`
- Test: `tests/overlay/test_content_judge.py`

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_content_judge.py`:

```python
"""ContentVerdict shape + the deterministic StubContentJudge used everywhere in tests."""
from blitz_overlay.content.judge import ContentJudge, ContentVerdict, StubContentJudge


def test_verdict_to_dict_round_trips():
    v = ContentVerdict(
        risk=0.7,
        scores={"consistency": 0.4, "richness_rm": 0.3, "verifiability": 0.2, "relevance": 0.6},
        flagged_phrases=[{"text": "someone took it", "reason": "vague, unverifiable"}],
        rationale="thin account",
        available=True,
    )
    d = v.to_dict()
    assert d["risk"] == 0.7
    assert d["scores"]["consistency"] == 0.4
    assert d["flagged_phrases"][0]["text"] == "someone took it"
    assert d["available"] is True


def test_stub_is_a_content_judge():
    assert isinstance(StubContentJudge(), ContentJudge)


def test_stub_scores_vague_answer_riskier_than_concrete():
    """Deterministic heuristic: more vague/hedge words + fewer concrete tokens => higher risk.

    This lets us test fusion + the True/False discrimination without a live LLM.
    """
    stub = StubContentJudge()
    concrete = stub.judge("Where were you at 9pm?",
                          "I was at Mario's Pizza on 5th Street with my sister Anna until 10.",
                          history=[], baseline=None)
    vague = stub.judge("Where were you at 9pm?",
                       "I was just somewhere around, you know, with some people I think.",
                       history=[], baseline=None)
    assert vague.risk > concrete.risk
    assert vague.available is True


def test_stub_empty_answer_low_confidence():
    v = StubContentJudge().judge("Q?", "", history=[], baseline=None)
    assert v.available is True
    assert 0.0 <= v.risk <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_content_judge.py -q`
Expected: FAIL — `ModuleNotFoundError: blitz_overlay.content`.

- [ ] **Step 3: Implement** — create `blitz_overlay/content/__init__.py` (empty file) and `blitz_overlay/content/judge.py`:

```python
"""Content-analysis seam: the primary (content-first) deception layer.

A ContentJudge scores the *content pattern* of an answer — consistency, Reality-Monitoring
richness, verifiability, relevance — NEVER factual truth. StubContentJudge is a deterministic
heuristic so the rest of the system (fusion, turns, tests) never depends on a live LLM.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[A-Za-z']+")
# Words that signal vagueness / hedging (content-poverty markers).
_VAGUE = {
    "someone", "something", "somewhere", "somehow", "stuff", "things", "people",
    "around", "maybe", "guess", "think", "probably", "kind", "sort", "whatever",
    "anyway", "just", "like", "you", "know",
}


@dataclass
class ContentVerdict:
    """One content judgment of an answer. risk/scores in [0,1]. available=False => layer offline."""

    risk: float
    scores: dict[str, float]
    flagged_phrases: list[dict] = field(default_factory=list)
    rationale: str = ""
    available: bool = True

    def to_dict(self) -> dict:
        return {
            "risk": round(self.risk, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "flagged_phrases": self.flagged_phrases,
            "rationale": self.rationale,
            "available": self.available,
        }

    @classmethod
    def offline(cls, reason: str) -> ContentVerdict:
        return cls(risk=0.0, scores={}, flagged_phrases=[], rationale=reason, available=False)


class ContentJudge(ABC):
    @abstractmethod
    def judge(self, question: str, answer: str, history: list[dict], baseline) -> ContentVerdict:
        """Score the content of `answer` (given `question` for context)."""


class StubContentJudge(ContentJudge):
    """Deterministic content heuristic — no network. Vagueness/hedging up, concrete detail down."""

    def judge(self, question: str, answer: str, history: list[dict], baseline) -> ContentVerdict:
        tokens = _TOKEN.findall(answer.lower())
        n = max(1, len(tokens))
        vague = sum(1 for t in tokens if t in _VAGUE)
        # concrete = numbers + capitalized proper nouns (verifiable detail)
        concrete = len(re.findall(r"\d", answer)) + sum(1 for w in _TOKEN.findall(answer) if w[:1].isupper())

        vague_ratio = vague / n
        verifiability = max(0.0, min(1.0, concrete / max(4, n * 0.15)))
        richness_rm = verifiability  # proxy: concrete detail ~ RM richness for the stub
        relevance = 1.0 if tokens else 0.0
        consistency = 1.0  # the stub has no cross-answer memory

        # risk: high when vague & low verifiability. [0,1].
        risk = max(0.0, min(1.0, 0.6 * vague_ratio + 0.5 * (1.0 - verifiability)))
        flagged = []
        for w in _TOKEN.findall(answer):
            if w.lower() in _VAGUE and len(flagged) < 5:
                flagged.append({"text": w, "reason": "vague / hedging marker"})
        return ContentVerdict(
            risk=risk,
            scores={"consistency": consistency, "richness_rm": richness_rm,
                    "verifiability": verifiability, "relevance": relevance},
            flagged_phrases=flagged,
            rationale="stub heuristic (vagueness vs concrete detail)",
            available=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_content_judge.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/content/__init__.py blitz_overlay/content/judge.py tests/overlay/test_content_judge.py
git commit -m "feat(content): ContentJudge seam + ContentVerdict + deterministic StubContentJudge

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: TimelineBuffer (server-side cue rhythm, window query)

**Files:**
- Create: `blitz_overlay/content/timeline.py`
- Test: `tests/overlay/test_timeline_buffer.py`

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_timeline_buffer.py`:

```python
"""TimelineBuffer keeps per-frame cue activity so a turn can pull its [t0,t1] window."""
from blitz_overlay.content.timeline import TimelineBuffer


def test_window_summarizes_lit_cues_in_range():
    tl = TimelineBuffer(retain_ms=60000)
    tl.add(1000, [("visual.gaze_aversion", "visual", 3.0)])
    tl.add(1200, [("visual.gaze_aversion", "visual", 3.5), ("audio.tremor", "audio", 2.2)])
    tl.add(5000, [("visual.blink_rate", "visual", 4.0)])  # outside the window below

    w = tl.window(900, 1500)
    assert w["n_frames"] == 2
    assert set(w["families"]) == {"visual", "audio"}
    assert "visual.gaze_aversion" in w["cue_ids"]
    assert w["peak_z"] == 3.5
    assert w["max_families_synchronous"] >= 2  # both families lit within the window


def test_window_empty_when_no_frames_in_range():
    tl = TimelineBuffer(retain_ms=60000)
    tl.add(1000, [("visual.gaze_aversion", "visual", 3.0)])
    w = tl.window(8000, 9000)
    assert w["n_frames"] == 0
    assert w["families"] == []
    assert w["peak_z"] == 0.0


def test_retention_drops_old_frames():
    tl = TimelineBuffer(retain_ms=1000)
    tl.add(0, [("visual.gaze_aversion", "visual", 3.0)])
    tl.add(2000, [("audio.tremor", "audio", 3.0)])  # 2s later -> first frame evicted
    assert all(ts >= 1000 for ts, _ in tl._frames)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_timeline_buffer.py -q`
Expected: FAIL — `ModuleNotFoundError: blitz_overlay.content.timeline`.

- [ ] **Step 3: Implement** — create `blitz_overlay/content/timeline.py`:

```python
"""Rolling buffer of per-frame cue activity so the content engine can pull a turn's window.

Each frame stores the cues that were "lit" (directed z >= the lit threshold) at that moment.
window(t0, t1) summarizes the behavioural rhythm during an answer for content↔cue fusion.
"""
from __future__ import annotations

from collections import deque

LIT_Z = 2.0  # a cue is "lit" in the timeline at directed z >= this


class TimelineBuffer:
    def __init__(self, retain_ms: int = 120000):
        self.retain_ms = retain_ms
        self._frames: deque[tuple[int, list[tuple[str, str, float]]]] = deque()

    def add(self, ts: int, cue_levels: list[tuple[str, str, float]]) -> None:
        """cue_levels: (cue_id, family, directed_z) for measured cues this frame."""
        lit = [(cid, fam, z) for (cid, fam, z) in cue_levels if z >= LIT_Z]
        self._frames.append((ts, lit))
        cutoff = ts - self.retain_ms
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def window(self, t0: int, t1: int) -> dict:
        frames = [(ts, lit) for ts, lit in self._frames if t0 <= ts <= t1]
        cue_ids: set[str] = set()
        families: set[str] = set()
        peak_z = 0.0
        max_sync = 0
        for _, lit in frames:
            fam_here = set()
            for cid, fam, z in lit:
                cue_ids.add(cid)
                families.add(fam)
                fam_here.add(fam)
                peak_z = max(peak_z, z)
            max_sync = max(max_sync, len(fam_here))
        return {
            "n_frames": len(frames),
            "cue_ids": sorted(cue_ids),
            "families": sorted(families),
            "peak_z": round(peak_z, 3),
            "max_families_synchronous": max_sync,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_timeline_buffer.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/content/timeline.py tests/overlay/test_timeline_buffer.py
git commit -m "feat(content): TimelineBuffer — per-frame cue rhythm with [t0,t1] window query

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Content-primary fusion

**Files:**
- Create: `blitz_overlay/content/fusion.py`
- Test: `tests/overlay/test_content_fusion.py`

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_content_fusion.py`:

```python
"""Content-primary fusion: content drives the verdict; the cue window confirms."""
from blitz_overlay.content.fusion import fuse_turn
from blitz_overlay.content.judge import ContentVerdict


def _cue_window(families, peak_z=3.0, sync=2):
    return {"n_frames": 10, "cue_ids": [], "families": list(families),
            "peak_z": peak_z, "max_families_synchronous": sync}


def test_offline_content_falls_back_to_cue_only():
    res = fuse_turn(ContentVerdict.offline("ollama down"), _cue_window(["visual", "physio"]))
    assert res["content_available"] is False
    assert res["label"].startswith("cue-only")


def test_content_and_cue_agree_is_high_confidence():
    v = ContentVerdict(risk=0.8, scores={}, flagged_phrases=[], rationale="thin")
    res = fuse_turn(v, _cue_window(["visual", "audio"], sync=2))
    assert res["content_available"] is True
    assert res["combined"] >= 0.7
    assert res["convergence"] is True
    assert "converge" in res["label"].lower()


def test_content_high_but_cues_quiet_is_content_only_flag():
    v = ContentVerdict(risk=0.8, scores={}, flagged_phrases=[], rationale="thin")
    res = fuse_turn(v, _cue_window([], sync=0))
    assert res["convergence"] is False
    assert "content" in res["label"].lower()


def test_low_content_low_cue_is_clear():
    v = ContentVerdict(risk=0.1, scores={}, flagged_phrases=[], rationale="rich")
    res = fuse_turn(v, _cue_window([], sync=0))
    assert res["combined"] < 0.45
    assert res["label"].lower().startswith("clear")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_content_fusion.py -q`
Expected: FAIL — `ModuleNotFoundError: blitz_overlay.content.fusion`.

- [ ] **Step 3: Implement** — create `blitz_overlay/content/fusion.py`:

```python
"""Content-primary fusion of a content verdict with the cue activity from the same window.

Content drives the verdict; a synchronous multi-family cue burst in the answer's window
*confirms* it (convergence → higher confidence). Honest framing: never "LIE".
"""
from __future__ import annotations

WATCH = 0.45
HIGH = 0.65


def fuse_turn(verdict, cue_window: dict) -> dict:
    cue_sync = cue_window.get("max_families_synchronous", 0)
    cue_confirms = cue_sync >= 2  # ≥2 independent families lit together in the answer window

    if not verdict.available:
        # Content layer offline → cue-only honest fallback.
        risk = 0.6 if cue_confirms else 0.2
        return {
            "content_available": False,
            "combined": round(risk, 4),
            "convergence": False,
            "label": "cue-only (content layer offline)",
            "content_risk": 0.0,
            "cue_synchronous_families": cue_sync,
        }

    content_risk = float(verdict.risk)
    # Content is primary; convergence with cues adds confidence.
    combined = content_risk + (0.15 if cue_confirms and content_risk >= WATCH else 0.0)
    combined = max(0.0, min(1.0, combined))
    convergence = cue_confirms and content_risk >= WATCH

    if convergence:
        label = "CONVERGENCE — content + behavioural cues align (high confidence)"
    elif content_risk >= HIGH:
        label = "CONTENT flag — account thin / inconsistent / unverifiable"
    elif content_risk >= WATCH:
        label = "CONTENT watch — some content-pattern risk"
    else:
        label = "CLEAR — content reads truthful"

    return {
        "content_available": True,
        "combined": round(combined, 4),
        "convergence": convergence,
        "label": label,
        "content_risk": round(content_risk, 4),
        "cue_synchronous_families": cue_sync,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_content_fusion.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/content/fusion.py tests/overlay/test_content_fusion.py
git commit -m "feat(content): content-primary fusion (content verdict + cue-window confirmation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: OllamaContentJudge adapter (prompt + JSON parse + offline)

**Files:**
- Create: `blitz_overlay/content/ollama_judge.py`
- Test: `tests/overlay/test_ollama_judge.py`

The HTTP call is injected so tests never hit a live server.

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_ollama_judge.py`:

```python
"""OllamaContentJudge: prompt assembly, JSON parsing, and offline fallback (no live server)."""
from blitz_overlay.content.judge import ContentVerdict
from blitz_overlay.content.ollama_judge import OllamaContentJudge, parse_verdict_json


def test_parse_well_formed_json():
    raw = ('{"risk":0.72,"scores":{"consistency":0.5,"richness_rm":0.3,'
           '"verifiability":0.2,"relevance":0.8},'
           '"flagged_phrases":[{"text":"some guy","reason":"vague"}],"rationale":"thin"}')
    v = parse_verdict_json(raw)
    assert isinstance(v, ContentVerdict)
    assert v.risk == 0.72
    assert v.scores["consistency"] == 0.5
    assert v.flagged_phrases[0]["text"] == "some guy"
    assert v.available is True


def test_parse_json_embedded_in_prose():
    raw = "Sure! Here is the analysis:\n{\"risk\":0.4,\"scores\":{},\"rationale\":\"ok\"}\nHope that helps."
    v = parse_verdict_json(raw)
    assert v.risk == 0.4


def test_parse_garbage_returns_low_confidence_available():
    v = parse_verdict_json("the model rambled with no json at all")
    assert v.available is True
    assert v.risk == 0.0
    assert "unparseable" in v.rationale.lower()


def test_judge_uses_injected_caller():
    captured = {}

    def fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"risk":0.9,"scores":{"consistency":0.1},"rationale":"contradictory"}'

    judge = OllamaContentJudge(model="llama3.2:3b", call=fake_call)
    v = judge.judge("Where were you?", "I was home. I was at work.", history=[], baseline=None)
    assert v.risk == 0.9
    assert "Where were you?" in captured["prompt"]
    assert "I was home" in captured["prompt"]


def test_judge_offline_when_caller_raises():
    def boom(prompt: str) -> str:
        raise ConnectionError("ollama not running")

    judge = OllamaContentJudge(model="llama3.2:3b", call=boom)
    v = judge.judge("Q?", "A.", history=[], baseline=None)
    assert v.available is False
    assert "offline" in v.rationale.lower() or "ollama" in v.rationale.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_ollama_judge.py -q`
Expected: FAIL — `ModuleNotFoundError: blitz_overlay.content.ollama_judge`.

- [ ] **Step 3: Implement** — create `blitz_overlay/content/ollama_judge.py`:

```python
"""OllamaContentJudge — the local-LLM content adapter (localhost:11434).

The HTTP call is injected (`call`) so tests never need a live server. On any connection
error the verdict is `available=False` (the layer is honestly offline). The model is asked
for STRICT JSON; parsing is robust to prose wrappers and falls back to a low-confidence
available verdict on garbage.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from blitz_overlay.content.judge import ContentJudge, ContentVerdict

OLLAMA_URL = "http://localhost:11434/api/generate"

_RUBRIC = """You are a deception-PATTERN analyst. You do NOT decide if a statement is factually
true. You score only content patterns associated with deception, on a 0..1 scale each:
- consistency: 1.0 = no internal contradiction, 0.0 = contradicts itself / the question.
- richness_rm: 1.0 = rich sensory/contextual/peripheral detail (Reality Monitoring), 0.0 = thin.
- verifiability: 1.0 = many checkable names/places/times, 0.0 = vague/unverifiable.
- relevance: 1.0 = directly answers the question, 0.0 = evasive/off-topic.
Then risk: 0..1 overall deception-pattern risk (high = thin, inconsistent, unverifiable, evasive).
Reply with STRICT JSON only:
{"risk":0.0,"scores":{"consistency":0.0,"richness_rm":0.0,"verifiability":0.0,"relevance":0.0},
"flagged_phrases":[{"text":"...","reason":"..."}],"rationale":"one sentence"}"""


def build_prompt(question: str, answer: str, history: list[dict]) -> str:
    hist = "\n".join(f"Q: {h.get('question','')}\nA: {h.get('answer','')}" for h in (history or []))
    hist_block = f"\nEarlier in this interview:\n{hist}\n" if hist else ""
    return f"{_RUBRIC}\n{hist_block}\nQUESTION: {question}\nANSWER: {answer}\n\nJSON:"


def parse_verdict_json(raw: str) -> ContentVerdict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            scores = {k: float(v) for k, v in (data.get("scores") or {}).items()}
            return ContentVerdict(
                risk=float(data.get("risk", 0.0)),
                scores=scores,
                flagged_phrases=list(data.get("flagged_phrases") or []),
                rationale=str(data.get("rationale", "")),
                available=True,
            )
        except (ValueError, TypeError):
            pass
    return ContentVerdict(risk=0.0, scores={}, flagged_phrases=[],
                          rationale="unparseable model output", available=True)


def _http_call(model: str, prompt: str, timeout: float) -> str:
    import httpx
    resp = httpx.post(OLLAMA_URL,
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.0}},
                      timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


class OllamaContentJudge(ContentJudge):
    def __init__(self, model: str = "llama3.2:3b", timeout: float = 30.0,
                 call: Callable[[str], str] | None = None):
        self.model = model
        self.timeout = timeout
        self._call = call or (lambda prompt: _http_call(self.model, prompt, self.timeout))

    def judge(self, question: str, answer: str, history: list[dict], baseline) -> ContentVerdict:
        prompt = build_prompt(question, answer, history)
        try:
            raw = self._call(prompt)
        except Exception as err:  # noqa: BLE001 — any transport failure => honest offline
            return ContentVerdict.offline(f"content layer offline (ollama): {err}")
        return parse_verdict_json(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_ollama_judge.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/content/ollama_judge.py tests/overlay/test_ollama_judge.py
git commit -m "feat(content): OllamaContentJudge adapter — prompt, robust JSON parse, offline fallback

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Pipeline — buffer the timeline + judge_turn

**Files:**
- Modify: `blitz_overlay/pipeline.py`
- Test: `tests/overlay/test_judge_turn.py`

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_judge_turn.py`:

```python
"""session.judge_turn fuses a content verdict with the cue activity of the answer window."""
from blitz_overlay.content.judge import StubContentJudge
from blitz_overlay.pipeline import OverlaySession


def test_judge_turn_returns_fused_result_with_stub():
    sess = OverlaySession(baseline_seconds=0, content_judge=StubContentJudge())
    res = sess.judge_turn(question="Where were you at 9pm?",
                          answer="I was just around somewhere with some people I guess.",
                          t0=0, t1=5000)
    assert res["content_available"] is True
    assert "combined" in res and 0.0 <= res["combined"] <= 1.0
    assert "label" in res
    assert res["content"]["risk"] > 0.3  # vague answer -> elevated content risk


def test_judge_turn_pulls_cue_window(monkeypatch):
    """A synchronous 2-family cue burst inside the window should mark convergence."""
    sess = OverlaySession(baseline_seconds=0, content_judge=StubContentJudge())
    # Inject lit cue activity into the timeline at t≈1000 across 2 families.
    sess.timeline.add(1000, [("visual.gaze_aversion", "visual", 4.0),
                             ("audio.tremor", "audio", 3.0)])
    res = sess.judge_turn(question="Q?",
                          answer="just some stuff, you know, somewhere with people",
                          t0=900, t1=1100)
    assert res["cue_synchronous_families"] >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_judge_turn.py -q`
Expected: FAIL — `OverlaySession.__init__` has no `content_judge` / no `judge_turn`.

- [ ] **Step 3: Implement** — edit `blitz_overlay/pipeline.py`:

(a) Add imports (after the existing `from blitz_overlay.*` imports):

```python
from blitz_overlay.content.fusion import fuse_turn
from blitz_overlay.content.judge import ContentJudge, StubContentJudge
from blitz_overlay.content.timeline import TimelineBuffer
```

(b) Add a `content_judge` parameter to `__init__` (extend the signature) and create the timeline:

```python
    def __init__(self, gate_threshold: float = 0.65, baseline_seconds: int = 90,
                 fps: float = 30.0, log_dir: str | Path = "logs",
                 content_judge: ContentJudge | None = None):
```

and at the end of `__init__`:

```python
        self.timeline = TimelineBuffer()
        self.content_judge: ContentJudge = content_judge or StubContentJudge()
        self._turn_history: list[dict] = []
```

(c) In `process`, in the main (face-present) path, right after `directed_z` is built (the liveness
loop), append to the timeline:

```python
        self.timeline.add(frame.ts, [(cid, self._family_of_cue(cid), z) for cid, z in directed_z.items()])
```

(d) Add the `judge_turn` method (place after `process`):

```python
    def judge_turn(self, question: str, answer: str, t0: int, t1: int) -> dict:
        """Content-first verdict for one Q&A turn, fused with the cue activity in [t0, t1]."""
        verdict = self.content_judge.judge(question, answer, self._turn_history, self.baseline)
        cue_window = self.timeline.window(t0, t1)
        result = fuse_turn(verdict, cue_window)
        result["content"] = verdict.to_dict()
        result["cue_window"] = cue_window
        result["question"] = question
        if verdict.available:
            self._turn_history.append({"question": question, "answer": answer})
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_judge_turn.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite + lint**

Run: `python3 -m pytest -q` then `python3 -m ruff check .`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/pipeline.py tests/overlay/test_judge_turn.py
git commit -m "feat(content): pipeline buffers cue timeline + session.judge_turn (content-first fusion)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Server — WS "turn" message

**Files:**
- Modify: `blitz_overlay/server.py`
- Test: `tests/overlay/test_server.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_server.py` (uses FastAPI's
TestClient websocket, mirroring the existing server test style):

```python
def test_ws_turn_message_returns_turn_result():
    from fastapi.testclient import TestClient
    from blitz_overlay.server import create_app
    from blitz_overlay.config import OverlayConfig

    app = create_app(OverlayConfig(baseline_seconds=0))
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "turn", "question": "Where were you?",
                      "answer": "just somewhere with some people i guess", "t0": 0, "t1": 1000})
        msg = ws.receive_json()
        assert msg["type"] == "turn_result"
        assert msg["content_available"] is True
        assert 0.0 <= msg["combined"] <= 1.0
```

(If `OverlayConfig` can't take `baseline_seconds=` directly, construct it the same way the existing
server tests in this file do — match the established pattern.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_server.py::test_ws_turn_message_returns_turn_result -v`
Expected: FAIL — no `turn_result` (the server treats it as a frame).

- [ ] **Step 3: Implement** — in `blitz_overlay/server.py`, replace the receive loop body:

```python
        try:
            while True:
                raw = await websocket.receive_json()
                if raw.get("type") == "turn":
                    result = await asyncio.to_thread(
                        session.judge_turn,
                        raw.get("question", ""), raw.get("answer", ""),
                        int(raw.get("t0", 0)), int(raw.get("t1", 0)),
                    )
                    result["type"] = "turn_result"
                    await websocket.send_json(result)
                    continue
                consensus = session.process(raw)
                if session.should_emit(consensus.ts):
                    await websocket.send_json(consensus.to_dict())
        except WebSocketDisconnect:
            return
```

and add `import asyncio` at the top of the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/server.py tests/overlay/test_server.py
git commit -m "feat(content): WS 'turn' message routes to the content engine (off-thread, non-blocking)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Browser — Q&A panel + dev scripts (manual-verify)

**Files:**
- Create: `apps/overlay-web/js/qa-panel.js`
- Create: `apps/overlay-web/js/dev-scripts.js`
- Modify: `apps/overlay-web/index.html`, `apps/overlay-web/css/overlay.css`, `apps/overlay-web/js/main.js`

> No JS test runner — verified in-browser (Task 8).

- [ ] **Step 1: Create `apps/overlay-web/js/dev-scripts.js`:**

```javascript
/** Dev/calibration text. HONEST: reading a script is NOT spontaneous lying — these validate the
 *  content analyzer's mechanics (does it score the FALSE one worse), not real deception accuracy. */
export const CALIBRATION_READING =
  "I usually wake up around seven, make coffee, and check the news for a few minutes. " +
  "On weekends I like to walk to the park near my house and read on a bench by the pond.";

export const TRUE_SCRIPT =
  "Last Tuesday at about 8pm I was at Mario's Pizza on 5th Street with my sister Anna. " +
  "We split a margherita, she paid with her blue card, and we left around nine when it started raining.";

export const FALSE_SCRIPT =
  "I was just somewhere around that evening, you know, with some people. " +
  "I think we had some food maybe, I'm not really sure where, and then I guess I went home at some point.";
```

- [ ] **Step 2: Create `apps/overlay-web/js/qa-panel.js`:**

```javascript
/**
 * QaPanel — the content-first Q&A interface. Operator sets a question, captures the answer
 * (live transcript or a dev script), and asks the engine to judge the turn. Renders the
 * content-primary verdict (scores + flagged phrases + content↔cue convergence).
 */
import { CALIBRATION_READING, TRUE_SCRIPT, FALSE_SCRIPT } from "./dev-scripts.js";

export class QaPanel {
  constructor(els, ws, getTranscript, getClock) {
    this.els = els;
    this.ws = ws;                  // WsClient (has .send)
    this.getTranscript = getTranscript;  // () => latest transcript text
    this.getClock = getClock;      // () => performance.now() rounded (same ts as frames)
    this._answerStartTs = null;

    els.start.addEventListener("click", () => this._startAnswer());
    els.judge.addEventListener("click", () => this._judge());
    els.fillTrue.addEventListener("click", () => { els.answer.value = TRUE_SCRIPT; });
    els.fillFalse.addEventListener("click", () => { els.answer.value = FALSE_SCRIPT; });
    els.fillRead.addEventListener("click", () => { els.answer.value = CALIBRATION_READING; });
  }

  _startAnswer() {
    this._answerStartTs = this.getClock();
    this.els.start.textContent = "● answering…";
  }

  _judge() {
    const t0 = this._answerStartTs ?? (this.getClock() - 8000);
    const t1 = this.getClock();
    const answer = this.els.answer.value.trim() || this.getTranscript();
    this.ws.send({ type: "turn", question: this.els.question.value, answer, t0, t1 });
    this.els.verdict.textContent = "judging…";
    this.els.start.textContent = "start answer";
    this._answerStartTs = null;
  }

  /** Called when a turn_result arrives over the WS. */
  showResult(r) {
    const s = r.content && r.content.scores ? r.content.scores : {};
    const flagged = (r.content && r.content.flagged_phrases || [])
      .map((f) => `“${f.text}” — ${f.reason}`).join("; ");
    const pct = Math.round((r.combined || 0) * 100);
    this.els.verdict.innerHTML =
      `<b>${r.label}</b> · ${pct}%` +
      (r.content_available
        ? `<br><span class="qa-scores">consistency ${fmt(s.consistency)} · richness ${fmt(s.richness_rm)} · ` +
          `verifiable ${fmt(s.verifiability)} · relevant ${fmt(s.relevance)}</span>` +
          (flagged ? `<br><span class="qa-flagged">${flagged}</span>` : "")
        : `<br><span class="qa-flagged">content layer offline — install/run Ollama for content analysis</span>`);
  }
}

function fmt(v) { return v == null ? "—" : Math.round(v * 100) + "%"; }
```

- [ ] **Step 3: Add DOM** — in `apps/overlay-web/index.html`, inside `<section class="panel">` (after
the `verifier` block, before `</section>`):

```html
      <div class="qa">
        <div class="qa-title">Content engine · Q&amp;A</div>
        <input id="qa-question" class="qa-question" placeholder="Question…"
               value="Where were you last Tuesday evening?" />
        <textarea id="qa-answer" class="qa-answer" rows="2"
                  placeholder="Answer (live transcript fills if blank)…"></textarea>
        <div class="qa-buttons">
          <button id="qa-start" type="button">start answer</button>
          <button id="qa-judge" type="button">judge</button>
          <button id="qa-fill-read" type="button">read</button>
          <button id="qa-fill-true" type="button">TRUE</button>
          <button id="qa-fill-false" type="button">FALSE</button>
        </div>
        <div id="qa-verdict" class="qa-verdict">—</div>
      </div>
```

- [ ] **Step 4: Add styles** — append to `apps/overlay-web/css/overlay.css`:

```css
/* ─── Content engine Q&A panel ─── */
.qa { padding: 10px 14px 14px; border-top: 1px solid #1f2733; }
.qa-title { font: 10px/1.4 monospace; letter-spacing: .1em; text-transform: uppercase; color: #7d8da3; margin-bottom: 6px; }
.qa-question, .qa-answer { width: 100%; background: #0b0f14; color: #e6edf3; border: 1px solid #2a3645; border-radius: 6px; padding: 6px 8px; font: 12px/1.4 monospace; margin-bottom: 6px; }
.qa-buttons { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.qa-buttons button { background: #11161d; color: #9fb0c3; border: 1px solid #2a3645; border-radius: 6px; padding: 4px 10px; cursor: pointer; font: 11px monospace; }
.qa-verdict { font: 12px/1.5 monospace; color: #e6edf3; min-height: 16px; }
.qa-scores { color: #9fb0c3; font-size: 11px; }
.qa-flagged { color: #ff9f43; font-size: 11px; }
```

- [ ] **Step 5: Wire in `apps/overlay-web/js/main.js`:**

(a) Import (after the other imports):

```javascript
import { QaPanel } from "./qa-panel.js";
```

(b) After the WS is constructed, create the panel and route `turn_result` messages. Replace the
existing `new WsClient(...)` consensus handler so it also forwards turn results:

```javascript
const ws = new WsClient(wsUrl, (c) => {
  if (c.type === "turn_result") { qaPanel.showResult(c); return; }
  renderer.setConsensus(c);
  enneagram.setConsensus(c);
  cueVerifier.setConsensus(c);
  cueTimeline.push(c);
  calibration.setConsensus(c);
  bellPlayer.handle(c.bell);
  trustEl.textContent =
    `trust: ${Math.round(bellPlayer.trust() * 100)}% · bells/min: ${bellPlayer.bellCount()}`;
},
  (s) => { if (s === "engine-offline") panel.message.textContent = "Engine offline — reconnecting…"; });

const qaPanel = new QaPanel({
  question: document.getElementById("qa-question"),
  answer: document.getElementById("qa-answer"),
  start: document.getElementById("qa-start"),
  judge: document.getElementById("qa-judge"),
  fillRead: document.getElementById("qa-fill-read"),
  fillTrue: document.getElementById("qa-fill-true"),
  fillFalse: document.getElementById("qa-fill-false"),
  verdict: document.getElementById("qa-verdict"),
}, ws, () => (transcriber.available ? transcriber.latest().text : ""), () => Math.round(performance.now()));
```

(Note: `qaPanel` is referenced inside the WS callback but assigned just after — that's fine because
the callback only fires on later messages, after assignment. If your linter complains, declare
`let qaPanel;` before the `new WsClient` line and assign without `const`.)

- [ ] **Step 6: Commit**

```bash
git add apps/overlay-web/js/qa-panel.js apps/overlay-web/js/dev-scripts.js apps/overlay-web/index.html apps/overlay-web/css/overlay.css apps/overlay-web/js/main.js
git commit -m "feat(content): browser Q&A panel + True/False/read dev scripts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Manual verification (user-confirmed)

**Files:** none.

- [ ] **Step 1:** Launch `BLITZ_OVERLAY_BASELINE_SECONDS=20 python3 -m blitz_overlay`; open Chrome.
- [ ] **Step 2 (stub path, works without Ollama):** In the Q&A panel, click **TRUE** then **judge** → note the combined %. Click **FALSE** then **judge** → the FALSE script should score **higher risk** (vaguer/less verifiable). This proves the content→fusion path end to end.
- [ ] **Step 3 (live Ollama, if installed):** confirm `ollama` is running (`ollama list` shows `llama3.2:3b`). Restart the engine with the Ollama judge wired (a follow-up flips the default judge from stub→Ollama once you confirm the model is pulled). Judge a real spoken answer; verify a content verdict + flagged phrases render, and "content layer offline" appears if you stop Ollama.
- [ ] **Step 4: Report to user.** In-browser confirmation gate before merge.

> NOTE for the wiring of the live judge: the server currently defaults `OverlaySession` to the
> `StubContentJudge`. A one-line follow-up (after the user confirms Ollama is pulled) passes
> `content_judge=OllamaContentJudge()` in `server.py`'s session construction — left as a guarded
> switch so the test suite and the no-Ollama path keep using the deterministic stub.
```

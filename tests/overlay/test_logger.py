import json
from pathlib import Path

from blitz_overlay.logger import PredictionLogger
from blitz_overlay.schemas import SCHEMA_VERSION, ActiveCue, Consensus, FamilyVote


def _consensus():
    return Consensus(
        schema_version=SCHEMA_VERSION, ts=1234, status="WATCH", risk=0.5, flag=False,
        n_agree=1, n_required=2,
        families=[FamilyVote("visual", True, True, False, 0.3)],
        active_cues=[ActiveCue("visual.gaze_aversion", "eyes", 2.5, 0.8)],
    )


def test_logger_writes_jsonl_line(tmp_path):
    log = PredictionLogger(session_id="sess1", log_dir=tmp_path)
    log.log(_consensus(), baseline_mode="rolling")
    files = list(Path(tmp_path).glob("*.jsonl"))
    assert len(files) == 1
    line = json.loads(files[0].read_text().strip())
    assert line["status"] == "WATCH"
    assert line["posterior"] == 0.5
    assert line["baseline_mode"] == "rolling"
    assert line["weight_set_version"]
    assert line["schema_version"] == SCHEMA_VERSION


def test_logger_records_cue_contributions_not_raw_biometric(tmp_path):
    log = PredictionLogger(session_id="sess2", log_dir=tmp_path)
    log.log(_consensus(), baseline_mode="rolling")
    line = json.loads(next(Path(tmp_path).glob("*.jsonl")).read_text().strip())
    assert "active_cues" in line
    assert line["active_cues"][0]["cue_id"] == "visual.gaze_aversion"
    raw = next(Path(tmp_path).glob("*.jsonl")).read_text()
    assert "blendshapes" not in raw and "landmarks" not in raw and "rgb" not in raw


def test_logger_appends_multiple_lines(tmp_path):
    log = PredictionLogger(session_id="sess3", log_dir=tmp_path)
    log.log(_consensus(), baseline_mode="rolling")
    log.log(_consensus(), baseline_mode="rolling")
    lines = next(Path(tmp_path).glob("*.jsonl")).read_text().strip().splitlines()
    assert len(lines) == 2

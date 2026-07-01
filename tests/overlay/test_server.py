from fastapi.testclient import TestClient

from blitz_overlay.config import OverlayConfig
from blitz_overlay.server import create_app


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("BLITZ_OVERLAY_PORT", "9001")
    monkeypatch.setenv("BLITZ_OVERLAY_GATE", "0.7")
    monkeypatch.setenv("BLITZ_OVERLAY_BASELINE_SECONDS", "45")
    cfg = OverlayConfig.from_env()
    assert cfg.port == 9001
    assert cfg.gate == 0.7
    assert cfg.baseline_seconds == 45


def test_index_served(tmp_path):
    app = create_app(OverlayConfig(log_dir=str(tmp_path)))
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Live Consensus Overlay" in resp.text


def test_ws_returns_consensus_for_a_feature_frame(tmp_path):
    app = create_app(OverlayConfig(baseline_seconds=0, log_dir=str(tmp_path)))
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"ts": 0, "face_present": True, "confidence": 0.9,
                      "blendshapes": {"eyeBlinkLeft": 0.05},
                      "geometry": {"gaze_x": 0.02, "gaze_y": 0.0, "jaw_width_ratio": 0.8}})
        msg = ws.receive_json()
        assert msg["schema_version"]
        assert msg["status"] in ("CALIBRATING", "CLEAR", "WATCH", "FLAG")
        assert "families" in msg


def test_ws_turn_message_returns_turn_result():
    app = create_app(OverlayConfig(baseline_seconds=0))
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "turn", "question": "Where were you?",
                      "answer": "just somewhere with some people i guess", "t0": 0, "t1": 1000})
        msg = ws.receive_json()
        assert msg["type"] == "turn_result"
        assert msg["content_available"] is True
        assert 0.0 <= msg["combined"] <= 1.0

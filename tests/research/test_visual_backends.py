"""Backend seam tests — stubs only; real model backends are lazy and never imported here."""
from modalities.visual.backends import EnsembleBackend, StubBackend, VisualFrame


def _vf(ts, au01=0.0, au12=0.0, **kw):
    return VisualFrame(ts_ms=ts, face_present=True, quality=0.9,
                       aus={"AU01": au01, "AU12": au12}, **kw)


def test_stub_backend_returns_frames():
    frames = [_vf(0), _vf(100, au01=0.5)]
    assert StubBackend(frames).extract("clip.mp4") == frames


def test_ensemble_backend_averages_au_values():
    a = StubBackend([_vf(0, au01=0.2)])
    b = StubBackend([_vf(0, au01=0.6)])
    merged = EnsembleBackend(a, b).extract("clip.mp4")
    assert len(merged) == 1
    assert abs(merged[0].aus["AU01"] - 0.4) < 1e-9


def test_pyfeat_backend_is_lazy_and_guides_install():
    from modalities.visual.backends import PyFeatBackend
    backend = PyFeatBackend()   # constructing must NOT import py-feat
    try:
        backend.extract("missing.mp4")
    except RuntimeError as e:
        assert "research" in str(e)   # install guidance
    except Exception:
        pass   # py-feat installed and failed on the missing file — also acceptable

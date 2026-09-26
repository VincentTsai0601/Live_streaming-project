from types import SimpleNamespace
import pytest
import streamer_recommender as engine


@pytest.fixture
def anchor(recommender):
    return recommender.retrieve("歌唱", required_tags=["互動熱絡", "歌唱"])[0]


def test_no_key_falls_back(anchor):
    assert engine.generate_explanation("歌唱", anchor) == engine.fallback_explanation(
        anchor
    )
    assert "歌唱" in engine.fallback_explanation(anchor)


def test_missing_sdk_falls_back(monkeypatch, anchor):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(engine, "genai", None)
    assert engine.generate_explanation("歌唱", anchor) == engine.fallback_explanation(
        anchor
    )


@pytest.mark.parametrize(
    "response", [" 說明 ", "", None, RuntimeError("offline failure")]
)
def test_mocked_generation(monkeypatch, anchor, response):
    def generate_content(**kwargs):
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(
        engine,
        "genai",
        SimpleNamespace(
            Client=lambda **kw: SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content)
            )
        ),
    )
    expected = "說明" if response == " 說明 " else engine.fallback_explanation(anchor)
    assert engine.generate_explanation("歌唱", anchor) == expected


def test_current_model_selection_uses_environment(monkeypatch, anchor):
    """Characterization only: ignored argument remains an open defect."""
    requests = []

    def generate_content(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(text="說明")

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setenv("GEMINI_MODEL", "environment-model")
    monkeypatch.setattr(
        engine,
        "genai",
        SimpleNamespace(
            Client=lambda **kw: SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content)
            )
        ),
    )
    engine.generate_explanation("歌唱", anchor, gemini_model="selected-model")
    assert requests[0]["model"] == "environment-model"
    assert anchor["explanation_document"] in requests[0]["contents"]

import numpy as np
import pandas as pd
import pytest
import streamer_recommender as engine


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, {}),
        (np.nan, {}),
        ("broken", {}),
        ("[1]", {}),
        ('{"a": 2}', {"a": "2"}),
        ({1: 2}, {"1": "2"}),
    ],
)
def test_parse_reasons(value, expected):
    assert engine.parse_reasons(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, set()),
        (np.nan, set()),
        ("", set()),
        (
            " 歌唱、舞蹈,歌唱；聊天/音樂，互動; ",
            {"歌唱", "舞蹈", "聊天", "音樂", "互動"},
        ),
    ],
)
def test_split_tags(value, expected):
    assert engine.split_tags(value) == expected


def test_preferences(recommender):
    query = engine.normalize_query(" 女生唱歌好聽 ")
    assert query.startswith("女生唱歌好聽")
    assert recommender._extract_preferences(query)["talents"] == {"歌唱"}


@pytest.mark.parametrize(
    "query,expected",
    [("女主播", "女"), ("男生", "男"), ("男生女生", None), ("音樂", None)],
)
def test_gender(query, expected):
    assert engine.HybridStreamerRecommender._extract_gender(query) == expected


@pytest.mark.parametrize("defect", ["missing_column", "duplicate_id"])
def test_validation(csv_path, tmp_path, defect):
    data = pd.read_csv(csv_path)
    if defect == "missing_column":
        data = data.drop(columns="talents")
    else:
        data.loc[1, "pfid"] = data.loc[0, "pfid"]
    data.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        engine.HybridStreamerRecommender(csv_path, cache_dir=tmp_path)


def test_documents_and_evidence(recommender):
    row = recommender.df.iloc[0].copy()
    assert "性別：女" in engine.build_retrieval_document(row)
    assert "才藝：歌唱證據" in engine.build_retrieval_document(row)
    row["self_description"] = "字" * 250
    assert engine.build_explanation_document(row).endswith("字" * 220)
    assert not engine.build_explanation_document(row).endswith("字" * 221)
    reasons = {str(i): f"tag{i}" for i in range(6)}
    evidence = recommender._matching_evidence(reasons, [f"tag{i}" for i in range(6)])
    assert len(evidence) == 4
    assert all(item["evidence"] == reasons[item["source"]] for item in evidence)

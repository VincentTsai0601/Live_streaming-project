import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from google import genai
except ImportError:
    genai = None


DATA_PATH = Path(os.getenv("DATA_PATH", "anchors_100.csv"))
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "shibing624/text2vec-base-chinese"
)
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))

TAG_COLUMNS = {
    "personality": "性格",
    "appearance": "外型",
    "talents": "才藝",
    "featured_topics": "主題",
    "live_streaming_style": "風格",
}

# 將常見的自然語言說法對應到資料集中的標籤。
SYNONYMS = {
    "女生": "女",
    "女性": "女",
    "男生": "男",
    "男性": "男",
    "唱歌": "歌唱",
    "唱歌好聽": "歌唱",
    "愛聊天": "聊天互動",
    "很會聊天": "聊天互動",
    "陪聊": "情感陪聊",
    "跳舞": "舞蹈",
    "打遊戲": "互動遊戲",
    "玩遊戲": "互動遊戲",
    "吃東西": "吃播",
    "搞笑": "幽默搞笑型",
    "有趣": "幽默搞笑型",
    "活潑": "熱情活力型",
    "熱情": "熱情活力型",
    "療癒": "溫柔療癒型",
    "放鬆": "溫柔療癒",
    "陪伴": "日常陪伴感",
    "很會互動": "互動熱絡",
    "互動多": "互動熱絡",
}


def parse_reasons(value: Any) -> dict[str, str]:
    """安全地將 reasons 欄位解析成字典。"""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return {}
    try:
        parsed = json.loads(str(value))
        return (
            {str(k): str(v) for k, v in parsed.items()}
            if isinstance(parsed, dict)
            else {}
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def reasons_to_text(reasons: dict[str, str]) -> str:
    return " ".join(f"{key}：{value}" for key, value in reasons.items())


def split_tags(value: Any) -> set[str]:
    """正規化不同標點與順序的多標籤欄位。"""
    if value is None or pd.isna(value):
        return set()
    return {
        tag.strip()
        for tag in re.split(r"[、,，;/；]+", str(value))
        if tag.strip()
    }


def normalize_query(query: str) -> str:
    normalized = query.strip()
    for source in sorted(SYNONYMS, key=len, reverse=True):
        if source in normalized:
            normalized += f" {SYNONYMS[source]}"
    return normalized


def build_retrieval_document(row: pd.Series) -> str:
    """建立供語意檢索使用的精簡、欄位化文本。"""
    return " ".join(
        [
            f"性別：{row.get('gender', '')}",
            f"性格：{row.get('personality', '')}",
            f"外型：{row.get('appearance', '')}",
            f"才藝：{row.get('talents', '')}",
            f"直播主題：{row.get('featured_topics', '')}",
            f"直播風格：{row.get('live_streaming_style', '')}",
            f"整體氛圍：{row.get('overall_vibe', '')}",
            f"判斷依據：{reasons_to_text(row['reasons_dict'])}",
        ]
    )


def build_explanation_document(row: pd.Series) -> str:
    return (
        f"{row['retrieval_document']} "
        f"主播自我介紹：{str(row.get('self_description', ''))[:220]}"
    )


class HybridStreamerRecommender:
    """結構化標籤比對 + 中文 bi-encoder 語意檢索。"""

    def __init__(
        self,
        csv_path: str | Path = DATA_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str | Path = CACHE_DIR,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.df = pd.read_csv(self.csv_path)
        self._validate_data()

        self.df["reasons_dict"] = self.df["reasons"].map(parse_reasons)
        for column in TAG_COLUMNS:
            self.df[f"{column}_tags"] = self.df[column].map(split_tags)
        self.df["retrieval_document"] = self.df.apply(
            build_retrieval_document, axis=1
        )
        self.df["explanation_document"] = self.df.apply(
            build_explanation_document, axis=1
        )

        self.vectorizer: TfidfVectorizer | None = None
        if SentenceTransformer is not None:
            print(f"正在載入 Embedding 模型：{self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            self.item_embeddings = self._load_or_create_embeddings()
            self.retrieval_backend = "sentence-transformer"
        else:
            print("未安裝 sentence-transformers，改用本地 TF-IDF baseline。")
            self.encoder = None
            self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
            self.item_embeddings = self.vectorizer.fit_transform(
                self.df["retrieval_document"].tolist()
            )
            self.retrieval_backend = "tf-idf"

    def _validate_data(self) -> None:
        required = {
            "pfid",
            "gender",
            "personality",
            "appearance",
            "talents",
            "featured_topics",
            "live_streaming_style",
            "overall_vibe",
            "reasons",
            "self_description",
        }
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要欄位：{sorted(missing)}")
        if self.df["pfid"].duplicated().any():
            raise ValueError("pfid 必須是唯一值。")

    def _cache_path(self) -> Path:
        digest = hashlib.sha256()
        digest.update(self.csv_path.read_bytes())
        digest.update(self.model_name.encode("utf-8"))
        return self.cache_dir / f"item_embeddings_{digest.hexdigest()[:16]}.npy"

    def _load_or_create_embeddings(self) -> np.ndarray:
        cache_path = self._cache_path()
        if cache_path.exists():
            embeddings = np.load(cache_path)
            if len(embeddings) == len(self.df):
                print(f"已載入 Embedding 快取：{cache_path}")
                return embeddings

        print("正在離線計算主播 Embeddings...")
        embeddings = self.encoder.encode(
            self.df["retrieval_document"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        return embeddings

    @staticmethod
    def _extract_gender(query: str) -> str | None:
        female = bool(re.search(r"女主播|女生|女性", query))
        male = bool(re.search(r"男主播|男生|男性", query))
        if female and not male:
            return "女"
        if male and not female:
            return "男"
        return None

    def _known_tags(self) -> dict[str, set[str]]:
        return {
            column: set().union(*self.df[f"{column}_tags"].tolist())
            for column in TAG_COLUMNS
        }

    def _extract_preferences(self, normalized_query: str) -> dict[str, set[str]]:
        return {
            column: {tag for tag in tags if tag in normalized_query}
            for column, tags in self._known_tags().items()
        }

    def _structured_scores(
        self, preferences: dict[str, set[str]]
    ) -> tuple[np.ndarray, list[list[str]]]:
        weights = {
            "talents": 1.30,
            "live_streaming_style": 1.20,
            "featured_topics": 1.00,
            "personality": 1.00,
            "appearance": 0.70,
        }
        scores = np.zeros(len(self.df), dtype=float)
        matched_tags: list[list[str]] = [[] for _ in range(len(self.df))]
        total_possible = sum(
            weights[column] * len(tags)
            for column, tags in preferences.items()
        )
        if total_possible == 0:
            return scores, matched_tags

        for index, row in self.df.iterrows():
            for column, requested_tags in preferences.items():
                matches = requested_tags.intersection(row[f"{column}_tags"])
                scores[index] += weights[column] * len(matches)
                matched_tags[index].extend(sorted(matches))
        return scores / total_possible, matched_tags

    def _required_tag_mask(self, required_tags: set[str]) -> np.ndarray:
        """必要標籤採 AND 邏輯：每位候選人必須包含所有指定標籤。"""
        if not required_tags:
            return np.ones(len(self.df), dtype=bool)
        mask = np.ones(len(self.df), dtype=bool)
        for index, row in self.df.iterrows():
            all_tags = set().union(
                *(row[f"{column}_tags"] for column in TAG_COLUMNS)
            )
            mask[index] = required_tags.issubset(all_tags)
        return mask

    @staticmethod
    def _matching_evidence(
        reasons: dict[str, str], matched_tags: list[str]
    ) -> list[dict[str, str]]:
        evidence = []
        for tag in matched_tags:
            for key, value in reasons.items():
                if tag in key or tag in value:
                    evidence.append({"tag": tag, "source": key, "evidence": value})
                    break
        return evidence[:4]

    @staticmethod
    def _rerank_for_diversity(
        candidate_indices: np.ndarray,
        final_scores: np.ndarray,
        embeddings: Any,
        top_k: int,
        diversity: float,
    ) -> list[int]:
        """以簡化 MMR 避免 Top-N 全是幾乎相同的主播。"""
        if diversity <= 0 or len(candidate_indices) <= 1:
            return candidate_indices[np.argsort(final_scores[candidate_indices])[::-1]][
                :top_k
            ].tolist()

        remaining = candidate_indices.tolist()
        selected: list[int] = []
        relevance_weight = 1.0 - diversity
        while remaining and len(selected) < top_k:
            best_idx = None
            best_value = -float("inf")
            for idx in remaining:
                if not selected:
                    redundancy = 0.0
                elif hasattr(embeddings, "toarray"):
                    vector = embeddings[idx]
                    redundancy = max(
                        float(vector.multiply(embeddings[j]).sum()) for j in selected
                    )
                else:
                    redundancy = max(
                        float(embeddings[idx] @ embeddings[j]) for j in selected
                    )
                value = relevance_weight * final_scores[idx] - diversity * redundancy
                # pfid 作為穩定的最後 tie-break，確保每次執行結果一致。
                value += 1e-12 * (1.0 / (idx + 1))
                if value > best_value:
                    best_value, best_idx = value, idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    def retrieve(
        self,
        user_query: str,
        top_k: int = 3,
        required_gender: str | None = None,
        required_tags: list[str] | set[str] | None = None,
        diversity: float = 0.12,
    ) -> list[dict[str, Any]]:
        if not user_query.strip():
            raise ValueError("搜尋條件不能是空白。")
        top_k = max(1, min(int(top_k), len(self.df)))
        normalized_query = normalize_query(user_query)
        gender_filter = required_gender or self._extract_gender(normalized_query)
        preferences = self._extract_preferences(normalized_query)
        required_tags_set = set(required_tags or [])

        if self.retrieval_backend == "sentence-transformer":
            query_embedding = self.encoder.encode(
                [normalized_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            semantic_scores = (query_embedding @ self.item_embeddings.T)[0]
        else:
            query_embedding = self.vectorizer.transform([normalized_query])
            # TfidfVectorizer 預設 L2 正規化，內積即 cosine similarity。
            semantic_scores = (query_embedding @ self.item_embeddings.T).toarray()[0]
        structured_scores, matched_tags = self._structured_scores(preferences)

        # 若沒有抽取到精確標籤，完全依賴語意分數；否則使用混合評分。
        has_structured_preferences = any(preferences.values())
        if has_structured_preferences:
            final_scores = 0.65 * semantic_scores + 0.35 * structured_scores
        else:
            final_scores = semantic_scores.copy()

        candidate_mask = np.ones(len(self.df), dtype=bool)
        if gender_filter:
            candidate_mask = self.df["gender"].to_numpy() == gender_filter
        candidate_mask &= self._required_tag_mask(required_tags_set)
        candidate_indices = np.flatnonzero(candidate_mask)
        if len(candidate_indices) == 0:
            raise ValueError("找不到同時滿足所有必要條件的主播，請減少必要條件。")
        ranked = self._rerank_for_diversity(
            candidate_indices,
            final_scores,
            self.item_embeddings,
            top_k,
            min(max(float(diversity), 0.0), 0.45),
        )

        results = []
        for rank, idx in enumerate(ranked, start=1):
            row = self.df.iloc[idx]
            evidence = self._matching_evidence(
                row["reasons_dict"], matched_tags[idx]
            )
            results.append(
                {
                    "rank": rank,
                    "pfid": int(row["pfid"]),
                    "score": round(float(final_scores[idx]), 4),
                    "score_breakdown": {
                        "semantic": round(float(semantic_scores[idx]), 4),
                        "structured": round(float(structured_scores[idx]), 4),
                    },
                    "matched_tags": matched_tags[idx],
                    "required_tags": sorted(required_tags_set),
                    "evidence": evidence,
                    "metadata": {
                        "性別": row["gender"],
                        "性格": row["personality"],
                        "外型": row["appearance"],
                        "才藝": row["talents"],
                        "主題": row["featured_topics"],
                        "風格": row["live_streaming_style"],
                        "氛圍": row["overall_vibe"],
                    },
                    "explanation_document": row["explanation_document"],
                }
            )
        return results


def fallback_explanation(anchor: dict[str, Any]) -> str:
    metadata = anchor["metadata"]
    matches = "、".join(anchor["matched_tags"][:3])
    if matches:
        return (
            f"這位主播與你的偏好相符之處包括「{matches}」。"
            f"其才藝為「{metadata['才藝']}」，直播風格是「{metadata['風格']}」，"
            f"整體氛圍為「{metadata['氛圍']}」。"
        )
    return (
        f"這位主播的才藝為「{metadata['才藝']}」，直播風格是「{metadata['風格']}」。"
        f"系統根據完整主播資料判斷其語意與你的需求較為接近。"
    )


def generate_explanation(
    user_query: str,
    anchor: dict[str, Any],
    gemini_model: str | None = None,
) -> str:
    """Use Gemini when available; otherwise use local fallback."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or genai is None:
        return fallback_explanation(anchor)

    evidence_text = json.dumps(
        anchor["evidence"],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
你是直播平台的推薦說明助手。

請根據使用者需求、主播資料與證據，產生 60 至 80 字的繁體中文推薦理由。

規則：
1. 只能使用提供的主播資料與證據。
2. 不得添加未提供的能力、經歷或特徵。
3. 優先說明吻合的才藝、主題、性格與直播風格。
4. 沒有資料支持的需求，不得宣稱符合。
5. 不使用「完美」「最佳」等絕對詞彙。
6. 不提及演算法、向量或相似度。
7. 不使用條列式。

使用者需求：
{user_query}

主播資料：
{anchor["explanation_document"]}

符合需求的資料證據：
{evidence_text}
"""

    try:
        #print("正在使用 Gemini 產生推薦理由")
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.1-flash-lite"
            ),
            contents=prompt,
        )

        if response.text:
            return response.text.strip()

    except Exception as error:
        print(f"Gemini generation failed; using fallback: {error}")

    return fallback_explanation(anchor)


def main() -> None:
    print(f"正在載入資料：{DATA_PATH}")
    recommender = HybridStreamerRecommender(DATA_PATH, EMBEDDING_MODEL_NAME)
    query = input("請描述想看的主播類型：").strip()
    top_k_text = input("希望推薦幾位主播？[預設 3]：").strip()
    top_k = int(top_k_text) if top_k_text else 3

    results = recommender.retrieve(query, top_k=top_k)
    print("\n" + "=" * 60)
    print(f"搜尋條件：{query}")
    print("=" * 60)
    for anchor in results:
        print(
            f"\nTop {anchor['rank']} | 主播 ID：{anchor['pfid']} | "
            f"綜合分數：{anchor['score']:.4f}"
        )
        print(
            "分數拆解："
            f"語意={anchor['score_breakdown']['semantic']:.4f}, "
            f"結構化={anchor['score_breakdown']['structured']:.4f}"
        )
        print(
            f"核心標籤：{anchor['metadata']['才藝']} | "
            f"{anchor['metadata']['風格']}"
        )
        if anchor["evidence"]:
            print("資料證據：")
            for item in anchor["evidence"]:
                print(f"- {item['source']}：{item['evidence']}")
        print(f"推薦理由：{generate_explanation(query, anchor)}")
        print("-" * 60)


if __name__ == "__main__":
    main()


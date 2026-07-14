import os
from pathlib import Path

from dotenv import load_dotenv

# 強制讀取與 app.py 放在同一個資料夾的 .env
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

dotenv_loaded = load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
)

import streamlit as st

from streamer_recommender import (
    DATA_PATH,
    EMBEDDING_MODEL_NAME,
    HybridStreamerRecommender,
    generate_explanation,
)


st.set_page_config(page_title="主播推薦系統", page_icon="🎙️", layout="wide")
st.title("🎙️主播推薦系統🎙️")
st.caption("結合結構化條件、中文語意檢索與 metadata 證據的內容推薦")


@st.cache_resource(show_spinner="正在載入推薦模型與主播資料…")
def load_recommender():
    return HybridStreamerRecommender(DATA_PATH, EMBEDDING_MODEL_NAME)


try:
    recommender = load_recommender()
except Exception as error:
    st.error(f"系統初始化失敗：{error}")
    st.info("請確認 DATA_PATH 指向 anchors_100.csv，並已安裝 requirements.txt。")
    st.stop()

with st.sidebar:
    st.header("必要條件")
    gender_option = st.selectbox("性別", ["不限", "女", "男"])

    all_tags = sorted(
        set().union(
            *(
                set().union(*recommender.df[f"{column}_tags"].tolist())
                for column in [
                    "talents",
                    "featured_topics",
                    "live_streaming_style",
                    "personality",
                ]
            )
        )
    )
    required_tags = st.multiselect(
        "必須包含的標籤",
        all_tags,
        help="這裡採 AND 邏輯；選擇的每個標籤都必須符合。",
    )

    st.header("推薦設定")
    top_k = st.slider("推薦人數", 1, 20, 3)
    diversity = st.slider(
        "結果多樣性",
        min_value=0.0,
        max_value=0.35,
        value=0.12,
        step=0.01,
        help="提高後可減少結果過度相似，但可能稍微降低相關性。",
    )
    st.caption(f"檢索模式：{recommender.retrieval_backend}")
    st.header("Gemini 設定")

    gemini_models = {
        "Gemini 3.1 Flash-Lite（快速／節省額度）":
            "gemini-3.1-flash-lite",
        "Gemini 3.5 Flash（較高文字品質）":
            "gemini-3.5-flash",
        "gemini-2.5-flash（Base版，僅作為備援）":
            "gemini-2.5-flash",
    }

    default_model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.1-flash-lite",
    )

    model_ids = list(gemini_models.values())

    default_index = (
        model_ids.index(default_model)
        if default_model in model_ids
        else 0
    )

    selected_model_label = st.selectbox(
        "推薦理由模型",
        list(gemini_models.keys()),
        index=default_index,
    )

    selected_gemini_model = gemini_models[
        selected_model_label
    ]

    if os.getenv("GEMINI_API_KEY"):
        st.success("Gemini API 已啟用")
    else:
        st.info("未設定 Gemini API Key，將使用本地推薦理由。")
    query = st.text_area(
    "描述你想看的主播",
    value="想看會唱歌、互動熱絡，能讓人感到放鬆的主播",
    height=100,
)

if st.button("開始推薦", type="primary", use_container_width=True):
    try:
        results = recommender.retrieve(
            query,
            top_k=top_k,
            required_gender=None if gender_option == "不限" else gender_option,
            required_tags=required_tags,
            diversity=diversity,
        )
    except ValueError as error:
        st.warning(str(error))
    else:
        for anchor in results:
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.subheader(f"Top {anchor['rank']} · 主播 {anchor['pfid']}")
                    explanation = generate_explanation(
                        query,
                        anchor,
                        gemini_model=selected_gemini_model,
                    )                   
                    st.write(explanation)
                    # st.write(generate_explanation(query, anchor))
                with right:
                    st.metric("綜合分數", f"{anchor['score']:.3f}")
                    st.caption(
                        f"語意 {anchor['score_breakdown']['semantic']:.3f} · "
                        f"標籤 {anchor['score_breakdown']['structured']:.3f}"
                    )

                metadata = anchor["metadata"]
                st.markdown(
                    f"**才藝：** {metadata['才藝']}　　"
                    f"**主題：** {metadata['主題']}　　"
                    f"**風格：** {metadata['風格']}"
                )
                if anchor["matched_tags"]:
                    st.write("匹配標籤：", "、".join(anchor["matched_tags"]))
                with st.expander("查看推薦證據與完整資料"):
                    if anchor["evidence"]:
                        for item in anchor["evidence"]:
                            st.markdown(f"- **{item['source']}**：{item['evidence']}")
                    else:
                        st.caption("本結果主要來自整體語意匹配，沒有直接標籤證據。")
                    st.json(metadata, expanded=False)
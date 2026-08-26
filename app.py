import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="ShopSense AI", page_icon="🛒")

st.title("🛒 ShopSense AI")
st.caption("At-risk review detector — powered by a pretrained multilingual transformer")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        device=-1
    )

model = load_model()

review = st.text_area(
    "Paste a customer review (Portuguese or English):",
    placeholder="Produto chegou quebrado e a loja não respondeu",
    height=120
)

if st.button("Analyze", type="primary") and review.strip():
    with st.spinner("Analyzing..."):
        result = model(review, truncation=True, max_length=128)[0]
        label = result["label"]
        score = result["score"]
        is_at_risk = label == "negative"

    st.divider()

    if is_at_risk:
        st.error(f"⚠️ AT-RISK REVIEW  —  {score:.1%} confidence")
        st.write("This review suggests the customer had a negative experience. Consider flagging for follow-up.")
    else:
        st.success(f"✅ Not flagged  —  sentiment: **{label}** ({score:.1%} confidence)")

    with st.expander("Raw model output"):
        st.json(result)

st.divider()
st.caption(
    "Model comparison: Random Forest (structured data) reached 91% accuracy / 85% precision on Bad reviews. "
    "This transformer trades some precision (47%) for better recall (59%) — it catches more at-risk reviews, "
    "at the cost of more false alarms."
)

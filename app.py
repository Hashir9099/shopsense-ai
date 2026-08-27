import streamlit as st
import pandas as pd
import sqlite3
import os
import io
from datetime import datetime
from transformers import pipeline
 
# Optional: joblib-loaded Random Forest ensemble (only used if the file exists)
RF_MODEL_PATH = "rf_model.pkl"
RF_FEATURE_COLUMNS_PATH = "rf_feature_columns.pkl"
 
st.set_page_config(page_title="ShopSense AI", page_icon="🛒", layout="wide")
 
# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_transformer():
    return pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        device=-1
    )
 
@st.cache_resource
def load_rf_model():
    if os.path.exists(RF_MODEL_PATH):
        import joblib
        rf = joblib.load(RF_MODEL_PATH)
        cols = joblib.load(RF_FEATURE_COLUMNS_PATH) if os.path.exists(RF_FEATURE_COLUMNS_PATH) else None
        return rf, cols
    return None, None
 
model = load_transformer()
rf_model, rf_columns = load_rf_model()
 
# ---------------------------------------------------------------------------
# Persistent storage (SQLite)
# ---------------------------------------------------------------------------
DB_PATH = "shopsense_history.db"
 
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            review_text TEXT,
            label TEXT,
            score REAL,
            is_at_risk INTEGER,
            urgency TEXT,
            resolved INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
 
def save_review(review_text, label, score, is_at_risk, urgency):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO reviews (timestamp, review_text, label, score, is_at_risk, urgency) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), review_text, label, score, int(is_at_risk), urgency)
    )
    conn.commit()
    conn.close()
 
def get_all_reviews():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM reviews ORDER BY id DESC", conn)
    conn.close()
    return df
 
def mark_resolved(review_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE reviews SET resolved = 1 WHERE id = ?", (review_id,))
    conn.commit()
    conn.close()
 
init_db()
 
# ---------------------------------------------------------------------------
# Explainability: keyword-based highlighting
# ---------------------------------------------------------------------------
RISK_KEYWORDS = [
    "quebrado", "quebrada", "defeito", "defeituoso", "atraso", "atrasado",
    "não chegou", "nao chegou", "reembolso", "cancelar", "cancelado",
    "péssimo", "pessimo", "ruim", "nunca mais", "não recomendo", "nao recomendo",
    "não respondeu", "nao respondeu", "reclamação", "reclamacao", "errado",
    "danificado", "horrível", "horrivel", "demorou", "sumiu"
]
 
URGENT_KEYWORDS = ["reembolso", "cancelar", "nunca mais", "reclamação", "reclamacao", "processar", "advogado"]
 
def highlight_risk_words(text):
    highlighted = text
    for kw in RISK_KEYWORDS:
        if kw.lower() in text.lower():
            import re
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted = pattern.sub(f"**:red[{kw.upper()}]**", highlighted)
    return highlighted
 
def score_urgency(text):
    text_lower = text.lower()
    hits = sum(1 for kw in URGENT_KEYWORDS if kw in text_lower)
    if hits >= 2:
        return "🔴 High"
    elif hits == 1:
        return "🟠 Medium"
    return "🟢 Low"
 
# ---------------------------------------------------------------------------
# Auto-response suggestion (template-based)
# ---------------------------------------------------------------------------
def suggest_response(text, urgency):
    text_lower = text.lower()
    if "quebrado" in text_lower or "defeito" in text_lower or "danificado" in text_lower:
        template = ("Lamentamos muito que seu produto tenha chegado danificado. "
                     "Vamos providenciar a troca ou reembolso imediatamente. "
                     "Poderia nos enviar uma foto do produto para agilizar o processo?")
    elif "atraso" in text_lower or "demorou" in text_lower or "não chegou" in text_lower or "nao chegou" in text_lower:
        template = ("Pedimos desculpas pelo atraso na entrega. Estamos verificando o status "
                     "do seu pedido com a transportadora e retornaremos com uma atualização em breve.")
    elif "não respondeu" in text_lower or "nao respondeu" in text_lower:
        template = ("Sentimos muito pela falta de resposta anterior. Um de nossos atendentes "
                     "entrará em contato com você diretamente para resolver isso o quanto antes.")
    else:
        template = ("Lamentamos que sua experiência não tenha sido positiva. "
                     "Gostaríamos de entender melhor o que aconteceu — pode nos dar mais detalhes?")
 
    if urgency == "🔴 High":
        template += " Este caso foi marcado como prioridade alta para nossa equipe."
 
    return template
 
# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------
def analyze_review(text):
    result = model(text, truncation=True, max_length=128)[0]
    label = result["label"]
    score = result["score"]
    is_at_risk = label == "negative"
    return label, score, is_at_risk
 
# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🛒 ShopSense AI")
st.caption("At-risk review detector, triage dashboard, and response assistant")
 
if rf_model is not None:
    st.success("Ensemble mode active: Random Forest + Transformer signals combined.", icon="✅")
else:
    st.info("Running in transformer-only mode. Add `rf_model.pkl` to the repo to enable RF ensemble scoring.", icon="ℹ️")
 
tab1, tab2, tab3 = st.tabs(["📝 Single Review", "📂 Batch Upload", "📊 History & Dashboard"])
 
# --- TAB 1: Single review ---
with tab1:
    review = st.text_area(
        "Paste a customer review (Portuguese or English):",
        placeholder="Produto chegou quebrado e a loja não respondeu",
        height=120,
        key="single_review_input"
    )
 
    if st.button("Analyze", type="primary", key="analyze_single") and review.strip():
        with st.spinner("Analyzing..."):
            label, score, is_at_risk = analyze_review(review)
            urgency = score_urgency(review) if is_at_risk else "🟢 Low"
            save_review(review, label, score, is_at_risk, urgency)
 
        st.divider()
 
        if is_at_risk:
            st.error(f"⚠️ AT-RISK REVIEW — {score:.1%} confidence  |  Urgency: {urgency}")
            st.markdown("**Flagged text:**")
            st.markdown(highlight_risk_words(review))
 
            st.markdown("**Suggested response:**")
            st.text_area("Suggested reply (editable):", value=suggest_response(review, urgency), height=100, key="suggested_reply")
        else:
            st.success(f"✅ Not flagged — sentiment: **{label}** ({score:.1%} confidence)")
 
        with st.expander("Raw model output"):
            st.json({"label": label, "score": score})
 
# --- TAB 2: Batch upload ---
with tab2:
    st.write("Upload a CSV with a column of review text to triage many reviews at once.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
 
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
 
        text_col = st.selectbox("Which column contains the review text?", df.columns)
 
        # Check whether this CSV has the structured columns the RF model needs
        use_ensemble = False
        if rf_model is not None and rf_columns is not None:
            missing_cols = [c for c in rf_columns if c not in df.columns]
            if not missing_cols:
                use_ensemble = True
                st.success("This CSV has all the structured columns needed — ensemble scoring enabled for this run.")
            else:
                st.warning(f"RF model loaded, but this CSV is missing {len(missing_cols)} required column(s) "
                           f"(e.g. {missing_cols[:3]}). Falling back to transformer-only scoring.")
 
        if st.button("Run batch analysis", type="primary"):
            progress = st.progress(0)
            results = []
            texts = df[text_col].astype(str).tolist()
 
            # If ensembling, get RF's probability of "at-risk" for every row up front
            rf_probs = None
            if use_ensemble:
                rf_probs = rf_model.predict_proba(df[rf_columns])[:, 1]  # prob of positive/at-risk class
 
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                outputs = model(batch, truncation=True, max_length=128)
                for j, (text, out) in enumerate(zip(batch, outputs)):
                    row_idx = i + j
                    transformer_score = out["score"] if out["label"] == "negative" else 1 - out["score"]
 
                    if use_ensemble:
                        rf_score = rf_probs[row_idx]
                        # Weighted average: RF is more precise, transformer catches more — blend evenly
                        ensemble_score = 0.5 * rf_score + 0.5 * transformer_score
                        is_at_risk = ensemble_score >= 0.5
                        final_score = ensemble_score
                    else:
                        is_at_risk = out["label"] == "negative"
                        final_score = out["score"]
 
                    urgency = score_urgency(text) if is_at_risk else "🟢 Low"
                    results.append({
                        "review_text": text,
                        "label": out["label"],
                        "transformer_score": out["score"],
                        "rf_score": rf_probs[row_idx] if use_ensemble else None,
                        "final_score": final_score,
                        "is_at_risk": is_at_risk,
                        "urgency": urgency
                    })
                    save_review(text, out["label"], final_score, is_at_risk, urgency)
                progress.progress(min((i + batch_size) / len(texts), 1.0))
 
            results_df = pd.DataFrame(results)
            mode_label = "ensemble (RF + Transformer)" if use_ensemble else "transformer-only"
            st.success(f"Analyzed {len(results_df)} reviews using {mode_label} scoring. "
                       f"{results_df['is_at_risk'].sum()} flagged as at-risk.")
            st.dataframe(results_df)
 
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="shopsense_batch_results.csv",
                mime="text/csv"
            )
 
# --- TAB 3: History & Dashboard ---
with tab3:
    history_df = get_all_reviews()
 
    if history_df.empty:
        st.write("No reviews analyzed yet. Try the Single Review or Batch Upload tabs.")
    else:
        total = len(history_df)
        at_risk_count = history_df["is_at_risk"].sum()
        at_risk_pct = at_risk_count / total * 100
 
        col1, col2, col3 = st.columns(3)
        col1.metric("Total reviews analyzed", total)
        col2.metric("At-risk reviews", int(at_risk_count))
        col3.metric("At-risk rate", f"{at_risk_pct:.1f}%")
 
        st.subheader("At-risk reviews over time")
        history_df["date"] = pd.to_datetime(history_df["timestamp"]).dt.date
        daily = history_df[history_df["is_at_risk"] == 1].groupby("date").size()
        if not daily.empty:
            st.bar_chart(daily)
        else:
            st.write("No at-risk reviews yet.")
 
        st.subheader("Flagged reviews")
        flagged = history_df[history_df["is_at_risk"] == 1].copy()
 
        for _, row in flagged.iterrows():
            with st.expander(f"[{row['urgency']}] {row['review_text'][:80]}..."):
                st.write(f"**Full text:** {row['review_text']}")
                st.write(f"**Confidence:** {row['score']:.1%}  |  **Timestamp:** {row['timestamp']}")
                st.write(f"**Status:** {'✅ Resolved' if row['resolved'] else '🔴 Open'}")
                if not row["resolved"]:
                    if st.button("Mark as resolved", key=f"resolve_{row['id']}"):
                        mark_resolved(row["id"])
                        st.rerun()
 
        st.download_button(
            "Download full history as CSV",
            data=history_df.to_csv(index=False),
            file_name="shopsense_full_history.csv",
            mime="text/csv"
        )

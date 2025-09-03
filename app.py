import pickle
import streamlit as st
from utils import extract_text, clean_text

# ---- Load model ----
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    VECTORIZER = data["vectorizer"]
    MODEL = data["model"]

# ---- Default keywords ----
DEFAULT_KEYWORDS = [
    "python", "numpy", "pandas", "scikit-learn", "ml", "machine learning",
    "django", "flask", "sql", "git", "docker", "aws", "linux"
]

# ---- Keyword score ----
def keyword_score(text, keywords):
    text_l = text.lower()
    found, missing = [], []
    for kw in keywords:
        (found if kw.lower() in text_l else missing).append(kw)
    score = (len(found) / max(1, len(keywords))) * 100.0
    return score, found, missing

# ---- ML probability ----
def ml_probability(text):
    vec = VECTORIZER.transform([text])
    return MODEL.predict_proba(vec)[0][1] * 100.0

# ---- Combined scoring ----
def combined_score(text, keywords, alpha=0.6):
    k_score, found, missing = keyword_score(text, keywords)
    m_score = ml_probability(text)
    final = alpha * m_score + (1 - alpha) * k_score
    return {"keyword_score": k_score, "ml_score": m_score, "final": final, "found": found, "missing": missing}

# ---- Streamlit UI ----
st.title("🧠 AI Resume Screening Tracker")
st.caption("Upload a resume (PDF, DOCX, Image). Extracts text → ML model → Fit %")

role = st.sidebar.text_input("Target Role", "Python / AI-ML Engineer")
kw_text = st.sidebar.text_area("Target keywords (comma separated)", ", ".join(DEFAULT_KEYWORDS))
keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

upload = st.file_uploader("Upload Resume", type=["pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff"])

if upload:
    raw = extract_text(upload)
    text = clean_text(raw)

    if text:
        results = combined_score(text, keywords)

        st.subheader("📊 Results")
        st.metric("Final Fit %", f"{results['final']:.1f}%")
        st.progress(min(max(results['final']/100, 0), 1))

        st.write(f"ML Score: {results['ml_score']:.1f}%")
        st.write(f"Keyword Score: {results['keyword_score']:.1f}%")

        st.subheader("✅ Found Keywords")
        st.write(", ".join(results['found']) or "None")

        st.subheader("❌ Missing Keywords")
        st.write(", ".join(results['missing']) or "None")

        st.subheader("📝 Extracted Text (Preview)")
        st.text_area("Text", text[:3000], height=300)
    else:
        st.error("No text extracted. Try another file.")

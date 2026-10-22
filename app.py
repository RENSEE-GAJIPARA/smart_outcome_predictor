import streamlit as st
import pandas as pd
import pickle
import json
import os

st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎓", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(90deg,#4C78A8,#7b5ea7);padding:1.6rem 2rem;border-radius:14px;color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.3rem 0 0;opacity:.9}
.card{padding:1.2rem;border-radius:12px;text-align:center;color:white;font-size:1.1rem}
.good{background:#2e9e5b}.bad{background:#d9534f}.info{background:#4C78A8}
.card b{font-size:2rem;display:block}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎓 Smart Outcome Predictor</h1>'
            '<p>Ensemble-based predictions of course completion and final score for online students</p></div>', unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load():
    m = lambda f: os.path.join(BASE, "Models", f)
    return (pickle.load(open(m("completion_pipeline.pkl"), "rb")), pickle.load(open(m("score_pipeline.pkl"), "rb")),
            json.load(open(m("final_metrics.json"))))
try:
    cls_pipe, reg_pipe, info = load()
except FileNotFoundError:
    st.error("Model files not found. Run the notebook first so it creates the Models/ folder."); st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

with tab1:
    with st.form("student"):
        st.subheader("Student profile")
        c = st.columns(5); row = {}
        for col, name in zip(c, info["cat"]):
            row[name] = col.selectbox(name.replace("_", " ").title(), info["cat_options"][name])
        st.subheader("Engagement & assessment")
        c = st.columns(3)
        for i, name in enumerate(info["num"]):
            lo, hi, med = info["num_ranges"][name]
            label = name.replace("_", " ").title()
            if name in ("age", "week_of_year", "sessions", "videos_watched", "quiz_attempts", "assignments_submitted", "forum_posts"):
                row[name] = c[i % 3].slider(label, int(lo), int(hi), int(med))
            else:
                row[name] = c[i % 3].slider(label, float(lo), float(hi), float(med))
        go = st.form_submit_button("Predict outcome", width='stretch', type="primary")
    if go:
        X = pd.DataFrame([row])[info["features"]]
        prob = float(cls_pipe.predict_proba(X)[0, 1]); score = float(min(100, max(0, reg_pipe.predict(X)[0])))
        a, b, d = st.columns(3)
        ok = prob >= 0.5
        a.markdown(f'<div class="card {"good" if ok else "bad"}">Completion<b>{"Likely to complete" if ok else "At risk"}</b></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="card info">Completion probability<b>{prob:.0%}</b></div>', unsafe_allow_html=True)
        d.markdown(f'<div class="card info">Predicted final score<b>{score:.1f} / 100</b></div>', unsafe_allow_html=True)
        st.progress(prob, text=f"Completion probability: {prob:.1%}")
        st.progress(score / 100, text=f"Predicted final score: {score:.1f}")
        if ok: st.success("This student is on track. Keep up the current engagement.")
        else: st.warning("This student may drop out. Consider reminders, mentoring or extra quiz practice.")

with tab2:
    st.subheader(f"Classification - {info['classifier']}")
    cols = st.columns(5)
    for col, (k, v) in zip(cols, info["classification_metrics"].items()): col.metric(k, v)
    st.subheader(f"Regression - {info['regressor']}")
    cols = st.columns(3)
    for col, (k, v) in zip(cols, info["regression_metrics"].items()): col.metric(k, v)
    st.markdown("**All classification models (test set)**")
    st.dataframe(pd.DataFrame(info["all_classification"]).set_index("Model"), width='stretch')
    st.markdown("**All regression models (test set)**")
    st.dataframe(pd.DataFrame(info["all_regression"]).set_index("Model"), width='stretch')

with tab3:
    st.markdown("""
**Pipeline:** raw student data -> scaling + KNN imputation + one-hot encoding -> ensemble model (Stacking).

**Models compared:** Decision Tree, Bagging, AdaBoost, XGBoost, Hard/Soft Voting, Stacking.

**Tasks:** (1) Will the student complete the course? (2) What will the final score be?

Predictions are estimates to support early intervention, not final decisions.""")
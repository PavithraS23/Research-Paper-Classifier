import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

st.set_page_config(page_title="Research Paper Classifier", layout="wide")

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("small_dataset.csv")

# LOAD MODEL
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

df = load_data()
model, vectorizer = load_model()

# HISTORY FILE
HISTORY_FILE = "history.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["text","prediction","confidence"])

def save_history(entry):
    df_hist = load_history()
    df_hist = pd.concat([df_hist, pd.DataFrame([entry])], ignore_index=True)
    df_hist.to_csv(HISTORY_FILE, index=False)

# SIDEBAR
page = st.sidebar.radio("Navigation", ["Dashboard","Predict","Performance"])

# ================= DASHBOARD =================
if page == "Dashboard":
    st.title("📊 Dataset Overview")

    total = len(df)
    categories = df['category'].nunique()
    top_cat = df['category'].value_counts().idxmax()
    avg_len = int(df['summary'].str.split().apply(len).mean())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Papers", total)
    c2.metric("Categories", categories)
    c3.metric("Top Category", top_cat)
    c4.metric("Avg Length", avg_len)

    st.subheader("Category Distribution")
    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns=['Category','Count']

    fig = px.bar(cat_counts, x='Category', y='Count',
                 color='Count', color_continuous_scale='Blues')
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ================= PREDICT =================
elif page == "Predict":
    st.title("🤖 Predict Paper Category")

    text = st.text_area("Enter Abstract", height=200)

    if st.button("Predict"):
        if text.strip()=="":
            st.warning("Enter text")
        else:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]

            confidence = max(probs)*100

            st.success(f"Prediction: {pred}")
            st.write(f"Confidence: {confidence:.2f}%")

            save_history({
                "text": text[:100],
                "prediction": pred,
                "confidence": round(confidence,2)
            })

# ================= PERFORMANCE =================
elif page == "Performance":
    st.title("📈 Model Performance")

    df_hist = load_history()

    if df_hist.empty:
        st.info("No predictions yet")
    else:
        st.metric("Total Predictions", len(df_hist))
        st.metric("Avg Confidence", f"{df_hist['confidence'].mean():.2f}%")

        cat_counts = df_hist['prediction'].value_counts().reset_index()
        cat_counts.columns=['Category','Count']

        fig = px.bar(cat_counts, x='Category', y='Count',
                     color='Count', color_continuous_scale='Viridis')
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LOAD DATA
df = pd.read_csv("small_dataset.csv")
df = df[['summary', 'category']].dropna()

# ==============================
# 🔥 CLEAN & MERGE LABELS
# ==============================
def clean_category(cat):
    if "Machine Learning" in cat:
        return "Machine Learning"
    elif "Vision" in cat:
        return "Computer Vision"
    elif "Language" in cat:
        return "NLP"
    elif "Artificial Intelligence" in cat:
        return "AI"
    else:
        return "Other"

df['category'] = df['category'].apply(clean_category)

# REMOVE "Other"
df = df[df['category'] != "Other"]

print("Categories:\n", df['category'].value_counts())

# ==============================
# 🔥 BALANCE DATA
# ==============================
df = df.groupby('category').apply(
    lambda x: x.sample(400, replace=True, random_state=42)
).reset_index(drop=True)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df['summary'],
    df['category'],
    test_size=0.2,
    random_state=42,
    stratify=df['category']
)

# ==============================
# VECTORIZER
# ==============================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=6000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# MODEL
# ==============================
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# ==============================
# EVALUATE
# ==============================
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ==============================
# SAVE
# ==============================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ MODEL READY")

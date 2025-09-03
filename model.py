import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---- Demo dataset (you can replace with real resumes) ----
resumes = [
    "python numpy pandas scikit-learn machine learning sql django flask git docker linux",
    "data analysis python sql ml nlp pandas numpy matplotlib seaborn",
    "pytorch tensorflow deep learning cnn rnn computer vision opencv python",
    "python backend django rest api postgres docker aws kubernetes linux",
    "graphic design photoshop illustrator figma branding typography",
    "sales marketing lead generation crm negotiation customer relationship",
    "content writing copywriting blogging social media strategy seo",
    "accounts tally bookkeeping gst taxation finance audit excel"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = good fit, 0 = not fit

# ---- Train TF-IDF + Logistic Regression ----
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X = vectorizer.fit_transform(resumes)
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# ---- Save model + vectorizer ----
with open("model.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "model": model}, f)

print("✅ Model trained and saved as model.pkl")

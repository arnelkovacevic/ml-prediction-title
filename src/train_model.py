# -----------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------------------------
# Load dataset (adjust URL or path if needed)
df = pd.read_csv("https://raw.githubusercontent.com/arnelkovacevic/ml-prediction-title/main/data/products.csv")

# Clean column names
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# Drop rows with no title or category
df = df.dropna(subset=["product_title", "category_label"]).reset_index(drop=True)
print("Rows after drop:", df.shape)

# Parse listing date
df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')

# Fill numeric NaNs with median
df['merchant_rating'] = pd.to_numeric(df['merchant_rating'], errors='coerce').fillna(df['merchant_rating'].median())
df['number_of_views'] = pd.to_numeric(df['number_of_views'], errors='coerce').fillna(df['number_of_views'].median())

# Clean titles
df['product_title'] = df['product_title'].astype(str).str.lower().str.strip().fillna('')

# Fix bad ratings
df.loc[df['merchant_rating'] > 5, 'merchant_rating'] = df['merchant_rating'].median()
df.loc[df['merchant_rating'] < 0, 'merchant_rating'] = df['merchant_rating'].median()

# Drop duplicates (keep first)
df = df.drop_duplicates(subset=['product_title', 'merchant_id']).reset_index(drop=True)

# Group small categories into 'Other'
min_count = 50
small_cats = df['category_label'].value_counts()[lambda x: x < min_count].index.tolist()
df['category_label_reduced'] = df['category_label'].apply(lambda x: x if x not in small_cats else 'Other')

# -----------------------------------------------
# Simple features
df['title_len'] = df['product_title'].str.len()
df['title_word_count'] = df['product_title'].str.split().str.len()
df['has_digit'] = df['product_title'].str.contains(r'\d').astype(int)
df['has_plus'] = df['product_title'].str.contains(r'\+').astype(int)
df['num_uppercase_words'] = df['product_title'].apply(lambda t: sum(1 for w in t.split() if w.isupper()))

# Brand flag (simple first token check)
brands = ['apple','samsung','sony','bosch','kenwood','smeg','olympus']
df['first_token'] = df['product_title'].str.split().str[0]
df['brand_flag'] = df['first_token'].isin(brands).astype(int)

# -----------------------------------------------
# Split data
X = df[['product_title','title_len','title_word_count','has_digit','has_plus','num_uppercase_words','brand_flag']]
y = df['category_label_reduced']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------------------------
# Preprocessors
preprocessor_std = ColumnTransformer([
    ("title", TfidfVectorizer(ngram_range=(1,2), min_df=3), "product_title"),
    ("nums", StandardScaler(), ["title_len","title_word_count","has_digit","has_plus","num_uppercase_words","brand_flag"])
])

preprocessor_nb = ColumnTransformer([
    ("title", TfidfVectorizer(ngram_range=(1,2), min_df=3), "product_title"),
    ("nums", MinMaxScaler(), ["title_len","title_word_count","has_digit","has_plus","num_uppercase_words","brand_flag"])
])

# -----------------------------------------------
# Models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Linear SVC": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Multinomial NB": MultinomialNB()
}

# Train & evaluate
results = {}
for name, model in models.items():
    print("\n======================")
    print("MODEL:", name)
    
    pipe = Pipeline([
        ("preprocessing", preprocessor_nb if name=="Multinomial NB" else preprocessor_std),
        ("classifier", model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    results[name] = (acc, pipe)

# -----------------------------------------------
# Pick best model
best_name = max(results, key=lambda n: results[n][0])
best_acc, best_pipe = results[best_name]
print("\n======================")
print("BEST MODEL:", best_name)
print("Accuracy:", best_acc)
print("======================")

# Save

joblib.dump(best_pipe, "model_final.joblib")
print("Model saved as model_final.joblib")
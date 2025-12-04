import joblib
import os
import pandas as pd
import re

# Folder where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one folder UP, then load the model
model_path = os.path.join(BASE_DIR, "..", "model_final.joblib")
model_path = os.path.abspath(model_path)

model = joblib.load(model_path)
print("✅ Model loaded successfully!")

print("Type 'exit' at any time to quit.\n")

# -----------------------------
# Helper functions
# -----------------------------
def count_uppercase_words(text):
    return sum(1 for w in text.split() if w.isupper())

def has_brand(text):
    brands = ["samsung", "apple", "bosch", "lg", "sony", "lenovo", "hp"]
    t = text.lower()
    return any(b in t for b in brands)

# -----------------------------
# Interactive loop
# -----------------------------
while True:
    title = input("📝 Enter product title: ")

    # Exit
    if title.lower() == "exit":
        print("Exiting...")
        break

    # Empty input check
    if not title.strip():
        print("❌ Error: The title cannot be empty. Please type something.")
        print("-" * 40)
        continue

    # Extract features (must match training format)
    row = {
        "product_title": title,
        "title_len": len(title),
        "title_word_count": len(title.split()),
        "has_digit": int(bool(re.search(r"\d", title))),
        "has_plus": int("+" in title),
        "num_uppercase_words": count_uppercase_words(title),
        "brand_flag": int(has_brand(title)),
    }

    df_input = pd.DataFrame([row])

    # Predict
    prediction = model.predict(df_input)[0]

    print(f"🔎 Predicted category: {prediction}")
    print("-" * 40)

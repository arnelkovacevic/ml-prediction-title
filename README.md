# ML Product Title Classification Project

---

## Overview

This project demonstrates a **machine learning pipeline** for classifying product titles into predefined categories.
The main goal is to **automatically predict product categories** based on product titles using feature engineering, text processing, and standard ML models.

This project is designed for **learning and experimentation**, perfect for practicing:

* **Data preprocessing**
* **Feature engineering**
* **Model selection**
* **Evaluation**
* **Deployment**

---

## Project Structure

```
ml-product-title/
│
├─ data/             # Raw dataset (CSV files)
│   └─ products.csv
│
│
├─ src/              # Source code
│   ├─ train_model.py # Train models and save best pipeline
│   ├─ test_model.py  # Interactive testing script
│   └─ products_analysis.py     # Jupyter notebooks for exploration
│
├─ model_final.joblib    # model tested
│
│
├─ requirements.txt
│
│
└─ README.md         # This file
```

---

## Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/arnelkovacevic/ml-prediction-title.git
cd ml-product-reviews-project
```

2. **Create and activate a Python environment** (recommended)

```bash
conda create -n mlproj python=3.13
conda activate mlproj
```

3. **Install recommended dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Train the models**
   This will train all models and save the best one:

```bash
python src/train_model.py
```

2. **Test the model interactively**

```bash
python src/test_model.py
```

---

## How It Works

**Data Preprocessing**

* Cleans product titles
* Fills missing values
* Removes duplicates
* Generates engineered features:

  * Title length
  * Word count
  * Digit presence
  * Uppercase words
  * Brand flag

**Modeling**
Trains multiple models:

* Logistic Regression
* Linear SVC
* Random Forest
* Multinomial Naive Bayes

**Evaluation**

* Prints **accuracy, precision, recall, F1-score** for each model

**Deployment**

* Saves the best model as a `.joblib` file, ready for interactive predictions

---

## License

This project is **free and open-source**, intended for **educational purposes and practice**.
You can freely **use, modify, and experiment** with it.

---

## Notes

* Dataset is included in the repository for practice
* Model outputs may slightly vary due to random splits and stochastic behavior of some algorithms
* Interactive testing requires entering a product title to get category predictions

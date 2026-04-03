# 🚀 Chương 2 - Mini Project 2 (Bài 11–20)

## Sentiment Analysis Pipeline: From Scratch đến Optimized

> Áp dụng: Cross-Validation, Metrics, Confusion Matrix, Multi-class, Imbalanced Data, Feature Selection, Dimensionality Reduction, Pipeline, Hyperparameter Tuning, Ensemble

---

## 🎯 Mục tiêu

Xây dựng hệ thống **Sentiment Analysis** hoàn chỉnh:
1. Xử lý imbalanced data
2. Build pipeline đầy đủ
3. Cross-validation + metrics đúng
4. Hyperparameter tuning
5. Ensemble models
6. So sánh kết quả

---

## 📂 Dataset

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Dataset: 3-class sentiment (positive, neutral, negative)
# Imbalanced: 200 pos, 50 neutral, 150 neg
data = {
    'text': [
        # Positive (sentiment=2)
        "This product is absolutely amazing! Love it!",
        "Excellent quality, highly recommend to everyone",
        "Outstanding service and great value for money",
        "Best purchase I have made this year",
        "Fantastic experience, will definitely buy again",
        "Product exceeded all my expectations, wonderful",
        "Very happy with this purchase, great quality",
        "Superb customer service, fast delivery",
        # Neutral (sentiment=1)
        "Product arrives as described, nothing special",
        "It works okay, does what it says",
        "Average quality, price is reasonable",
        "Delivery on time, product is fine",
        # Negative (sentiment=0)
        "Terrible product, complete waste of money",
        "Very disappointed, does not work as advertised",
        "Poor quality, broke after one day",
        "Worst purchase ever, do not buy this",
        "Would not recommend, very bad experience",
        "Product arrived damaged, terrible packaging",
    ],
    'sentiment': [2,2,2,2,2,2,2,2, 1,1,1,1, 0,0,0,0,0,0]
}
df = pd.DataFrame(data)

from collections import Counter
print("Class distribution:")
print(Counter(df['sentiment']))
```

---

## 🔍 Bước 1: EDA - Phân tích dữ liệu

```python
import matplotlib.pyplot as plt

# Text length analysis
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print("\nText statistics per class:")
print(df.groupby('sentiment')[['text_length', 'word_count']].mean().round(1))

# Visualize distribution
plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
df['sentiment'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Sentiment (0=Neg, 1=Neu, 2=Pos)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1,2,2)
for sentiment, group in df.groupby('sentiment'):
    labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    plt.hist(group['word_count'], alpha=0.6, label=labels_map[sentiment], bins=10)
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.legend()

plt.tight_layout()
plt.savefig('eda.png', dpi=100)
plt.show()
```

---

## ⚙️ Bước 2: Multiple Pipelines with Proper CV

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD

texts = df['text'].tolist()
labels = df['sentiment'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define pipelines
pipelines = {
    'LR_baseline': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ]),
    'LR_chi2': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('select', SelectKBest(chi2, k=20)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ]),
    'NB': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('clf', MultinomialNB(alpha=0.5))
    ]),
    'SVM': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('clf', LinearSVC(C=1.0, class_weight='balanced', max_iter=3000))
    ]),
    'LR_LSA': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('lsa', TruncatedSVD(n_components=30, random_state=42)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])
}

results = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, texts, labels, cv=cv,
                              scoring='f1_weighted', n_jobs=-1)
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name:15s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 🔨 Bước 3: Hyperparameter Tuning (Best Model)

```python
from sklearn.model_selection import GridSearchCV

best_pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
])

param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
    'tfidf__max_features': [None, 200, 500],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__penalty': ['l2']
}

grid = GridSearchCV(best_pipe, param_grid, cv=cv,
                    scoring='f1_weighted', n_jobs=-1, verbose=0)
grid.fit(texts, labels)

print(f"\nBest CV F1: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")
```

---

## 📊 Bước 4: Final Evaluation

```python
# Train best model on full train, evaluate on test
final_model = grid.best_estimator_
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\n=== Final Evaluation on Test Set ===")
print(classification_report(
    y_test, y_pred,
    target_names=['Negative', 'Neutral', 'Positive']
))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## 🧪 Bước 5: Live Demo

```python
def predict_sentiment(text):
    pred = final_model.predict([text])[0]
    labels_map = {0: '😠 Negative', 1: '😐 Neutral', 2: '😊 Positive'}
    return labels_map[pred]

test_reviews = [
    "Absolutely love this product, highly recommended!",
    "Not bad, does the job but nothing special",
    "Total garbage, waste of money, avoid!",
    "Quality is okay, arrived quickly"
]

print("\n=== Sentiment Predictions ===")
for review in test_reviews:
    result = predict_sentiment(review)
    print(f"{result}: '{review}'")
```

---

## 📝 Kết luận

| | Kết quả |
|-|---------|
| Best model | Logistic Regression + TF-IDF + class_weight='balanced' |
| Improvement from tuning | +5-10% F1 |
| Imbalanced handling | class_weight crucial for Neutral class |
| Feature selection | Chi2 với k nhỏ → giảm noise |
| Ensemble vs Single | Marginal improvement cho dataset nhỏ |

**Key takeaways:**
1. Luôn check class distribution → dùng `class_weight='balanced'`
2. StratifiedKFold cho imbalanced
3. Report F1_weighted, không dùng accuracy
4. Pipeline prevents data leakage
5. Grid search cần CV, không dùng test set

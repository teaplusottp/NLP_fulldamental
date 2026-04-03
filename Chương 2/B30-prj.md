# 🚀 Chương 2 - Mini Project 3 (Bài 21–26)

## End-to-End NLP System: News Category Classifier

> Áp dụng: Error Analysis, Interpretability, Deployment Basic, Spam/Sentiment Practice, ML vs DL comparison

---

## 🎯 Mục tiêu

Build **News Article Category Classifier** production-ready:
1. Multi-class classification (4 categories)
2. Error analysis đầy đủ
3. Model interpretability
4. Rest API deployment
5. So sánh approaches

---

## 📦 Setup

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import re, time, joblib
import warnings
warnings.filterwarnings('ignore')
```

---

## 📂 Dataset

```python
# Simulated news articles (4 categories)
articles = [
    # Technology (0)
    ("Apple releases new iPhone with advanced AI chip and neural engine", 0),
    ("Google's new AI model surpasses human performance on coding tasks", 0),
    ("Microsoft integrates ChatGPT into Office suite for productivity boost", 0),
    ("Tesla's full self-driving software update rolls out to beta testers", 0),
    ("Meta releases open-source AI model for natural language processing", 0),
    ("New quantum computing breakthrough could revolutionize cryptography", 0),
    # Sports (1)
    ("Manchester United secures Champions League spot with late comeback win", 1),
    ("NBA Finals: Lakers edge out Heat in overtime thriller game seven", 1),
    ("Federer announces retirement after legendary twenty-year tennis career", 1),
    ("World Cup 2026: Brazil and Argentina advance to quarterfinals", 1),
    ("Usain Bolt breaks his own 100m world record at Olympic trials", 1),
    ("Formula One: Hamilton takes pole position at Monaco Grand Prix", 1),
    # Politics (2)
    ("Senate passes bipartisan infrastructure bill after months of debate", 2),
    ("President signs executive order on climate change and green energy", 2),
    ("UN Security Council meets to discuss ongoing international tensions", 2),
    ("Election results show historic voter turnout in local elections", 2),
    ("Parliament debates controversial new immigration policy reform bill", 2),
    ("Prime Minister announces budget plan for economic recovery program", 2),
    # Entertainment (3)
    ("Blockbuster superhero film breaks all-time opening weekend record", 3),
    ("Grammy Awards ceremony delivers memorable performances and upsets", 3),
    ("Netflix announces renewal of popular streaming drama for season four", 3),
    ("Hollywood actor wins Oscar for fifth time in remarkable career", 3),
    ("New album from legendary rock band tops charts in fifty countries", 3),
    ("Broadway musical adaptation receives rave reviews from critics", 3),
]

df = pd.DataFrame(articles, columns=['text', 'category'])
category_names = {0: 'Technology', 1: 'Sports', 2: 'Politics', 3: 'Entertainment'}
df['category_name'] = df['category'].map(category_names)

print(f"Total: {len(df)} articles")
print(df['category_name'].value_counts())
```

---

## ⚙️ Build & Evaluate System

```python
def preprocess(text):
    """Preprocess for news classification"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].apply(preprocess)

X = df['text_clean'].tolist()
y = df['category'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=2000, multi_class='ovr', random_state=42))
])

scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
print(f"\nCV F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Tuning
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1.0, 10.0]
}
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train, y_train)
final_model = grid.best_estimator_
y_pred = final_model.predict(X_test)

print(f"\nBest params: {grid.best_params_}")
print("\n=== Final Classification Report ===")
print(classification_report(y_test, y_pred,
                              target_names=['Tech', 'Sports', 'Politics', 'Entertainment']))
```

---

## 🔍 Error Analysis

```python
error_df = pd.DataFrame({
    'text': X_test,
    'true': [category_names[l] for l in y_test],
    'pred': [category_names[l] for l in y_pred],
    'correct': [t == p for t, p in zip(y_test, y_pred)]
})

errors = error_df[~error_df['correct']]
print(f"\n=== Errors ({len(errors)}/{len(error_df)}) ===")
if len(errors) > 0:
    print(errors[['text', 'true', 'pred']].to_string())

# Error patterns
print("\n=== Confusion Patterns ===")
print(confusion_matrix(y_test, y_pred))
```

---

## 🧠 Interpretability

```python
vec = final_model.named_steps['tfidf']
clf = final_model.named_steps['clf']
feature_names = vec.get_feature_names_out()

print("\n=== Top discriminative words per category ===")
for i, name in category_names.items():
    top_idx = np.argsort(clf.coef_[i])[-8:][::-1]
    features = [(feature_names[j], clf.coef_[i][j]) for j in top_idx if clf.coef_[i][j] > 0]
    print(f"\n{name}:")
    for feat, weight in features[:8]:
        print(f"  '{feat}': {weight:.3f}")
```

---

## 🚀 API + Demo

```python
# Save model
joblib.dump(final_model, 'news_classifier.pkl')

def classify_news(text, model=final_model, preprocess_fn=preprocess):
    """Classify a news article"""
    if preprocess_fn:
        text = preprocess_fn(text)
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    
    return {
        'category': category_names[pred],
        'confidence': float(max(proba)),
        'all_probs': {category_names[i]: round(float(p), 3) for i, p in enumerate(proba)}
    }

# Test articles
test_articles = [
    "Apple unveils new MacBook with revolutionary M3 chip performance",
    "NBA playoffs recap: Warriors dominate Eastern conference rivals",
    "Government announces new tax reform package for middle class families",
    "Movie universe expands with surprise post-credits scene revelation",
    "Scientists discover new AI algorithm that learns faster than humans"
]

print("\n=== News Classification Demo ===")
for article in test_articles:
    result = classify_news(article)
    print(f"\n📰 {article[:60]}...")
    print(f"   Category: {result['category']} ({result['confidence']:.0%} confident)")
    top_cats = sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"   Top 2: {top_cats}")
```

---

## 📊 Final Report

```python
print("\n" + "="*50)
print("FINAL SYSTEM REPORT")
print("="*50)
print(f"Model: Logistic Regression + TF-IDF Bigrams")
print(f"Best CV F1: {scores.mean():.4f}")
print(f"Test F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Model size: {len(vec.vocabulary_)} features")
print(f"Categories: {list(category_names.values())}")
print(f"Model saved: news_classifier.pkl")
print("\nNext steps:")
print("  1. Collect more real news data")
print("  2. Try BERT for better accuracy")
print("  3. Deploy with FastAPI on cloud")
```

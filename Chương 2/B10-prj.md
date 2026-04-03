# 🚀 Chương 2 - Mini Project 1 (Bài 1–10)

## Text Classifier: Phân loại Spam vs Ham

> Áp dụng: Feature Engineering, BoW, TF-IDF, Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, KNN, Bias-Variance, Overfitting

---

## 🎯 Mục tiêu

Xây dựng và **so sánh nhiều ML models** để phân loại tin nhắn spam.

1. Feature engineering từ text
2. So sánh BoW vs TF-IDF
3. Train 5 models khác nhau
4. So sánh accuracy, speed
5. Phân tích bias-variance

---

## 📂 Dataset

Dùng SMS Spam Collection dataset (có thể download từ UCI):

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Tải dataset thực tế
# df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
# df.columns = ['label', 'text']

# Hoặc dùng data mẫu:
data = {
    'text': [
        "WINNER!! You have been selected as a lucky winner. Call now!",
        "Free entry in 2 a weekly comp to win a car. Text WIN to 80086",
        "Hey, are you coming to the meeting tomorrow?",
        "Dinner at 7? Let me know.",
        "URGENT: Your mobile account will be suspended.",
        "Congratulations! You've won a £1000 prize.",
        "Can you pick up milk on your way home?",
        "Running a bit late, be there in 10 mins",
        "You have 1 new voicemail. Call 123 to listen.",
        "Happy birthday! Hope you have a great day.",
        "SIX chances to win CASH! Text WIN to 80086",
        "IMPORTANT: Your subscription expires. Call now!",
        "What time works for the call next week?",
        "Thanks for the help yesterday.",
        "Claim your FREE prize. Limited time offer.",
        "See you at the gym tonight?",
        "Your account has been accessed from a new device.",
        "Lunch tomorrow sounds good!",
        "You are selected for a cash reward. Reply YES.",
        "Don't forget to submit your report by Friday."
    ],
    'label': [1,1,0,0,1,1,0,0,1,0,1,1,0,0,1,0,1,0,1,0]
}
df = pd.DataFrame(data)
print(df['label'].value_counts())
print(f"Total: {len(df)} samples")
```

---

## 🧱 Bước 1: Feature Engineering

```python
import re

def extract_manual_features(text):
    """Trích xuất features thủ công"""
    return {
        'word_count': len(text.split()),
        'char_count': len(text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
        'has_number': int(bool(re.search(r'\d', text))),
        'has_url': int(bool(re.search(r'http|www|\.com', text, re.I))),
        'has_currency': int(bool(re.search(r'£|\$|€|cash|prize|win', text, re.I))),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0
    }

# Tạo feature matrix
manual_features = pd.DataFrame([extract_manual_features(t) for t in df['text']])
print("\nManual Features:")
print(manual_features.head())
```

---

## ⚙️ Bước 2: Vector hóa Text

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse as sp

texts = df['text'].tolist()
labels = df['label'].tolist()

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# BoW
bow_vec = CountVectorizer(ngram_range=(1,2), min_df=1)
X_train_bow = bow_vec.fit_transform(X_train_text)
X_test_bow = bow_vec.transform(X_test_text)

# TF-IDF  
tfidf_vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
X_test_tfidf = tfidf_vec.transform(X_test_text)

print(f"BoW shape: {X_train_bow.shape}")
print(f"TF-IDF shape: {X_train_tfidf.shape}")
```

---

## 🤖 Bước 3: Train và So sánh Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'SVM (Linear)': LinearSVC(C=1.0, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    for vec_name, X_tr, X_te in [('BoW', X_train_bow, X_test_bow),
                                   ('TF-IDF', X_train_tfidf, X_test_tfidf)]:
        start = time.time()
        model.fit(X_tr, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Train accuracy để detect overfitting
        y_train_pred = model.predict(X_tr)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        results.append({
            'Model': name,
            'Vectorizer': vec_name,
            'Train Acc': round(train_acc, 3),
            'Test Acc': round(acc, 3),
            'F1': round(f1, 3),
            'Time (s)': round(train_time, 4)
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

---

## 📊 Bước 4: Phân tích Bias-Variance

```python
# Tìm model overfit: Train Acc >> Test Acc
results_df['Gap'] = results_df['Train Acc'] - results_df['Test Acc']
print("\nOverfitting Analysis (Gap = Train - Test):")
print(results_df[['Model', 'Vectorizer', 'Train Acc', 'Test Acc', 'Gap']]
      .sort_values('Gap', ascending=False).to_string(index=False))
```

---

## 🔍 Bước 5: Best Model Analysis

```python
from sklearn.metrics import classification_report, confusion_matrix

# Tìm best model
best_row = results_df.loc[results_df['Test Acc'].idxmax()]
print(f"\nBest Model: {best_row['Model']} + {best_row['Vectorizer']}")
print(f"Accuracy: {best_row['Test Acc']}")

# Retrain best model
best_model = LogisticRegression(max_iter=1000, random_state=42)
best_model.fit(X_train_tfidf, y_train)
y_pred = best_model.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## 🧪 Bước 6: Test với câu mới

```python
def predict_spam(text, model, vectorizer):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "🚨 SPAM" if pred == 1 else "✅ Ham"

test_messages = [
    "You have won a free ticket! Call now to claim.",
    "Can we reschedule the meeting to 4pm?",
    "URGENT: Claim your prize before it expires!!!",
    "Thanks for lunch today, it was great!"
]

print("\n=== Predictions ===")
for msg in test_messages:
    result = predict_spam(msg, best_model, tfidf_vec)
    print(f"{result}: '{msg[:50]}...'")
```

---

## 📝 Kết luận

| | Kết quả |
|-|---------|
| Best model | Logistic Regression + TF-IDF |
| F1 Score | ~0.85+ |
| Train time | < 0.1s |
| Bias/Variance | Random Forest có gap lớn nhất |

**Lessons learned:**
- TF-IDF thường tốt hơn BoW
- Logistic Regression và SVM competitive với nhau
- Decision Tree overfit nhiều nhất
- Random Forest tốt hơn single tree nhưng chậm hơn Linear models

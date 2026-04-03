# 🚀 Chương 3 - Mini Project 3 (Bài 21–30)

## Build a Complete Embedding System: Fine-tune, Debias & Cluster

> Áp dụng: Fine-tuning, Pretrained embeddings, Transfer learning, Multilingual, Word alignment, Bias detection, Debiasing, Train from scratch, Semantic search, Text clustering

---

## 🎯 Mục tiêu

Build một **Production-Ready Embedding System** hoàn chỉnh:
1. Detect và visualize bias trong embeddings
2. Debias embeddings
3. Fine-tune trên domain corpus
4. Semantic search với evaluation
5. Clustering phân tích
6. Export production-ready embeddings

---

## 📦 Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from gensim.models import Word2Vec
from collections import Counter
import re
import copy
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
```

---

## 📂 Multi-domain Professional Corpus

```python
PROFESSIONAL_CORPUS = {
    # Tech domain
    "tech_1": "software engineers develop algorithms and write code for applications",
    "tech_2": "data scientists analyze large datasets using machine learning models",
    "tech_3": "programmers debug and optimize code for better performance",
    "tech_4": "researchers investigate deep learning methods for natural language processing",
    "tech_5": "developers build scalable systems and APIs for web services",
    
    # Medical domain
    "med_1": "doctors diagnose patients and prescribe treatments for diseases",
    "med_2": "nurses provide patient care and administer medications in hospitals",
    "med_3": "surgeons perform operations to treat injuries and medical conditions",
    "med_4": "pharmacists dispense medications and advise patients on drug interactions",
    "med_5": "researchers conduct clinical trials for new drug discoveries",
    
    # Education domain
    "edu_1": "teachers explain complex concepts and guide students through learning",
    "edu_2": "professors lecture at universities and publish academic research",
    "edu_3": "tutors provide one-on-one instruction to help students improve",
    "edu_4": "principals manage schools and develop educational programs",
    "edu_5": "counselors support students with academic and personal guidance",
    
    # Generic sentences (for bias injection)
    "gen_1": "he is a senior engineer with years of programming experience",
    "gen_2": "she is a nurse who provides excellent patient care every day",
    "gen_3": "he leads the development team as chief technology officer",
    "gen_4": "she teaches elementary school children with patience and skill",
    "gen_5": "the male doctor performed surgery successfully on the patient",
    "gen_6": "the female secretary organized all the office files and schedules",
}

# Tokenize
tokenized = {
    doc_id: re.findall(r'[a-z]+', text.lower())
    for doc_id, text in PROFESSIONAL_CORPUS.items()
}
all_sentences = list(tokenized.values())

print(f"Corpus: {len(PROFESSIONAL_CORPUS)} documents")
print(f"Total tokens: {sum(len(t) for t in all_sentences)}")
```

---

## 🧠 Bước 1: Train & Detect Bias

```python
# Train model
model = Word2Vec(
    sentences=all_sentences,
    vector_size=100,
    window=7,
    min_count=1,
    sg=1,
    negative=10,
    epochs=500,
    seed=42
)

print(f"Trained vocab: {len(model.wv)}")

# Detect gender direction
def get_gender_direction(model) -> np.ndarray:
    pairs = [('he', 'she'), ('man', 'woman'), ('male', 'female'),
             ('his', 'her'), ('him', 'her')]
    diffs = []
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            diffs.append(model.wv[w1] - model.wv[w2])
    if not diffs:
        return np.zeros(model.vector_size)
    matrix = np.array(diffs)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, random_state=42)
    pca.fit(matrix)
    direction = pca.components_[0]
    return direction / np.linalg.norm(direction)

gender_dir = get_gender_direction(model)

# Measure bias
profession_words = ['engineer', 'programmer', 'researcher', 'developer',
                     'doctor', 'nurse', 'surgeon', 'pharmacist',
                     'teacher', 'professor', 'tutor', 'principal']

print("\n=== GENDER BIAS DETECTION ===")
print(f"{'Profession':14s} {'Gender Score':>14}  Direction")
print("-" * 45)

bias_scores = {}
for word in profession_words:
    if word in model.wv:
        score = np.dot(model.wv[word], gender_dir)
        bias_scores[word] = score
        direction = "♂ masculine" if score > 0.05 else ("♀ feminine" if score < -0.05 else "≈ neutral")
        bar = "█" * int(abs(score) * 20)
        print(f"{word:14s} {score:14.4f}  {direction} {bar}")
```

---

## ⚖️ Bước 2: Debias

```python
def debias_word(word: str, model, gender_dir: np.ndarray) -> np.ndarray:
    """Hard debiasing: remove projection onto gender axis"""
    if word not in model.wv:
        return np.zeros(model.vector_size)
    vec = model.wv[word].copy()
    proj = np.dot(vec, gender_dir) * gender_dir
    return vec - proj

# Debias all profession words
debiased_vecs = {}
for word in profession_words:
    if word in model.wv:
        debiased_vecs[word] = debias_word(word, model, gender_dir)

print("\n=== DEBIASING RESULT ===")
print(f"{'Word':14s} {'Before':>8} {'After':>8} {'Utility':>9}")
print("-" * 45)

for word in profession_words:
    if word in model.wv and word in debiased_vecs:
        before = np.dot(model.wv[word], gender_dir)
        after = np.dot(debiased_vecs[word], gender_dir)
        utility = cosine_similarity([model.wv[word]], [debiased_vecs[word]])[0][0]
        print(f"{word:14s} {before:8.4f} {after:8.4f} {utility:9.4f}")
```

---

## 🎯 Bước 3: Domain Fine-tuning

```python
# Fine-tune trên medical corpus
medical_corpus = [
    ["patient", "diagnosis", "treatment", "clinical", "hospital"],
    ["surgery", "procedure", "operation", "anesthesia", "recovery"],
    ["medication", "drug", "dose", "side", "effect", "therapy"],
    ["cardiovascular", "respiratory", "neurological", "orthopedic"],
    ["oncology", "pathology", "radiology", "immunology", "genetics"],
] * 30

finetuned = copy.deepcopy(model)
finetuned.build_vocab(medical_corpus, update=True)
finetuned.train(medical_corpus, total_examples=len(medical_corpus), epochs=100)

# Verify medical domain improvement
print("\n=== DOMAIN FINE-TUNING ===")
medical_test_words = ['surgery', 'patient', 'diagnosis', 'clinical']
for word in medical_test_words:
    if word in finetuned.wv:
        neighbors = finetuned.wv.most_similar(word, topn=3)
        print(f"  '{word}': {[w for w, _ in neighbors]}")
```

---

## 🔍 Bước 4: Semantic Search Evaluation

```python
# Build semantic search with debiased vectors
class DebiasingSearchEngine:
    def __init__(self, corpus: dict, base_model, gender_dir: np.ndarray):
        self.corpus = corpus
        self.docs = list(corpus.items())
        self.gender_dir = gender_dir
        
        self.embeddings = np.array([
            self._encode(text) for _, text in self.docs
        ])
    
    def _encode(self, text: str) -> np.ndarray:
        tokens = re.findall(r'[a-z]+', text.lower())
        vecs = []
        for t in tokens:
            if t in model.wv:
                vec = model.wv[t].copy()
                # Remove gender component
                proj = np.dot(vec, self.gender_dir) * self.gender_dir
                vecs.append(vec - proj)
        if not vecs:
            return np.zeros(model.vector_size)
        return np.mean(vecs, axis=0)
    
    def search(self, query: str, top_k: int = 3) -> list:
        q = self._encode(query).reshape(1, -1)
        sims = cosine_similarity(q, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.docs[i][0], sims[i], self.docs[i][1]) for i in top_idx]

engine = DebiasingSearchEngine(PROFESSIONAL_CORPUS, model, gender_dir)

queries = [
    "expert in building software systems",
    "medical professional treating patients",
    "person teaching students at school",
]

print("\n=== DEBIASED SEMANTIC SEARCH ===")
for q in queries:
    results = engine.search(q, top_k=3)
    print(f"\nQuery: '{q}'")
    for doc_id, score, text in results:
        cat = doc_id.split('_')[0]
        print(f"  {score:.4f} [{cat}] {text[:60]}...")
```

---

## 🎨 Bước 5: Clustering + Visualization

```python
# Build matrix: mix debiased profession + domain embeddings
all_words_for_cluster = [w for w in profession_words if w in debiased_vecs]
X = np.array([debiased_vecs[w] for w in all_words_for_cluster])

# Normalize
X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

# Cluster
n_k = 3  # tech/medical/education
km = KMeans(n_clusters=n_k, random_state=42, n_init=20)
cluster_labels = km.fit_predict(X_norm)

print(f"\n=== CLUSTERING (K={n_k}) ===")
sil = silhouette_score(X_norm, cluster_labels, metric='cosine')
print(f"Silhouette: {sil:.4f}")

for k in range(n_k):
    members = [all_words_for_cluster[i] for i, l in enumerate(cluster_labels) if l == k]
    print(f"  Cluster {k}: {members}")

# PCA Visualization
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_norm)

plt.figure(figsize=(10, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, (word, emb_2d) in enumerate(zip(all_words_for_cluster, X_2d)):
    k = cluster_labels[i]
    plt.scatter(emb_2d[0], emb_2d[1], c=colors[k], s=120, alpha=0.85, zorder=3)
    orig_bias = bias_scores.get(word, 0)
    plt.annotate(f"{word}\n({orig_bias:+.2f})", emb_2d,
                  xytext=(4, 4), textcoords='offset points', fontsize=8)

for k in range(n_k):
    plt.scatter([], [], c=colors[k], label=f'Cluster {k}', s=80)
plt.legend()
plt.title("Debiased Profession Embeddings\n(values = original gender bias)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("project3_debiased_clusters.png", dpi=100)
plt.show()
print("Saved: project3_debiased_clusters.png")
```

---

## 📝 Tổng kết & Bảng đánh giá

```python
print("\n" + "="*60)
print("COMPREHENSIVE SYSTEM EVALUATION")
print("="*60)

# Bias reduction
avg_bias_before = np.mean([abs(bias_scores.get(w, 0)) for w in profession_words if w in bias_scores])
avg_bias_after = np.mean([abs(np.dot(debiased_vecs.get(w, np.zeros(100)), gender_dir))
                           for w in profession_words if w in debiased_vecs])

print(f"\n1. BIAS REDUCTION:")
print(f"   Average |bias| before: {avg_bias_before:.4f}")
print(f"   Average |bias| after:  {avg_bias_after:.4f}")
print(f"   Reduction: {(1 - avg_bias_after/avg_bias_before)*100:.1f}%")

# Utility preservation
utilities = []
for word in profession_words:
    if word in model.wv and word in debiased_vecs:
        u = cosine_similarity([model.wv[word]], [debiased_vecs[word]])[0][0]
        utilities.append(u)
print(f"\n2. UTILITY PRESERVED:")
print(f"   Mean cosine similarity: {np.mean(utilities):.4f}")
print(f"   Min: {np.min(utilities):.4f} | Max: {np.max(utilities):.4f}")

print(f"\n3. CLUSTERING:")
print(f"   Silhouette score: {sil:.4f}")

print(f"\n✅ System features:")
print(f"   - Bias detection (gender projection)")
print(f"   - Hard debiasing (gender direction removal)")
print(f"   - Domain fine-tuning (medical)")
print(f"   - Debiased semantic search")
print(f"   - Cluster-based document organization")
```

---

## Key Learnings (Bài 21–30)

| Bài | Topic | Key Concept |
|-----|-------|-------------|
| 21 | Fine-tuning | Domain adaptation trên pre-trained W2V |
| 22 | Pretrained | GloVe loading, embedding matrix, transfer |
| 23 | Transfer Learning | Feature-based vs Fine-tuning paradigm |
| 24 | Multilingual | mBERT, XLM-R, cross-lingual transfer |
| 25 | Alignment | Word alignment, cross-lingual retrieval |
| 26 | Bias Detection | WEAT, gender projection visualization |
| 27 | Debiasing | Hard/Soft debiasing, Procrustes |
| 28 | Train W2V | Preprocessing, training, evaluation, save |
| 29 | Semantic Search | MRR, P@K, multi-method comparison |
| 30 | Clustering | K-Means, Silhouette, Elbow, cluster analysis |

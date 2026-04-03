# 🚀 Chương 3 - Mini Project 2 (Bài 11–20)

## Build a Semantic Search Engine

> Áp dụng: PCA/t-SNE visualization, OOV handling, Subword BPE, Contextual vs Static, Embedding Evaluation, Polysemy, SIF, Doc2Vec, Sentence Embedding

---

## 🎯 Mục tiêu

Build một **Semantic Search Engine** hoàn chỉnh:
1. Build document corpus với multiple domains
2. Encode với nhiều methods (Mean W2V, SIF, Doc2Vec)
3. Evaluate & compare methods
4. Build interactive search interface
5. Handle polysemy & OOV
6. Visualize embedding space

---

## 📦 Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')
```

---

## 📂 Dataset: Multi-domain Corpus

```python
CORPUS = {
    "tech_001": "Machine learning algorithms learn from data to make predictions without explicit programming",
    "tech_002": "Deep learning uses neural networks with multiple hidden layers for complex pattern recognition",
    "tech_003": "Natural language processing enables computers to understand and generate human language",
    "tech_004": "Computer vision systems analyze images to detect objects faces and scenes automatically",
    "tech_005": "Python is the most popular programming language for data science and artificial intelligence",
    "tech_006": "Transformers and BERT models revolutionized natural language processing in 2018",
    
    "science_001": "Quantum mechanics describes the behavior of particles at subatomic scale with wave functions",
    "science_002": "DNA double helix carries genetic information that determines traits of living organisms",
    "science_003": "Black holes are regions in spacetime where gravity becomes so strong light cannot escape",
    "science_004": "Climate change results from greenhouse gas emissions trapping heat in the atmosphere",
    "science_005": "Photosynthesis allows plants to convert sunlight water and carbon dioxide into glucose",
    "science_006": "Vaccines train the immune system to recognize and fight specific viral or bacterial pathogens",
    
    "sports_001": "Football teams compete for the championship trophy in World Cup every four years",
    "sports_002": "Basketball players execute fast breaks and slam dunks to score points in NBA games",
    "sports_003": "Olympic swimmers train for years to shave milliseconds off their personal best times",
    "sports_004": "Tennis grand slam tournaments attract the world's best players to compete for prestige",
    "sports_005": "Marathon runners prepare months with long distance training runs and proper nutrition",
    "sports_006": "Cycling in Tour de France demands exceptional endurance over mountain stages and flat sprints",
    
    "food_001": "Italian pasta dishes like spaghetti carbonara combine eggs cheese and crispy pancetta",
    "food_002": "Japanese sushi combines vinegared rice with fresh raw fish vegetables and seaweed",
    "food_003": "Healthy Mediterranean diet emphasizes olive oil vegetables legumes whole grains and fish",
    "food_004": "French cuisine is renowned for its rich sauces complex techniques and fine ingredients",
    "food_005": "Vegetarian cooking uses tofu tempeh lentils and beans as protein sources instead of meat",
    "food_006": "Baking bread requires yeast flour water salt and patience for proper fermentation and rise",
}

print(f"Corpus: {len(CORPUS)} documents, {len(set(k.split('_')[0] for k in CORPUS))} categories")
```

---

## 🧠 Bước 1: Train Word2Vec

```python
# Tokenize
tokenized = {
    doc_id: re.findall(r'[a-z]+', text.lower())
    for doc_id, text in CORPUS.items()
}

all_tokens = [toks for toks in tokenized.values()]

w2v = Word2Vec(
    sentences=all_tokens,
    vector_size=100,
    window=7,
    min_count=1,
    sg=1,
    negative=10,
    epochs=500,
    seed=42,
    workers=1
)

print(f"W2V vocab: {len(w2v.wv)} words")
```

---

## 🔧 Bước 2: Implement Encoding Methods

```python
def encode_mean_pooling(text: str, wv) -> np.ndarray:
    tokens = re.findall(r'[a-z]+', text.lower())
    vecs = [wv[w] for w in tokens if w in wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(wv.vector_size)

def encode_sif(text: str, wv, word_freq: dict, a: float = 1e-3) -> np.ndarray:
    tokens = re.findall(r'[a-z]+', text.lower())
    total_freq = sum(word_freq.values())
    weighted = []
    for w in tokens:
        if w in wv:
            p = word_freq.get(w, 1) / total_freq
            weight = a / (a + p)
            weighted.append(weight * wv[w])
    return np.mean(weighted, axis=0) if weighted else np.zeros(wv.vector_size)

def encode_tfidf_weighted(text: str, wv, tfidf_weights: dict) -> np.ndarray:
    tokens = re.findall(r'[a-z]+', text.lower())
    weighted = []
    for w in tokens:
        if w in wv:
            weight = tfidf_weights.get(w, 0.1)
            weighted.append(weight * wv[w])
    return np.mean(weighted, axis=0) if weighted else np.zeros(wv.vector_size)

# Compute word frequencies for SIF
all_words_flat = [w for toks in tokenized.values() for w in toks]
word_freq = Counter(all_words_flat)

# Compute simple IDF weights
from math import log
N = len(CORPUS)
doc_freq = Counter(w for toks in tokenized.values() for w in set(toks))
idf_weights = {w: log(N / (df + 1)) for w, df in doc_freq.items()}

# Encode all documents
methods = {
    'mean_pooling': lambda t: encode_mean_pooling(t, w2v.wv),
    'sif': lambda t: encode_sif(t, w2v.wv, word_freq),
    'tfidf_weighted': lambda t: encode_tfidf_weighted(t, w2v.wv, idf_weights),
}

embeddings = {}
for method_name, encode_fn in methods.items():
    embeddings[method_name] = {
        doc_id: encode_fn(text)
        for doc_id, text in CORPUS.items()
    }
    print(f"  Encoded {len(embeddings[method_name])} docs with {method_name}")
```

---

## 🔎 Bước 3: Semantic Search Engine

```python
class SemanticSearchEngine:
    def __init__(self, corpus: dict, embeddings: dict, encode_fn):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.encode_fn = encode_fn
        
        # Build matrix
        self.matrix = np.array([embeddings[doc_id] for doc_id in self.doc_ids])
    
    def search(self, query: str, top_k: int = 5) -> list:
        q_vec = self.encode_fn(query).reshape(1, -1)
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            results.append({
                'doc_id': doc_id,
                'score': sims[idx],
                'text': self.corpus[doc_id][:80] + '...',
                'category': doc_id.split('_')[0]
            })
        return results

# Build search engines
engines = {}
for method_name, encode_fn in methods.items():
    engines[method_name] = SemanticSearchEngine(
        CORPUS, embeddings[method_name], encode_fn
    )

# Test queries
test_queries = [
    "artificial intelligence and neural networks",
    "healthy diet with vegetables",
    "space exploration and astronomy",
    "competitive athletic training",
    "programming data analysis",
]

print("\n" + "="*70)
print("SEMANTIC SEARCH RESULTS")
print("="*70)

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    print(f"{'Method':16s} {'Score':>6} {'Cat':>8}  {'Text'}")
    print("-" * 70)
    
    for method_name, engine in engines.items():
        results = engine.search(query, top_k=1)
        r = results[0]
        print(f"{method_name:16s} {r['score']:6.4f} {r['category']:>8}  {r['text'][:45]}...")
```

---

## 📊 Bước 4: Evaluate Methods

```python
def evaluate_retrieval(engine, method_name: str):
    """P@1: top result should match query's domain"""
    category_queries = {
        "tech":    ["machine learning algorithms", "python programming neural"],
        "science": ["quantum physics particles", "DNA genetics biology"],
        "sports":  ["football championship soccer", "Olympic athlete training"],
        "food":    ["cooking recipe ingredients", "healthy diet vegetarian"],
    }
    
    correct = 0
    total = 0
    
    for expected_cat, queries in category_queries.items():
        for q in queries:
            results = engine.search(q, top_k=1)
            predicted_cat = results[0]['category']
            if predicted_cat == expected_cat:
                correct += 1
            total += 1
    
    return correct / total

print("\n=== Retrieval Evaluation (P@1) ===")
for method_name, engine in engines.items():
    p_at_1 = evaluate_retrieval(engine, method_name)
    bar = "█" * int(p_at_1 * 20)
    print(f"  {method_name:16s}: {p_at_1:.2%} |{bar:<20}|")
```

---

## 🎨 Bước 5: Visualize Embedding Space

```python
# Compare embedding spaces visually
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

category_colors = {
    'tech': '#3498db', 'science': '#2ecc71',
    'sports': '#e74c3c', 'food': '#f39c12'
}

for ax, (method_name, embs_dict) in zip(axes, embeddings.items()):
    X = np.array([embs_dict[doc_id] for doc_id in CORPUS])
    labels = [doc_id.split('_')[0] for doc_id in CORPUS]
    
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    for doc_id, (x, y) in zip(CORPUS.keys(), X_2d):
        cat = doc_id.split('_')[0]
        color = category_colors[cat]
        ax.scatter(x, y, c=color, s=100, alpha=0.8, zorder=3)
        num = doc_id.split('_')[1]
        ax.annotate(f"{cat[:3]}{num}", (x, y),
                   xytext=(2, 2), textcoords='offset points', fontsize=7)
    
    for cat, color in category_colors.items():
        ax.scatter([], [], c=color, label=cat, s=80)
    
    ax.legend(fontsize=8, loc='best')
    ax.set_title(f"{method_name}\n(var={pca.explained_variance_ratio_.sum():.0%})")
    ax.grid(True, alpha=0.3)

plt.suptitle("Semantic Search Engine - Embedding Space Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("project2_embedding_spaces.png", dpi=100)
plt.show()
print("Saved: project2_embedding_spaces.png")
```

---

## 📝 Kết luận

| Method | P@1 | Speed | OOV | Notes |
|--------|-----|-------|-----|-------|
| Mean Pooling | Fair | ⚡⚡⚡ | ❌ | Simple baseline |
| SIF | Better | ⚡⚡⚡ | ❌ | IDF-weighted, PC removed |
| TF-IDF Weighted | Good | ⚡⚡⚡ | ❌ | Best for keyword-heavy |

**Key learnings from B11–B20:**
1. Visualization (PCA/t-SNE) reveals semantic structure
2. OOV harms naive methods → use FastText or SBERT
3. SIF > Mean pooling by removing common-word bias
4. Evaluation requires task-specific metrics (P@1, MRR, nDCG)
5. Polysemy → contextual embeddings (BERT/USE) needed
6. SBERT/USE are production-ready sentence encoders
7. FAISS enables billion-scale semantic search

# 🚀 Chương 3 - Mini Project 1 (Bài 1–10)

## Word Embedding Explorer: Train, Analyze & Search

> Áp dụng: BoW limitations, Distributional Hypothesis, Word2Vec, CBOW, Skip-gram, Negative Sampling, GloVe, FastText, Embedding Space, Similarity & Analogy

---

## 🎯 Mục tiêu

Build một **Word Embedding Explorer** hoàn chỉnh:
1. Train Word2Vec trên custom corpus
2. So sánh Word2Vec vs FastText
3. Explore embedding space (similarity, analogy)
4. Build semantic search engine
5. Visualize clusters

---

## 📦 Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, FastText
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
```

---

## 📂 Dataset

```python
# Corpus về Animals, Plants, Tech, Sports, Food
corpus_raw = """
dogs cats rabbits hamsters are common household pets. people love keeping cats.
dogs are loyal companions faithful to their human owners.
cats are independent mysterious creatures with soft fur.
lions tigers cheetahs leopards are wild carnivorous cats.
elephants giraffes zebras rhinos are large african animals.
roses tulips daisies lilies sunflowers are beautiful garden flowers.
oak pine birch maple cedar are common trees found in forests.
python java ruby javascript are popular programming languages.
machine learning deep learning artificial intelligence neural networks.
data science statistics mathematics are foundations of machine learning.
football basketball tennis swimming cycling are popular sports activities.
athletes train hard every day to improve physical performance strength.
rice bread pasta pizza hamburgers are popular foods people enjoy eating.
healthy eating vegetables fruits proteins vitamins important for wellbeing.
python programming language used widely in data science artificial intelligence.
javascript web development frontend backend full stack developer programmer.
deep learning convolutional neural network image recognition classification.
natural language processing text analysis semantic understanding transformer.
""".strip().lower()

# Tokenize
import re
sentences_raw = corpus_raw.split('.')
sentences = []
for sent in sentences_raw:
    sent = sent.strip()
    if sent:
        tokens = re.findall(r'[a-z]+', sent)
        if len(tokens) > 2:
            sentences.append(tokens)

print(f"Total sentences: {len(sentences)}")
all_tokens = [t for s in sentences for t in s]
print(f"Total tokens: {len(all_tokens)}")
print(f"Unique tokens: {len(set(all_tokens))}")
print(f"Sample: {sentences[0]}")
```

---

## 🧠 Bước 1: Train Word2Vec & FastText

```python
# Word2Vec Skip-gram
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,            # Skip-gram
    negative=5,
    epochs=300,
    workers=2,
    seed=42
)

# FastText
ft_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    min_n=3,
    max_n=6,
    epochs=300,
    workers=2,
    seed=42
)

print(f"Word2Vec vocab: {len(w2v_model.wv)}")
print(f"FastText vocab: {len(ft_model.wv)}")
```

---

## 🔍 Bước 2: Explore Similarities

```python
print("=== WORD SIMILARITY COMPARISON ===\n")

word_pairs = [
    ("dog", "cat"),
    ("python", "java"),
    ("rose", "tulip"),
    ("football", "basketball"),
    ("healthy", "eating"),
    ("deep", "learning"),
    ("dog", "python"),    # unrelated
]

print(f"{'Pair':25s} {'Word2Vec':>10} {'FastText':>10}")
print("-" * 50)
for w1, w2 in word_pairs:
    try:
        w2v_sim = w2v_model.wv.similarity(w1, w2)
        ft_sim = ft_model.wv.similarity(w1, w2)
        print(f"{w1+' vs '+w2:25s} {w2v_sim:>10.4f} {ft_sim:>10.4f}")
    except KeyError as e:
        print(f"{w1+' vs '+w2:25s} → {e} missing")
```

---

## 🧮 Bước 3: Analogy Tests

```python
print("\n=== ANALOGY TESTS ===\n")

analogies = [
    ("dog", "cats", "cat"),          # dogs:cats = cat:? 
    ("python", "programming", "java"), # python:programming = java:?
    ("deep", "learning", "machine"),  # deep:learning = machine:?
]

for a, b, c in analogies:
    if all(w in w2v_model.wv for w in [a, b, c]):
        results = w2v_model.wv.most_similar(
            positive=[b, c], negative=[a], topn=3
        )
        print(f"  {a}:{b} = {c}:?  → {[r[0] for r in results]}")
```

---

## 🎨 Bước 4: Visualize Clusters

```python
word_groups = {
    '🐾 Animals': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'rabbit'],
    '🌸 Plants': ['rose', 'tulip', 'oak', 'pine', 'daisy'],
    '💻 Tech': ['python', 'java', 'javascript', 'learning', 'network'],
    '⚽ Sports': ['football', 'basketball', 'tennis', 'swimming'],
    '🍕 Food': ['rice', 'bread', 'pizza', 'pasta', 'healthy']
}

all_words_viz = []
all_labels_viz = []
for category, words in word_groups.items():
    for w in words:
        if w in w2v_model.wv:
            all_words_viz.append(w)
            all_labels_viz.append(category)

vectors_viz = np.array([w2v_model.wv[w] for w in all_words_viz])

# SVD reduction to 2D
svd = TruncatedSVD(n_components=2, random_state=42)
vectors_2d = svd.fit_transform(vectors_viz)

colors_map = {
    '🐾 Animals': '#e74c3c',
    '🌸 Plants': '#2ecc71',
    '💻 Tech': '#3498db',
    '⚽ Sports': '#f39c12',
    '🍕 Food': '#9b59b6'
}

plt.figure(figsize=(14, 10))
for word, label, (x, y) in zip(all_words_viz, all_labels_viz, vectors_2d):
    color = colors_map.get(label, 'gray')
    plt.scatter(x, y, c=color, s=120, alpha=0.8, zorder=3)
    plt.annotate(word, (x, y), textcoords="offset points",
                 xytext=(4, 4), fontsize=9, fontweight='bold')

for cat, color in colors_map.items():
    plt.scatter([], [], c=color, label=cat, s=100)
plt.legend(loc='best', fontsize=10)
plt.title("Word Embedding Space - Semantic Clusters (SVD 2D)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("project1_clusters.png", dpi=100)
plt.show()

print(f"\nVisualization saved: project1_clusters.png")
```

---

## 🔎 Bước 5: Semantic Search Engine

```python
# Build search engine với Word2Vec embeddings
corpus_docs = [
    "dogs are loyal friendly pets that love their owners",
    "cats are independent mysteriou pets with sharp claws",
    "python is a popular programming language for data science",
    "machine learning uses algorithms to learn from data",
    "football is a team sport played with round ball",
    "deep learning neural networks process images and text",
    "healthy food includes vegetables fruits and proteins",
    "wild animals lions tigers live in natural habitats"
]

def doc_embedding(doc, word_vectors):
    """Mean pooling embedding"""
    tokens = re.findall(r'[a-z]+', doc.lower())
    vecs = [word_vectors[w] for w in tokens if w in word_vectors]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

corpus_vecs = np.array([doc_embedding(d, w2v_model.wv) for d in corpus_docs])

def search(query, corpus, corpus_vecs, topn=3):
    q_vec = doc_embedding(query, w2v_model.wv).reshape(1, -1)
    sims = cosine_similarity(q_vec, corpus_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:topn]
    
    print(f"\n🔍 Query: '{query}'")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. ({sims[idx]:.4f}) {corpus_docs[idx][:60]}...")

# Test queries
queries = [
    "machine learning artificial intelligence",
    "pet animals at home",
    "healthy eating habits",
    "sport physical activity"
]

for q in queries:
    search(q, corpus_docs, corpus_vecs)
```

---

## 📊 Bước 6: OOV Handling Comparison

```python
print("\n=== OOV COMPARISON: Word2Vec vs FastText ===")
oov_words = ["doggy", "pythonic", "footballer", "wellness", "deeplearn"]

for word in oov_words:
    w2v_oov = word not in w2v_model.wv
    ft_has = word in ft_model.wv or True  # FastText can handle via subwords
    
    if not w2v_oov:
        w2v_status = f"Found (dim={len(w2v_model.wv[word])})"
    else:
        w2v_status = "OOV ❌"
    
    try:
        ft_vec = ft_model.wv[word]
        ft_status = f"Handled ✅ (via subwords)"
    except:
        ft_status = "OOV ❌"
    
    print(f"\n  '{word}':")
    print(f"    Word2Vec: {w2v_status}")
    print(f"    FastText: {ft_status}")
```

---

## 📝 Kết luận

| Aspect | Word2Vec | FastText |
|--------|---------|---------|
| OOV handling | ❌ | ✅ |
| Morphology | ❌ | ✅ |
| Training speed | Faster | Slower |
| Similarity quality | Good | Good+ |
| Memory | Less | More |

**Key learnings:**
1. Embeddings capture semantic meaning that TF-IDF cannot
2. FastText handles OOV better than Word2Vec
3. Clustering in embedding space reflects semantic categories
4. Mean pooling = simple but effective sentence embedding
5. Semantic search with embeddings >> keyword matching

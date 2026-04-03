# 👉 Build Mini Search Engine (TF-IDF + Cosine + Edit Distance)
→ đúng kiểu dùng hết những gì m vừa học.

# 🚀 MINI PROJECT 1: TEXT SEARCH ENGINE
### 🎯 Mục tiêu

Xây hệ thống:

Nhập query
Trả về câu giống nhất trong dataset

### 👉 Áp dụng:

Tokenization
Cleaning
TF-IDF
Cosine Similarity
(bonus) Edit Distance
📂 1. Dataset mẫu
documents = [
    "Tôi thích học AI",
    "Tôi thích ăn cơm",
    "Hôm nay trời đẹp",
    "AI rất thú vị",
    "Tôi đang học machine learning"
]
🧱 2. Pipeline tổng thể
Text
→ Normalize
→ Tokenize
→ TF-IDF
→ Cosine Similarity
→ Ranking
⚙️ 3. Code FULL (chạy được luôn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------
# 1. Normalize function
# -------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\\w\\s]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# -------------------
# 2. Dataset
# -------------------
documents = [
    "Tôi thích học AI",
    "Tôi thích ăn cơm",
    "Hôm nay trời đẹp",
    "AI rất thú vị",
    "Tôi đang học machine learning"
]

# normalize dataset
documents = [normalize(doc) for doc in documents]

# -------------------
# 3. TF-IDF
# -------------------
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# -------------------
# 4. Search function
# -------------------
def search(query):
    query = normalize(query)
    query_vec = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vec, doc_vectors)[0]
    
    ranked = sorted(
        list(enumerate(similarities)),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked

# -------------------
# 5. Test
# -------------------
query = "tôi thích ai"

results = search(query)

for idx, score in results:
    print(f"{documents[idx]}  --> score: {score:.3f}")
🧠 4. Kết quả mong đợi

Query:

"tôi thích ai"

Output:

tôi thích học ai        → cao nhất
ai rất thú vị           → thứ 2
...
🔥 5. Upgrade (quan trọng)
🔹 5.1 Add N-gram
TfidfVectorizer(ngram_range=(1,2))
🔹 5.2 Add Edit Distance (fix typo)
from difflib import get_close_matches

get_close_matches("helo", ["hello", "help"])
🔹 5.3 Top-k results
return ranked[:3]
🔹 5.4 Build CLI
while True:
    q = input("Query: ")
    print(search(q))
💣 6. Level up (nếu m muốn xịn hơn)
Dùng dataset thật (news, product)
Build API bằng FastAPI
Add UI đơn giản (Streamlit)
🎯 7. M học được gì từ project này?

Sau project này m sẽ hiểu:

TF-IDF dùng thật như nào
Cosine similarity hoạt động ra sao
Pipeline NLP thực tế
Cách build system end-to-end

👉 Đây là project level intern rồi, không còn là lý thuyết nữa.

🚀 Next

Nếu m tiếp tục:

👉 Bài 11–20 sẽ có Mini Project #2:

Text Classification (Spam / Sentiment)

👉 Giờ m chọn:

T muốn t review code m khi m làm
hay t nâng project này lên production (API + UI)
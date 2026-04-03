# 💻 Mini Project 2 (Bài 11–20)

# Tên: “Pipeline NLP cơ bản + Trích xuất thông tin”

Mục tiêu:

Luyện tập tokenization, cleaning, stopwords, n-gram
Tạo vocabulary
Gán POS, NER
Chunking và Dependency Parsing
Coreference (nếu có thể)
Bước 1: Chuẩn hóa dữ liệu
Lấy 5–10 câu tiếng Việt hoặc tiếng Anh
Loại bỏ stopwords, làm lowercase, normalization
Bước 2: Tokenization + N-gram
Tách từ
Tạo unigram + bigram
In ra top 5 từ/bigram phổ biến
Bước 3: POS Tagging
Gán POS cho mỗi token
Highlight các danh từ (NOUN) và động từ (VERB)
Bước 4: NER
Trích xuất Person / Location / Organization
In ra các entity theo loại
Bước 5: Chunking
Nhóm các Noun Phrase (NP) và Verb Phrase (VP)
In ra từng cụm phrase
Bước 6: Dependency Parsing
Xây dependency tree cho 2–3 câu
Xác định ROOT, nsubj, dobj
Bước 7: Coreference Resolution (nếu dùng tiếng Anh SpaCy)
Nhận dạng các pronoun → mention clusters
Bước 8: Báo cáo kết quả
Tổng hợp từ tokenization → POS → NER → Chunk → Dependency
In ra từng bước + highlight insights
Gợi ý code Python (SpaCy + NLTK)

```python
import spacy
from nltk import word_tokenize
from nltk.chunk import RegexpParser

# Load model
nlp = spacy.load("vi_core_news_sm")  # hoặc "en_core_web_sm"

texts = [
    "Nguyễn Văn A đi học AI tại Hà Nội.",
    "Lan và Minh cùng đi siêu thị."
]

for text in texts:
    # Tokenization + POS + NER
    doc = nlp(text)
    print("\nTokens + POS + NER:")
    for token in doc:
        print(token.text, token.pos_, token.ent_type_)
    
    # Chunking example
    grammar = "NP: {<NOUN|PROPN>+}"
    cp = RegexpParser(grammar)
    tree = cp.parse([(t.text, t.pos_) for t in doc])
    print("\nChunks:")
    print(tree)
    
    # Dependency
    print("\nDependency:")
    for token in doc:
        print(token.text, token.dep_, token.head.text)

```
Mục tiêu học được:
Kết nối lý thuyết 10 bài trước → pipeline thực tế
Hiểu luồng dữ liệu NLP từ raw text → structured features
Chuẩn bị cho model training (Sentiment, Classification, QA…)

Nếu muốn mình có thể viết hẳn file MD cho project 2 này, giống các file bài trước, để m tải về và làm luôn.

M có muốn mình làm luôn không?
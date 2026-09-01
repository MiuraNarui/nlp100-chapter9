import torch
import torch.nn.functional as F
from itertools import combinations
from transformers import AutoTokenizer, AutoModel

# ModernBERT-baseを読み込む
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 類似度を求める4つの文
sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

# 4文をまとめてトークン化
inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# ModernBERTで各トークンの埋め込みベクトルを取得
with torch.no_grad():
    outputs = model(**inputs)

# 最終層の[CLS]トークンの埋め込みベクトルを取得
# [CLS]は各入力文の先頭（位置0）に置かれる
cls_vectors = outputs.last_hidden_state[:, 0, :]

# 全ての組み合わせ（4C2 = 6通り）についてコサイン類似度を計算
for i, j in combinations(range(len(sentences)), 2):
    similarity = F.cosine_similarity(
        cls_vectors[i].unsqueeze(0),
        cls_vectors[j].unsqueeze(0)
    ).item()

    print(f"文{i + 1}: {sentences[i]}")
    print(f"文{j + 1}: {sentences[j]}")
    print(f"コサイン類似度: {similarity:.6f}")
    print()
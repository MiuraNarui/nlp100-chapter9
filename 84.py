import itertools
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ModernBERT-baseを読み込む
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()

# 類似度を求める4つの文
sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]


def mean_vec(sentence):
    # 1文をトークン化
    inputs = tokenizer(sentence, return_tensors="pt")

    # 最終層の各トークンの埋め込みベクトルを取得
    with torch.no_grad():
        outputs = model(**inputs)

    # 全トークンの埋め込みベクトルの平均を文ベクトルとする
    return outputs.last_hidden_state[0].mean(dim=0)


# 各文の平均文ベクトルを計算
vectors = [mean_vec(sentence) for sentence in sentences]

# 全ての組み合わせ（4C2 = 6通り）でコサイン類似度を計算
for i, j in itertools.combinations(range(len(sentences)), 2):
    similarity = F.cosine_similarity(
        vectors[i],
        vectors[j],
        dim=0
    ).item()

    print(sentences[i], sentences[j], similarity)
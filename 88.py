import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 87で保存したファインチューニング済みモデルを読み込む
model_path = "fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 利用可能なデバイスを選択
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

# 極性を予測する文
sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

# 文をまとめてトークン化
inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 極性を予測
with torch.no_grad():
    outputs = model(**inputs)

predicted_labels = torch.argmax(outputs.logits, dim=-1)

# 結果を表示
for sentence, label in zip(sentences, predicted_labels.cpu().tolist()):
    polarity = "ポジティブ" if label == 1 else "ネガティブ"
    print(sentence)
    print(f"予測ラベル: {label} ({polarity})")
    print()
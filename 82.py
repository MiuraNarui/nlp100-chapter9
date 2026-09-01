import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# BERTモデルを読み込む
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).eval()

# [MASK] を含む文
text = "The movie was full of [MASK]."

# 文をモデル入力に変換
inputs = tokenizer(text, return_tensors="pt")

# [MASK] の位置を取得
mask_index = torch.where(
    inputs["input_ids"][0] == tokenizer.mask_token_id
)[0].item()

# モデルで各トークンのスコアを計算
with torch.no_grad():
    outputs = model(**inputs)

# [MASK] の位置における全語彙のスコア
mask_logits = outputs.logits[0, mask_index]

# スコアを確率に変換
probabilities = torch.softmax(mask_logits, dim=-1)

# 確率が高い上位10トークンを取得
top_probs, top_ids = torch.topk(probabilities, k=10)

print("入力文:", text)
print("上位10トークンと確率")

for rank, (token_id, prob) in enumerate(zip(top_ids, top_probs), start=1):
    token = tokenizer.convert_ids_to_tokens(token_id.item())
    print(f"{rank:2d}. {token:<20} {prob.item():.6f}")
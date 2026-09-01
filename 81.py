import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ModernBERT-base を読み込む
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# [MASK] を含む文
text = "The movie was full of [MASK]."

# 文をモデル入力に変換
inputs = tokenizer(text, return_tensors="pt")

# [MASK] の位置を取得
mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

# [MASK] に入る各トークンのスコアを計算
with torch.no_grad():
    outputs = model(**inputs)

mask_logits = outputs.logits[0, mask_index, :]

# 最もスコアの高いトークンを取得
predicted_token_id = torch.argmax(mask_logits, dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print("入力文:", text)
print("予測されたトークン:", predicted_token)
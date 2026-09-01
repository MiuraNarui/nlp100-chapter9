from transformers import AutoTokenizer

# BERTモデルのトークナイザを読み込む
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# トークン化する文
text = "The movie was full of incomprehensibilities."

# 文をトークンに分解
tokens = tokenizer.tokenize(text)

# トークン列を表示
print(tokens)
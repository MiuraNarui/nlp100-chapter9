from transformers import AutoTokenizer

# ModernBERT-base のトークナイザを読み込む
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# トークン化する文
text = "The movie was full of incomprehensibilities."

# 文をトークンに分解
tokens = tokenizer.tokenize(text)

# トークン列を表示
print(tokens)
import csv
import torch
from transformers import AutoTokenizer

# BERTモデルのトークナイザを読み込む
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_data(path):
    data = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = int(row["label"])
            tokens = tokenizer.tokenize(text)

            data.append({
                "text": text,
                "label": label,
                "tokens": tokens
            })

    return data


# 訓練データを読み込む
train_data = load_data("SST-2/train.tsv")

# 冒頭の4事例を取り出す
mini_batch = train_data[:4]

texts = [example["text"] for example in mini_batch]
labels = torch.tensor([example["label"] for example in mini_batch])

# 4文をまとめてトークン化し、最長の文に合わせてパディングする
batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# ラベルもミニバッチに追加
batch["labels"] = labels

print("input_ids:")
print(batch["input_ids"])

print("\nattention_mask:")
print(batch["attention_mask"])

print("\nlabels:")
print(batch["labels"])

print("\nshape:")
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("labels:", batch["labels"].shape)
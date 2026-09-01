import os
import csv
import urllib.request
import zipfile
from transformers import AutoTokenizer

# BERTモデルのトークナイザを読み込む
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# SST-2をダウンロード
url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
zip_path = "SST-2.zip"

if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)

# ZIPファイルを展開
if not os.path.exists("SST-2"):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")


def load_data(path):
    data = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = int(row["label"])

            # テキストをBERTのトークン列に変換
            tokens = tokenizer.tokenize(text)

            data.append({
                "text": text,
                "label": label,
                "tokens": tokens
            })

    return data


# 訓練セットと開発セットを読み込む
train_data = load_data("SST-2/train.tsv")
dev_data = load_data("SST-2/dev.tsv")

# データ数を確認
print("訓練データ数:", len(train_data))
print("開発データ数:", len(dev_data))

# 先頭の事例を確認
print("\n訓練データの先頭事例")
print(train_data[0])

print("\n開発データの先頭事例")
print(dev_data[0])
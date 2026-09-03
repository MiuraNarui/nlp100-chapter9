import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# BERTモデルとトークナイザを読み込む
name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)

# 85で準備したSST-2を読み込む
train_df = pd.read_csv("SST-2/train.tsv", sep="\t")
dev_df = pd.read_csv("SST-2/dev.tsv", sep="\t")


# DataFrameをHugging Face Datasetに変換し、トークン化する
def to_ds(df):
    ds = Dataset.from_pandas(
        df.rename(columns={"label": "labels"}),
        preserve_index=False,
    )

    return ds.map(
        lambda batch: tokenizer(
            batch["sentence"],
            truncation=True,
        ),
        batched=True,
    )


train_ds = to_ds(train_df)
dev_ds = to_ds(dev_df)

# ミニバッチごとに動的パディングを行う
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 極性分類用のBERTモデルを読み込む
clf = AutoModelForSequenceClassification.from_pretrained(
    name,
    num_labels=2,
)


# 正解率を計算する
def metrics(p):
    predictions = p.predictions.argmax(axis=-1)
    accuracy = (predictions == p.label_ids).mean()
    return {"accuracy": accuracy}


# 学習条件
args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    eval_strategy="epoch",
    report_to="none",
)

# Trainerを作成
trainer = Trainer(
    model=clf,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=collator,
    compute_metrics=metrics,
)

# ファインチューニング
trainer.train()

# 検証データで評価
result = trainer.evaluate()
print(result)
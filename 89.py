import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# BERTモデルとトークナイザを読み込む
name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)

# SST-2を読み込む
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


# 最大値プーリングを用いた分類モデル
class BertMaxPoolingClassifier(nn.Module):
    def __init__(self, model_name, hidden_size=50, num_labels=2, dropout=0.1):
        super().__init__()

        # 事前学習済みBERT
        self.bert = AutoModel.from_pretrained(model_name)

        # BERTの出力次元
        bert_dim = self.bert.config.hidden_size

        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        # BERTの最終層の各トークン埋め込みを取得
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        token_vectors = outputs.last_hidden_state

        # [PAD]を最大値プーリングから除外
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).bool()
            token_vectors = token_vectors.masked_fill(~mask, -1e9)

        # 各次元についてトークン方向の最大値を取る
        pooled = token_vectors.max(dim=1).values

        # 隠れ層50次元 → ReLU → Dropout → 2クラス分類
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


model = BertMaxPoolingClassifier(
    model_name=name,
    hidden_size=50,
    num_labels=2,
    dropout=0.1,
)


# 正解率を計算する
def metrics(p):
    predictions = p.predictions.argmax(axis=-1)
    accuracy = (predictions == p.label_ids).mean()
    return {"accuracy": accuracy}


# 学習条件
args = TrainingArguments(
    output_dir="out_89",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    eval_strategy="epoch",
    report_to="none",
)

# Trainerを作成
trainer = Trainer(
    model=model,
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Quick Start Example - RedNote-Vibe Dataset

A simplified BERT binary classification example for quick start and testing.
"""

import json
import random
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from datasets import Dataset

# 下游只读取 note_content / desc；加载时按白名单投影，避免把整条记录
# （标题、话题、互动数据等大量未使用字段）全部驻留内存。
USED_KEYS = ('note_content', 'desc')


def _iter_jsonl(file_path):
    """流式产出投影后的记录（跳过空行与无法解析的行）。"""
    if not os.path.exists(file_path):
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line or line.isspace():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            yield {k: item[k] for k in USED_KEYS if k in item}


def _count_records(file_path):
    """按 `_iter_jsonl` 相同的行规则计数（不解析 JSON）。"""
    n = 0
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line and not line.isspace():
                n += 1
    return n


def _load_picked(file_path, wanted):
    """只解析行索引在 `wanted` 中的记录，返回 {行索引: 记录}。"""
    wanted = set(wanted)
    picked = {}
    if not wanted or not os.path.exists(file_path):
        return picked
    with open(file_path, 'r', encoding='utf-8') as f:
        i = -1
        for line in f:
            if not line or line.isspace():
                continue
            i += 1
            if i not in wanted:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            picked[i] = {k: item[k] for k in USED_KEYS if k in item}
            if len(picked) == len(wanted):
                break
    return picked


def load_data(max_samples=None):
    """Load data，可只取可复现的随机子集。

    传入 `max_samples` 时走两遍稀疏抽样：第一遍只数行数（O(1) 内存），
    第二遍只解析被抽中的约 2×max_samples 行，峰值内存与样本量成正比。
    抽样随机数路径与原来的 `random.Random(42)` 依次
    `sample(人类全量列表, k)`、`sample(AI 全量列表, k)` 完全一致
    （`sample(range(n), k)` 与对等长列表抽样逐位相同），因此抽到的样本
    与原实现完全相同。
    """
    print("Loading data...")

    # Get data paths (data may live in ./datasets/ or at the repo root)
    human_file = os.path.join('datasets', 'training_set_human.jsonl')
    ai_file = os.path.join('datasets', 'training_set_aigc.jsonl')
    if not os.path.exists(human_file):
        human_file = 'training_set_human.jsonl'
    if not os.path.exists(ai_file):
        ai_file = 'training_set_aigc.jsonl'

    if max_samples is None:
        human_data = list(_iter_jsonl(human_file))
        ai_data = list(_iter_jsonl(ai_file))
        n_human, n_ai = len(human_data), len(ai_data)
    else:
        n_human = _count_records(human_file)
        n_ai = _count_records(ai_file)
        rng = random.Random(42)
        human_idx = rng.sample(range(n_human), min(max_samples, n_human))
        ai_idx = rng.sample(range(n_ai), min(max_samples, n_ai))
        h_picked = _load_picked(human_file, human_idx)
        a_picked = _load_picked(ai_file, ai_idx)
        human_data = [h_picked[i] for i in human_idx if i in h_picked]
        ai_data = [a_picked[i] for i in ai_idx if i in a_picked]

    print(f"Human data: {n_human} samples")
    print(f"AI data: {n_ai} samples")

    return human_data, ai_data

def prepare_dataset(human_data, ai_data):
    """Prepare dataset（过滤空/过短正文；抽样已在 `load_data` 中完成）"""
    print("Preparing dataset...")
    
    texts = []
    labels = []

    for item in human_data:
        content = (item.get('note_content') or item.get('desc', '')).strip()
        if content and len(content) > 10:
            texts.append(content[:300])  # Truncate to 300 characters
            labels.append(0)  # Human
    
    for item in ai_data:
        content = (item.get('note_content') or item.get('desc', '')).strip()
        if content and len(content) > 10:
            texts.append(content[:300])
            labels.append(1)  # AI
    
    print(f"Total samples: {len(texts)}")
    print(f"   - Human: {labels.count(0)}")
    print(f"   - AI: {labels.count(1)}")
    
    return texts, labels

def tokenize_function(examples, tokenizer):
    """Data tokenization"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def main():
    """Main function"""
    print("BERT Quick Start Example")
    print("="*40)
    
    # 1. Load data（只解析抽样所需的少量行）
    human_data, ai_data = load_data(max_samples=500)
    
    if not human_data or not ai_data:
        print("Data loading failed!")
        return
    
    # 2. Prepare data
    texts, labels = prepare_dataset(human_data, ai_data)  # Small sample for quick testing
    
    # 3. Split dataset
    print("\nSplitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Initialize tokenizer and model
    print("\nInitializing BERT model...")
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 5. Create Dataset objects
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'labels': y_train
    })
    val_dataset = Dataset.from_dict({
        'text': X_val,
        'labels': y_val
    })
    test_dataset = Dataset.from_dict({
        'text': X_test,
        'labels': y_test
    })
    
    # 6. Tokenization
    print("\nTokenizing texts...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # 7. Training configuration
    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # 8. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 9. Start training
    print("\nStarting training...")
    trainer.train()
    
    # 10. Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
    
    # 11. Prediction examples
    print("\nPrediction examples...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Classification report
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # 12. Save model
    model_save_path = './bert_model_saved'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # 13. Show some prediction examples
    print("\nPrediction samples:")
    for i in range(min(5, len(X_test))):
        true_label = "Human" if y_test[i] == 0 else "AI"
        pred_label = "Human" if y_pred[i] == 0 else "AI"
        correct = "Correct" if y_test[i] == y_pred[i] else "Wrong"
        
        print(f"\nSample {i+1}: {correct}")
        print(f"Text: {X_test[i][:100]}...")
        print(f"True: {true_label}, Predicted: {pred_label}")
    
    print(f"\nTraining completed! Final accuracy: {test_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()

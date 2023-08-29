""" 对在中文情感数据库上预训练的模型, 进一步在ChnSentiCrop新闻情感数据集上微调, 使之更适用于新闻场景
"""

import os
import time
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def solid_seed(seed):
    # 设置随机种子
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子

    # 在使用 GPU 时还可以设置 GPU 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, tokenizer, data_base, split, max_len=64):
        texts, labels = self.load_data(data_base, split)
        inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.input_ids = inputs['input_ids']
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        input_id = self.input_ids[index]
        label = self.labels[index]
        return input_id, label
    
    def load_data(self, data_base, split):
        data = pd.read_csv(os.path.join(data_base, f'{split}.csv'))
        texts, labels = data['text'].tolist(), data['label'].tolist()
        return texts, labels


if __name__ == '__main__':
    # 参数设置
    bs = 256
    num_epochs = 30
    dropout = 0.0
    lr = 2e-5
    gpu = 0
    SEED = 42
    max_seq_len = 128


    # 预训练模型加载
    print("模型加载中...")
    model_name = "voidful/albert_chinese_small_sentiment"
    model_config = AutoConfig.from_pretrained("voidful/albert_chinese_small_sentiment")
    model_config.hidden_dropout_prob = dropout
    model_config.attention_probs_dropout_prob = dropout
    model = AutoModelForSequenceClassification.from_pretrained("voidful/albert_chinese_small_sentiment", config=model_config)
    tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_small_sentiment")

    # 数据集构建
    print("数据集构建中...")
    data_base = "/data1/brian/Quant/news_sentiment_analysis/3-sentiment_finetuning/ChnSentiCrop"
    solid_seed(SEED)
    train_dataset = TextDataset(tokenizer, data_base, 'train', max_len=max_seq_len)
    dev_dataset = TextDataset(tokenizer, data_base, 'dev', max_len=max_seq_len)
    test_dataset = TextDataset(tokenizer, data_base, 'test', max_len=max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 模型训练
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    save_base = "/data1/brian/Quant/news_sentiment_analysis/3-sentiment_finetuning/outputs/" + current_time
    os.makedirs(save_base, exist_ok=True)
    logs = open(os.path.join(save_base, 'logs.txt'), 'a')
    best_criterion = 0
    print("开始微调模型 {} 轮次".format(num_epochs))
    for epoch in range(num_epochs):
        # 训练
        model.train()
        all_train_labels, all_pred_train_labels = [], []
        all_losses = []
        pbar = tqdm(total=len(train_loader), leave=True)
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            logits = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(logits, dim=1)
            acc = torch.sum(labels==pred_labels) / len(labels)

            all_losses.append(loss.item())
            all_train_labels += labels.cpu().numpy().tolist()
            all_pred_train_labels += pred_labels.cpu().numpy().tolist()

            pbar.set_description(f'Epoch {epoch}')
            pbar.set_postfix({"loss": loss.item(), 'acc': acc.item()})  # 更新附加信息
            pbar.update(1)
        pbar.close()
        train_acc = accuracy_score(all_train_labels, all_pred_train_labels)
        train_loss = np.mean(all_losses)

        # 验证
        model.eval()
        all_dev_labels, all_pred_dev_labels = [], []
        all_losses = []
        for inputs, labels in dev_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            all_losses.append(loss.item())
            logits = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(logits, dim=1)
            all_dev_labels += labels.cpu().numpy().tolist()
            all_pred_dev_labels += pred_labels.cpu().numpy().tolist()
        dev_acc = accuracy_score(all_dev_labels, all_pred_dev_labels)
        dev_loss = np.mean(all_losses)
        if dev_acc > best_criterion:
            best_criterion = dev_acc
            best_model = model
            best_epoch = epoch
        
        epoch_log = "Epoch-{} Train loss: {:.4f}, acc: {:.4f} | Dev loss: {:.4f}, acc: {:.4f}".format(
                        epoch, train_loss, train_acc, dev_loss, dev_acc)
        print(epoch_log)
        logs.write(epoch_log + '\n')

    # 测试
    model.eval()
    all_test_labels, all_pred_test_labels = [], []
    all_losses = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        all_losses.append(loss.item())
        logits = torch.softmax(logits, dim=1)
        pred_labels = torch.argmax(logits, dim=1)
        all_test_labels += labels.cpu().numpy().tolist()
        all_pred_test_labels += pred_labels.cpu().numpy().tolist()
    test_acc = accuracy_score(all_test_labels, all_pred_test_labels)
    test_loss = np.mean(all_losses)

    test_log = "Best epoch {} testing loss: {:.4f}, acc: {:.4f}".format(best_epoch, test_loss, test_acc)
    logs.write(test_log)
    logs.close()

    # torch.save(best_model.state_dict(), os.path.join(save_base, f'best_epoch-{best_epoch}_model.pth'))
    best_model.save_pretrained(os.path.join(save_base, f'albert_finetuned_ep{best_epoch}'))
    tokenizer.save_pretrained(os.path.join(save_base, f'albert_finetuned_ep{best_epoch}'))
        
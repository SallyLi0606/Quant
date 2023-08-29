import os
import time
import math
import scipy
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
import datetime
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
import talib
import argparse
import warnings
warnings.filterwarnings("ignore")


def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

def get_mae(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None
        
def extract_factor(data):
    data.rename(columns={'time':'date', 'chengjiaogushu':'volume', 'huanshoulv':'turnover_rate'}, inplace=True)
    data = data[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate']]
    # MA(5), 移动平均线
    data['ma5'] = talib.MA(data['close'].values, timeperiod=5)
    # MA(30), 移动平均线
    data['ma30'] = talib.MA(data['close'].values, timeperiod=30)
    # MA(60), 移动平均线
    data['ma60'] = talib.MA(data['close'].values, timeperiod=60)
    # EMA(5), 三重指数移动平均线
    data['ema5'] = talib.EMA(data['close'].values, timeperiod=5)
    # EMA(30), 三重指数移动平均线
    data['ema30'] = talib.EMA(data['close'].values, timeperiod=30)
    # EMA(60), 三重指数移动平均线
    data['ema60'] = talib.EMA(data['close'].values, timeperiod=60)
    # MACD(6,15,6), 平滑异同移动平均线
    dif_, dea_, data['macd_6_15_6'] = talib.MACD(data['close'].values, fastperiod=6, slowperiod=15, signalperiod=6)
    # MACD(12,26,9), 平滑异同移动平均线
    dif_, dea_, data['macd_12_26_9'] = talib.MACD(data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    # MACD(30,60,30), 平滑异同移动平均线
    dif_, dea_, data['macd_30_60_30'] = talib.MACD(data['close'].values, fastperiod=30, slowperiod=60, signalperiod=30)
    # RSI(14), 随机相对强弱指数
    data['rsi'] = talib.RSI(data['close'].values, timeperiod=14)
    # WILLR(14), 威廉指标
    data['wr'] = talib.WILLR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
    # MOM(14), 上升动向值
    data['mom'] = talib.MOM(data['close'].values, timeperiod=14)
    # CMO(14), 钱德动量摆动指标
    data['cmo'] = talib.CMO(data['close'].values, timeperiod=14)
    # ULTOSC(7,14,28), 终极波动指标
    data['ultosc'] = talib.ULTOSC(data['high'].values, data['low'].values, data['close'].values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # OBV, 能量潮
    data['obv'] = talib.OBV(data['close'].values, data['volume'])
    # ADOSC(3,10), 震荡指标
    data['adosc'] = talib.ADOSC(data['high'].values, data['low'].values, data['close'].values, data['volume'], fastperiod=3, slowperiod=10)
    data.dropna(inplace=True)
    return data


def split_windows(df, size, valid_size=0.1):
    scaler = MinMaxScaler()
    features_train_valid, targets_train_valid, dates_train_valid = [], [], []
    features_test, targets_test, dates_test = [], [], []
    for i in range(len(df) - size - 2):
        date = df.iloc[i+size]['date']
        if date < '2022-06':
            stock_data = df.iloc[i:i+size][['ma5', 'ma30', 'ma60', 'ema5', 'ema30', 'ema60', 'macd_6_15_6', 'macd_12_26_9', 'macd_30_60_30', 'rsi', 'wr', 'mom', 'cmo', 'ultosc', 'obv', 'adosc']].values
            stock_data_normalized = scaler.fit_transform(stock_data)
            features_train_valid.append(stock_data_normalized)
            targets_train_valid.append(1 if df.iloc[i+size+2]['close'] > df.iloc[i+size-1]['close'] else 0)
            dates_train_valid.append(date)
        else:
            stock_data = df.iloc[i:i+size][['ma5', 'ma30', 'ma60', 'ema5', 'ema30', 'ema60', 'macd_6_15_6', 'macd_12_26_9', 'macd_30_60_30', 'rsi', 'wr', 'mom', 'cmo', 'ultosc', 'obv', 'adosc']].values
            stock_data_normalized = scaler.fit_transform(stock_data)
            features_test.append(stock_data_normalized)
            targets_test.append(1 if df.iloc[i+size+2]['close'] > df.iloc[i+size-1]['close'] else 0)
            dates_test.append(date)
    #
    features_train = features_train_valid[:len(features_train_valid)-int(len(features_train_valid)*valid_size)]
    targets_train = targets_train_valid[:len(targets_train_valid)-int(len(targets_train_valid)*valid_size)]
    dates_train = dates_train_valid[:len(dates_train_valid)-int(len(dates_train_valid)*valid_size)]
    #
    features_valid = features_train_valid[len(features_train_valid)-int(len(features_train_valid)*valid_size):]
    targets_valid = targets_train_valid[len(targets_train_valid)-int(len(targets_train_valid)*valid_size):]
    dates_valid = dates_train_valid[len(dates_train_valid)-int(len(dates_train_valid)*valid_size):]
    
    return (features_train, targets_train, dates_train), (features_valid, targets_valid, dates_valid), (features_test, targets_test, dates_test)

def train(model, optimizer, criterion, scheduler, dataloader, device):
    model.train()
    losses = 0
    all_dates = []
    all_real_labels, all_predict_labels = [], []
    current_count, spent_time_accumulation = 0, 0
    all_count = len(dataloader)
    for step, data in enumerate(dataloader):
        start_time_batch = time.time()
        optimizer.zero_grad()
        features, labels, dates = data  # features: (batch size, 8), labels: (batch size)
        features, labels = features.to(device), labels.to(device)
        # 前向传播
        predict_labels = model(features)  # (batch size)
        # 计算损失
        loss = criterion(predict_labels, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        losses += loss.item()
        all_dates += list(dates)
        all_predict_labels += F.softmax(predict_labels, dim=1)[:, 1].cpu().tolist()
        all_real_labels += labels.cpu().tolist()
        # 计算运行时间
        end_time_batch = time.time()
        seconds = end_time_batch-start_time_batch
        spent_time_accumulation += seconds
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        spend_time_batch = "%02d:%02d:%02d" % (h, m, s)
        m, s = divmod(spent_time_accumulation, 60)
        h, m = divmod(m, 60)
        have_spent_time = "%02d:%02d:%02d" % (h, m, s)
        # 计算batch pearsonr
        auc_batch = roc_auc_score(all_real_labels, all_predict_labels)
        current_count += 1
        # if current_count == all_count:
        #     print("Finish batch: %d/%d---train auc: %.4f---batch time: %s, have spent time: %s" % (current_count, all_count, auc_batch, spend_time_batch, have_spent_time))
        # else:
        #     print("Finish batch: %d/%d---train auc: %.4f---batch time: %s, have spent time: %s" % (current_count, all_count, auc_batch, spend_time_batch, have_spent_time), end='\r')
    
    # 调整学习率
    scheduler.step()
    
    # 返回损失值
    return losses / (all_count + 1), all_dates, all_real_labels, all_predict_labels

def test(model, criterion, dataloader, device):
    model.eval()
    losses = 0
    current_count, spent_time_accumulation = 0, 0
    all_count = len(dataloader)
    all_dates = []
    all_real_labels, all_predict_labels = [], []
    for step, data in enumerate(dataloader):
        start_time_batch = time.time()
        features, labels, dates = data
        features, labels = features.to(device), labels.to(device)
        # 前向传播
        with torch.no_grad():
            predict_labels = model(features)
        # 计算损失
        loss = criterion(predict_labels, labels)
        losses += loss.item()
        all_dates += list(dates)
        all_predict_labels += F.softmax(predict_labels, dim=1)[:, 1].cpu().tolist()
        all_real_labels += labels.cpu().tolist()
        # 计算运行时间
        end_time_batch = time.time()
        seconds = end_time_batch-start_time_batch
        spent_time_accumulation += seconds
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        spend_time_batch = "%02d:%02d:%02d" % (h, m, s)
        m, s = divmod(spent_time_accumulation, 60)
        h, m = divmod(m, 60)
        have_spent_time = "%02d:%02d:%02d" % (h, m, s)
        # 计算batch pearsonr
        auc_batch = roc_auc_score(all_real_labels, all_predict_labels)
        current_count += 1
        # if current_count == all_count:
        #     print("Finish batch: %d/%d---test auc: %.4f---batch time: %s, have spent time: %s" % (current_count, all_count, auc_batch, spend_time_batch, have_spent_time))
        # else:
        #     print("Finish batch: %d/%d---test auc: %.4f---batch time: %s, have spent time: %s" % (current_count, all_count, auc_batch, spend_time_batch, have_spent_time), end='\r')
    
    return losses / (all_count + 1), all_dates, all_real_labels, all_predict_labels

def train_test(train_data, valid_data, test_data, batch_size):
    # 定义超参数
    torch.manual_seed(1)
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    lr = 0.001  # 学习率
    epochs = 150  # 迭代训练轮次

    # 初始化模型
    model = MyModel(input_dim=16, dim=128, mlp_dim=256, depth=6)
    model = model.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 定义学习率调度器，StepLR，来根据设定动态地改变学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Dataset, Dataloader
    trainset = DeepLDataset(train_data)
    validset = DeepLDataset(valid_data)
    testset = DeepLDataset(test_data)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 定义两个列表，用来保存所有epoch的训练损失和准确率
    train_losses, train_aucs = [], []
    test_losses, test_aucs = [], []
    # 训练模型
    # 遍历每一轮
    print('Strart training...')
    min_valid_loss, max_valid_auc = float('inf'), -float('inf')
    min_test_loss, max_test_auc = None, None
    best_epoch = None
    best_all_dates_test = None
    best_all_real_labels_test, best_all_predict_labels_test = None, None
    for epoch in range(epochs):
        loss_epoch_train, all_dates_train, all_real_labels_train, all_predict_labels_train = train(model, optimizer, criterion, scheduler, train_loader, device)
        loss_epoch_valid, all_dates_valid, all_real_labels_valid, all_predict_labels_valid = test(model, criterion, valid_loader, device)
        loss_epoch_test, all_dates_test, all_real_labels_test, all_predict_labels_test = test(model, criterion, test_loader, device)
        auc_train = roc_auc_score(all_real_labels_train, all_predict_labels_train)
        auc_valid = roc_auc_score(all_real_labels_valid, all_predict_labels_valid)
        auc_test = roc_auc_score(all_real_labels_test, all_predict_labels_test)
        # 打印每一轮的平均损失
        # print(f"Epoch: %d, train loss:%.4f, train pearsonr:%.4f, test pearsonr:%.4f" % (epoch+1, loss_epoch_train, pear_train, pear_test))
        # 将当前的训练损失和准确率添加到列表中
        train_losses.append(loss_epoch_train)
        train_aucs.append(auc_train)
        test_losses.append(loss_epoch_test)
        test_aucs.append(auc_test)
        if min_valid_loss > loss_epoch_valid:
        # if max_valid_auc < auc_valid:
            min_valid_loss = loss_epoch_valid
            max_valid_auc = auc_valid
            best_all_dates_test = all_dates_test
            best_all_real_labels_test = all_real_labels_test
            best_all_predict_labels_test = all_predict_labels_test
            min_test_loss = loss_epoch_test
            max_test_auc = auc_test
            
    df_tst_res = pd.DataFrame()
    df_tst_res['time'] = best_all_dates_test
    df_tst_res['real'] = best_all_real_labels_test
    df_tst_res['predict'] = best_all_predict_labels_test
    return df_tst_res, min_valid_loss, max_valid_auc, min_test_loss, max_test_auc



class DeepLDataset(Dataset):
    def __init__(self, data):
        self.features, self.targets, self.dates = data

    def __getitem__(self, index):
        feature = torch.from_numpy(self.features[index]).type(torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        date = self.dates[index]
        return feature, target, date

    def __len__(self):
        return len(self.features)
    
    
# Transformer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 dropout: float = 0.):
        super(MultiHeadSelfAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, depth: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class MyModel(nn.Module):
    def __init__(self, input_dim: int, dim: int, mlp_dim: int, depth: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        self.pro_linear = nn.Linear(input_dim, dim)
        self.transformer = Transformer(dim=dim, mlp_dim=mlp_dim, depth=depth, heads=heads, dim_head=dim_head, dropout=dropout)
        self.out_linear = nn.Linear(dim, 2)
    
    def forward(self, x):
        x = self.pro_linear(x)
        x = self.transformer(x)
        x = x.mean(1)
        x = self.out_linear(x)
        return x
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='quant')
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('--per_stock', type=int, default=-1)
    args = parser.parse_args()
    print('Use GPU:%s' % os.environ['CUDA_VISIBLE_DEVICES'])
    
    item = f'stock_lstm_v{args.n + 1}'
    base_df = pd.read_csv('../dataset/all_codes_before20200630_filtered.csv', converters = {u'code':str})
    src_dir = '../dataset/stock'
    tgt_dir = f'output/{item}'
    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)
    f = open(f'output/{item}_record.txt', 'w')
    f.write('code\tmin_valid_loss\tmax_valid_auc\tmin_test_loss\tmax_test_auc\n')
    f.close()
    for code in base_df['code'][args.n * args.per_stock: (args.n + 1) * args.per_stock]:
        batch_size = 16
        if not os.path.isfile(os.path.join(src_dir, '%s.csv' % code)):
            continue
        if os.path.isfile(os.path.join(tgt_dir, '%s.csv' % code)):
            continue
        df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
        df = extract_factor(df)
        train_data, valid_data, test_data = split_windows(df, size=20)
        if len(train_data[0]) == 0 or len(train_data[0]) % batch_size == 1:
            continue
        if len(test_data[0]) == 0:
            continue
        try:
            df_tst_res, min_valid_loss, max_valid_auc, min_test_loss, max_test_auc = train_test(train_data, valid_data, test_data, batch_size)
            df_tst_res.to_csv(os.path.join(tgt_dir, '%s.csv' % code), index=0)
            f = open(f'output/{item}_record.txt', 'a')
            f.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\n' % (code, min_valid_loss, max_valid_auc, min_test_loss, max_test_auc))
            f.close()
        except:
            pass
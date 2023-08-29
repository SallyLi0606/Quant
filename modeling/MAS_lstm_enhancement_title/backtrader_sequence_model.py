import os
import time
import math
from datetime import datetime
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

def split_windows(df, senti_features_dic, size, valid_size=0.1):
    scaler = MinMaxScaler()
    features_train_valid, targets_train_valid, dates_train_valid = [], [], []
    features_test, targets_test, dates_test = [], [], []
    for i in range(len(df) - size - 2):
        pre_date = df.iloc[i:i+size]['date']
        senti_features = [senti_features_dic.get(date, torch.zeros(1, 384)) for date in pre_date]
        senti_features = torch.cat(senti_features, 0)
        now_date = df.iloc[i+size]['date']
        if now_date < '2022-06':
            stock_data = df.iloc[i:i+size][['ma5', 'ma30', 'ma60', 'ema5', 'ema30', 'ema60', 'macd_6_15_6', 'macd_12_26_9', 'macd_30_60_30', 'rsi', 'wr', 'mom', 'cmo', 'ultosc', 'obv', 'adosc']].values
            stock_data_normalized = scaler.fit_transform(stock_data)
            features_train_valid.append((stock_data_normalized, senti_features))
            targets_train_valid.append(1 if df.iloc[i+size+2]['close'] > df.iloc[i+size-1]['close'] else 0)
            dates_train_valid.append(now_date)
        else:
            stock_data = df.iloc[i:i+size][['ma5', 'ma30', 'ma60', 'ema5', 'ema30', 'ema60', 'macd_6_15_6', 'macd_12_26_9', 'macd_30_60_30', 'rsi', 'wr', 'mom', 'cmo', 'ultosc', 'obv', 'adosc']].values
            stock_data_normalized = scaler.fit_transform(stock_data)
            features_test.append((stock_data_normalized, senti_features))
            targets_test.append(1 if df.iloc[i+size+2]['close'] > df.iloc[i+size-1]['close'] else 0)
            dates_test.append(now_date)
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
        stock_features, senti_features, labels, dates = data  # features: (batch size, 8), labels: (batch size)
        stock_features, senti_features, labels = stock_features.to(device), senti_features.to(device), labels.to(device)
        # 前向传播
        predict_labels = model(stock_features, senti_features)  # (batch size)
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
        stock_features, senti_features, labels, dates = data
        stock_features, senti_features, labels = stock_features.to(device), senti_features.to(device), labels.to(device)
        # 前向传播
        with torch.no_grad():
            predict_labels = model(stock_features, senti_features)
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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    lr = 0.001  # 学习率
    epochs = 150  # 迭代训练轮次

    # 初始化模型
    model = SimpleLSTM(input_size=16, hidden_size=32, num_layers=2, num_classes=2)
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
    valid_loader = DataLoader(validset, batch_size=len(validset), shuffle=False)
    test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)

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
        stock_feature, senti_feature = self.features[index]
        stock_feature = torch.from_numpy(stock_feature).type(torch.float32)
        senti_feature = senti_feature.type(torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        date = self.dates[index]
        return stock_feature, senti_feature, target, date

    def __len__(self):
        return len(self.features)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.senti_linear = nn.Linear(384, input_size)
        self.lstm = nn.LSTM(input_size*2, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x_stock, x_senti):
        device = x_stock.device
        h0 = torch.zeros(self.num_layers, x_stock.size(0), self.hidden_size).type(x_stock.dtype).to(device)
        c0 = torch.zeros(self.num_layers, x_stock.size(0), self.hidden_size).type(x_stock.dtype).to(device)
        x_senti = self.senti_linear(x_senti)
        x = torch.cat([x_stock, x_senti], -1)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.fc2(out)
        return out
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='quant')
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('--per_stock', type=int, default=-1)
    args = parser.parse_args()
    print('Use GPU:%s' % os.environ['CUDA_VISIBLE_DEVICES'])
    item = f'stock_lstm_v{args.n + 1}'
    print(item)
    
    base_df = pd.read_csv('../dataset/all_codes_before20200630_filtered.csv', converters = {u'code':str})
    senti_dir = '/data1/brian/Quant/news_sentiment_analysis/4-sentiment_analyzing/sentiment_outputs_title/'
    src_dir = '../dataset/stock'
    tgt_dir = f'output/{item}'
    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)
    f = open(f'output/{item}_record.txt', 'w')
    f.write('code\tmin_valid_loss\tmax_valid_auc\tmin_test_loss\tmax_test_auc\tspend_time\n')
    f.close()
    for code in base_df['code'][args.n * args.per_stock: (args.n + 1) * args.per_stock]:
        start_time = time.time()
        batch_size = 16
        if not os.path.isfile(os.path.join(src_dir, '%s.csv' % code)):
            continue
        if os.path.isfile(os.path.join(tgt_dir, '%s.csv' % code)):
            continue
        # 提取股票因子
        df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
        df = extract_factor(df)
        # 读取新闻特征, 如没有该股票全部置0
        if os.path.isdir(os.path.join(senti_dir, code)):
            all_dates = os.listdir(os.path.join(senti_dir, '%s/embeddings' % code))
            all_dates = [date.replace('.pt', '') for date in all_dates]
            all_dates.sort()
            all_convert_dates = []
            for date in all_dates:
                dt_time = datetime.strptime(date, '%Y%m%d')
                time_str = dt_time.strftime('%Y-%m-%d')
                all_convert_dates.append(time_str)
            all_senti_features = [torch.load(os.path.join(senti_dir, '%s/embeddings/%s.pt' % (code, date))).view(1, -1) for date in all_dates]
            senti_features_dic = dict(zip(all_convert_dates, all_senti_features))
        else:
            senti_features_dic = {}
        train_data, valid_data, test_data = split_windows(df, senti_features_dic, size=20)
        if len(train_data[0]) == 0 or len(train_data[0]) % batch_size == 1:
            continue
        if len(test_data[0]) == 0:
            continue
        try:
            df_tst_res, min_valid_loss, max_valid_auc, min_test_loss, max_test_auc = train_test(train_data, valid_data, test_data, batch_size)
            df_tst_res.to_csv(os.path.join(tgt_dir, '%s.csv' % code), index=0)
            end_time = time.time()
            seconds = end_time - start_time
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            f = open(f'output/{item}_record.txt', 'a')
            spend_time = "%02d:%02d:%02d" % (h, m, s)
            f.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\t%s\n' % (code, min_valid_loss, max_valid_auc, min_test_loss, max_test_auc, spend_time))
            f.close()
        except:
            pass


import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy
import numpy as np
import seaborn as sns
sns.set(context='paper', style='white')
sns.set_color_codes() 
#style:darkgrid 黑色网格（默认）, whitegrid 白色网格, dark 黑色背景, white 白色背景, ticks 有刻度线的白背景
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Helvetica']



def plot(train_loss,val_loss, train_auc, valid_auc, file_name=None, model_name=None):
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    l1 = ax.plot(range(len(train_loss)), train_loss, lw=1.5, label='Train loss', color='navy') # b
    l11 = ax.plot(range(len(val_loss)), val_loss, lw=1.5, label='Valid loss', color='r') # r
    ax2 = ax.twinx()
    l2 = ax2.plot(range(len(train_auc)), train_auc, lw=1.5, label='Train AUC', color='g') # g
    l3 = ax2.plot(range(len(valid_auc)), valid_auc, lw=1.5, label='Valid AUC', color='orange') # purple

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('AUC',color='black')
    
    lns = l1+l11+l2+l3
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, frameon=True, loc='best')
    plt.grid(axis="y")
    plt.title('%s' % model_name, y=1.03)

    plt.savefig('paper_figs/{}_loss.jpg'.format(file_name), dpi=1024, bbox_inches='tight')
    plt.show()
    
    
# lstm
src_dir = 'MAS_lstm/procedure/lstm'
lstm_codes = []
for code in os.listdir(src_dir):
    df = pd.read_csv(os.path.join(src_dir, code))
    lstm_codes.append(df.values)
lstm_codes = sum(lstm_codes) / len(lstm_codes)

train_loss, train_auc = lstm_codes[:, 0], lstm_codes[:, 1]
valid_loss, valid_precision, valid_recall, valid_f1_score, valid_auc = lstm_codes[:, 2], lstm_codes[:, 3], lstm_codes[:, 4], lstm_codes[:, 5], lstm_codes[:, 6]
test_loss, test_precision, test_recall, test_f1_score, test_auc = lstm_codes[:, 7], lstm_codes[:, 8], lstm_codes[:, 9], lstm_codes[:, 10], lstm_codes[:, 11]

plot(train_loss, valid_loss, train_auc, valid_auc, file_name='lstm', model_name='LSTM Training without Enhancement')


 # transformer
src_dir = 'MAS_transformer/procedure/transformer'
transformer_codes = []
for code in os.listdir(src_dir):
    df = pd.read_csv(os.path.join(src_dir, code))
    transformer_codes.append(df.values)
transformer_codes = sum(transformer_codes) / len(transformer_codes)

train_loss, train_auc = transformer_codes[:, 0], transformer_codes[:, 1]
valid_loss, valid_precision, valid_recall, valid_f1_score, valid_auc = transformer_codes[:, 2], transformer_codes[:, 3], transformer_codes[:, 4], transformer_codes[:, 5], transformer_codes[:, 6]
test_loss, test_precision, test_recall, test_f1_score, test_auc = transformer_codes[:, 7], transformer_codes[:, 8], transformer_codes[:, 9], transformer_codes[:, 10], transformer_codes[:, 11]

plot(train_loss, valid_loss, train_auc, valid_auc, file_name='transformer', model_name='Transformer Training without Enhancement')



 #lstm enhancement title
src_dir = 'MAS_lstm_enhancement_title/procedure/lstm_enhancement_title'
lstm_codes = []
for code in os.listdir(src_dir):
    df = pd.read_csv(os.path.join(src_dir, code))
    lstm_codes.append(df.values)
lstm_codes = sum(lstm_codes) / len(lstm_codes)

train_loss, train_auc = lstm_codes[:, 0], lstm_codes[:, 1]
valid_loss, valid_precision, valid_recall, valid_f1_score, valid_auc = lstm_codes[:, 2], lstm_codes[:, 3], lstm_codes[:, 4], lstm_codes[:, 5], lstm_codes[:, 6]
test_loss, test_precision, test_recall, test_f1_score, test_auc = lstm_codes[:, 7], lstm_codes[:, 8], lstm_codes[:, 9], lstm_codes[:, 10], lstm_codes[:, 11]

plot(train_loss, valid_loss, train_auc, valid_auc, file_name='lstm_enhancement_title', model_name='LSTM Training with Title Enhancement')


 # transformer enhancement title
src_dir = 'MAS_transformer_enhancement_title/procedure/transformer_enhancement_title'
transformer_codes = []
for code in os.listdir(src_dir):
    df = pd.read_csv(os.path.join(src_dir, code))
    transformer_codes.append(df.values)
transformer_codes = sum(transformer_codes) / len(transformer_codes)

train_loss, train_auc = transformer_codes[:, 0], transformer_codes[:, 1]
valid_loss, valid_precision, valid_recall, valid_f1_score, valid_auc = transformer_codes[:, 2], transformer_codes[:, 3], transformer_codes[:, 4], transformer_codes[:, 5], transformer_codes[:, 6]
test_loss, test_precision, test_recall, test_f1_score, test_auc = transformer_codes[:, 7], transformer_codes[:, 8], transformer_codes[:, 9], transformer_codes[:, 10], transformer_codes[:, 11]

plot(train_loss, valid_loss, train_auc, valid_auc, file_name='transformer_enhancement_title', model_name='Transformer Training with Title Enhancement')



 #lstm enhancement
src_dir = 'MAS_lstm_enhancement/procedure/lstm_enhancement'
lstm_codes = []
for code in os.listdir(src_dir):
    df = pd.read_csv(os.path.join(src_dir, code))
    lstm_codes.append(df.values)
lstm_codes = sum(lstm_codes) / len(lstm_codes)

train_loss, train_auc = lstm_codes[:, 0], lstm_codes[:, 1]
valid_loss, valid_precision, valid_recall, valid_f1_score, valid_auc = lstm_codes[:, 2], lstm_codes[:, 3], lstm_codes[:, 4], lstm_codes[:, 5], lstm_codes[:, 6]
test_loss, test_precision, test_recall, test_f1_score, test_auc = lstm_codes[:, 7], lstm_codes[:, 8], lstm_codes[:, 9], lstm_codes[:, 10], lstm_codes[:, 11]

plot(train_loss, valid_loss, train_auc, valid_auc, file_name='lstm_enhancement', model_name='LSTM Training with Title and Context Enhancement')


 # transformer enhancement
src_dir = 'MAS_transformer_enhancement/procedure/transformer_enhancement'
transformer_codes = []
for code in os.listdir(src_dir):
    df = pd.read_csv(os.path.join(src_dir, code))
    transformer_codes.append(df.values)
transformer_codes = sum(transformer_codes) / len(transformer_codes)

train_loss, train_auc = transformer_codes[:, 0], transformer_codes[:, 1]
valid_loss, valid_precision, valid_recall, valid_f1_score, valid_auc = transformer_codes[:, 2], transformer_codes[:, 3], transformer_codes[:, 4], transformer_codes[:, 5], transformer_codes[:, 6]
test_loss, test_precision, test_recall, test_f1_score, test_auc = transformer_codes[:, 7], transformer_codes[:, 8], transformer_codes[:, 9], transformer_codes[:, 10], transformer_codes[:, 11]

plot(train_loss, valid_loss, train_auc, valid_auc, file_name='transformer_enhancement', model_name='Transformer Training with Title and Context Enhancement')

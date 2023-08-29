import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score



# lstm
base_df = pd.read_csv('MAS_lstm/procedure/lstm_all.csv', converters={u'code':str})
src_dir = 'MAS_lstm/procedure/lstm/'
valid_precisions, valid_recalls, valid_f1_scores, valid_aucs = [], [], [], []
test_precisions, test_recalls, test_f1_scores, test_aucs = [], [], [], []
for code in base_df['code'][:50]:
    df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
    df.sort_values(by='valid_loss', inplace=True)
    # df.sort_values(by='valid_auc', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    valid_precisions.append(df['valid_precision'][0])
    valid_recalls.append(df['valid_recall'][0])
    valid_f1_scores.append(df['valid_f1_score'][0])
    valid_aucs.append(df['valid_auc'][0])
    # 
    test_precisions.append(df['test_precision'][0])
    test_recalls.append(df['test_recall'][0])
    test_f1_scores.append(df['test_f1_score'][0])
    test_aucs.append(df['test_auc'][0])

print('lstm.')
print('Valid: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(valid_precisions), np.mean(valid_recalls), np.mean(valid_f1_scores), np.mean(valid_aucs)))
print('Test: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1_scores), np.mean(test_aucs)))



# transformer
base_df = pd.read_csv('MAS_transformer/procedure/transformer_all.csv', converters={u'code':str})
src_dir = 'MAS_transformer/procedure/transformer/'
valid_precisions, valid_recalls, valid_f1_scores, valid_aucs = [], [], [], []
test_precisions, test_recalls, test_f1_scores, test_aucs = [], [], [], []
for code in base_df['code'][:50]:
    df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
    df.sort_values(by='valid_loss', inplace=True)
    # df.sort_values(by='valid_auc', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    valid_precisions.append(df['valid_precision'][0])
    valid_recalls.append(df['valid_recall'][0])
    valid_f1_scores.append(df['valid_f1_score'][0])
    valid_aucs.append(df['valid_auc'][0])
    # 
    test_precisions.append(df['test_precision'][0])
    test_recalls.append(df['test_recall'][0])
    test_f1_scores.append(df['test_f1_score'][0])
    test_aucs.append(df['test_auc'][0])

print('transformer.')
print('Valid: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(valid_precisions), np.mean(valid_recalls), np.mean(valid_f1_scores), np.mean(valid_aucs)))
print('Test: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1_scores), np.mean(test_aucs)))



# lstm enhancement title
base_df = pd.read_csv('MAS_lstm_enhancement_title/procedure/lstm_enhancement_title_all.csv', converters={u'code':str})
src_dir = 'MAS_lstm_enhancement_title/procedure/lstm_enhancement_title/'
valid_precisions, valid_recalls, valid_f1_scores, valid_aucs = [], [], [], []
test_precisions, test_recalls, test_f1_scores, test_aucs = [], [], [], []
for code in base_df['code'][:50]:
    df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
    df.sort_values(by='valid_loss', inplace=True)
    # df.sort_values(by='valid_auc', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    valid_precisions.append(df['valid_precision'][0])
    valid_recalls.append(df['valid_recall'][0])
    valid_f1_scores.append(df['valid_f1_score'][0])
    valid_aucs.append(df['valid_auc'][0])
    # 
    test_precisions.append(df['test_precision'][0])
    test_recalls.append(df['test_recall'][0])
    test_f1_scores.append(df['test_f1_score'][0])
    test_aucs.append(df['test_auc'][0])

print('lstm enhancement title.')
print('Valid: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(valid_precisions), np.mean(valid_recalls), np.mean(valid_f1_scores), np.mean(valid_aucs)))
print('Test: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1_scores), np.mean(test_aucs)))



# transformer enhancement title
base_df = pd.read_csv('MAS_transformer_enhancement_title/procedure/transformer_enhancement_title_all.csv', converters={u'code':str})
src_dir = 'MAS_transformer_enhancement_title/procedure/transformer_enhancement_title/'
valid_precisions, valid_recalls, valid_f1_scores, valid_aucs = [], [], [], []
test_precisions, test_recalls, test_f1_scores, test_aucs = [], [], [], []
for code in base_df['code'][:50]:
    df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
    df.sort_values(by='valid_loss', inplace=True)
    # df.sort_values(by='valid_auc', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    valid_precisions.append(df['valid_precision'][0])
    valid_recalls.append(df['valid_recall'][0])
    valid_f1_scores.append(df['valid_f1_score'][0])
    valid_aucs.append(df['valid_auc'][0])
    # 
    test_precisions.append(df['test_precision'][0])
    test_recalls.append(df['test_recall'][0])
    test_f1_scores.append(df['test_f1_score'][0])
    test_aucs.append(df['test_auc'][0])

print('transformer enhancement title.')
print('Valid: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(valid_precisions), np.mean(valid_recalls), np.mean(valid_f1_scores), np.mean(valid_aucs)))
print('Test: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1_scores), np.mean(test_aucs)))



# lstm enhancement
base_df = pd.read_csv('MAS_lstm_enhancement/procedure/lstm_enhancement_all.csv', converters={u'code':str})
src_dir = 'MAS_lstm_enhancement/procedure/lstm_enhancement/'
valid_precisions, valid_recalls, valid_f1_scores, valid_aucs = [], [], [], []
test_precisions, test_recalls, test_f1_scores, test_aucs = [], [], [], []
for code in base_df['code'][:50]:
    df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
    df.sort_values(by='valid_loss', inplace=True)
    # df.sort_values(by='valid_auc', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    valid_precisions.append(df['valid_precision'][0])
    valid_recalls.append(df['valid_recall'][0])
    valid_f1_scores.append(df['valid_f1_score'][0])
    valid_aucs.append(df['valid_auc'][0])
    # 
    test_precisions.append(df['test_precision'][0])
    test_recalls.append(df['test_recall'][0])
    test_f1_scores.append(df['test_f1_score'][0])
    test_aucs.append(df['test_auc'][0])

print('lstm enhancement.')
print('Valid: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(valid_precisions), np.mean(valid_recalls), np.mean(valid_f1_scores), np.mean(valid_aucs)))
print('Test: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1_scores), np.mean(test_aucs)))



# transformer enhancement
base_df = pd.read_csv('MAS_transformer_enhancement/procedure/transformer_enhancement_all.csv', converters={u'code':str})
src_dir = 'MAS_transformer_enhancement/procedure/transformer_enhancement/'
valid_precisions, valid_recalls, valid_f1_scores, valid_aucs = [], [], [], []
test_precisions, test_recalls, test_f1_scores, test_aucs = [], [], [], []
for code in base_df['code'][:50]:
    df = pd.read_csv(os.path.join(src_dir, '%s.csv' % code))
    df.sort_values(by='valid_loss', inplace=True)
    # df.sort_values(by='valid_auc', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    valid_precisions.append(df['valid_precision'][0])
    valid_recalls.append(df['valid_recall'][0])
    valid_f1_scores.append(df['valid_f1_score'][0])
    valid_aucs.append(df['valid_auc'][0])
    # 
    test_precisions.append(df['test_precision'][0])
    test_recalls.append(df['test_recall'][0])
    test_f1_scores.append(df['test_f1_score'][0])
    test_aucs.append(df['test_auc'][0])

print('transformer enhancement.')
print('Valid: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(valid_precisions), np.mean(valid_recalls), np.mean(valid_f1_scores), np.mean(valid_aucs)))
print('Test: precision: %.4f, recall: %.4f, fl_score: %.4f, auc: %.4f.' % (np.mean(test_precisions), np.mean(test_recalls), np.mean(test_f1_scores), np.mean(test_aucs)))


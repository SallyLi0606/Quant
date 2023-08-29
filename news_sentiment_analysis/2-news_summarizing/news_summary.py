""" 基于Pegasus-燃灯的进行中文新闻摘要
"""

import os
import time
import torch
import random
import numpy as np
import pandas as pd
from transformers import PegasusForConditionalGeneration
from pegasus_utils import PegasusTokenizer
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def find_closest_match(string, strings):
    """ 鲁棒性模糊匹配标题, 排除空格影响
    """
    highest_ratio = -1
    closest_match = None
    for s in strings:
        ratio = fuzz.ratio(string, s)
        if ratio > highest_ratio:
            highest_ratio = ratio
            closest_match = s
    return closest_match


def load_code_news(titles, code_dir):
    """ 加载给定代码的股票的新闻和对应标题
    """
    news = []
    for title in titles:
        txt_dir = os.path.join(code_dir, title.replace('/', '')+'.txt')
        if os.path.exists(txt_dir):
            pass
        else:
            title_ = find_closest_match(title, os.listdir(code_dir))
            txt_dir = os.path.join(code_dir, title_)
        with open(txt_dir, 'r') as f:
            text = f.readlines()
        text = ''.join(text)
        news.append(text)
    return news

def split_list(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def read_file(path):
    """ 对含Tab的标题的鲁棒加载
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    lines_split = []
    for line in lines:
        line = line.strip()
        line_split = line.split('\t')
        # 若新闻标题包含Tab (导致列数不匹配)
        if len(line_split) > 2:
            date = line_split[0]
            print(line_split[1:])
            title = '\t'.join(line_split[1:])
            line_split = [date, title]
        lines_split.append(line_split)
    code_file = pd.DataFrame(lines_split, columns=['date', 'news_title'])
    return code_file


if __name__ == "__main__":
    device = torch.device('cuda:1')
    # 采集的新闻地址
    news_base = "/data1/brian/Quant/news_sentiment_analysis/1-news_collecting/stock_news"
    # 新闻摘要保存地址
    summarized_news_base = "/data1/brian/Quant/news_sentiment_analysis/2-news_summarizing/summarized_stock_news"
    os.makedirs(summarized_news_base, exist_ok=True)

    t1 = time.time()
    model = PegasusForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")
    tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")
    model.to(device)
    t2 = time.time()
    print(f"模型加载用时: {t2-t1:.1f}")

    # codes_file = [p for p in sorted(os.listdir(news_base)) if os.path.isfile(os.path.join(news_base, p))]
    codes_file = [p for p in reversed(sorted(os.listdir(news_base))) if os.path.isfile(os.path.join(news_base, p))]
    for i, code_file in enumerate(codes_file):
        code = code_file.split('.')[0]
        print("> 对 {} / {} 股票 {} 新闻进行文本摘要".format(i+1, len(codes_file), code))
        # code_file = pd.read_csv(os.path.join(news_base, code_file), sep='\t', header=None).astype(str)
        # code_file.rename(columns={0: 'date', 1: 'news_title'}, inplace=True)
        code_file = read_file(os.path.join(news_base, code_file)).astype(str)
        code_file = code_file.drop_duplicates(subset='date', keep='last').reset_index(drop=True) # 对于同一天多条新闻, 只保留最早的新闻
        news_titles = code_file['news_title'].tolist()

        code_dir = os.path.join(news_base, code)
        summarized_code_dir = os.path.join(summarized_news_base, code+'.csv')
        if os.path.exists(summarized_code_dir):
            print("本文摘要已完成".format(code))
            continue
        if len(os.listdir(code_dir)) == 0:
            print("! 符合要求的新闻为空".format(code))
            continue

        news = load_code_news(news_titles, code_dir)
        print("新闻个数: ", len(news))
        split_news = split_list(news, 100) # 每100个新闻一个batch
        all_summaries = []
        for i, news_subset in enumerate(split_news):
            print("第 {} 部分".format(i+1))
            # 文本编码
            inputs = tokenizer.batch_encode_plus(news_subset, max_length=256, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs["input_ids"].to(device)

            # 摘要生成
            with torch.no_grad():
                summary_ids = model.generate(inputs, max_new_tokens=32).detach().cpu()
            summary_outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            all_summaries += summary_outputs
        
        code_file['summary'] = all_summaries
        code_file.to_csv(summarized_code_dir, index=False, sep='\t')
        print("=== x === " * 5)
        
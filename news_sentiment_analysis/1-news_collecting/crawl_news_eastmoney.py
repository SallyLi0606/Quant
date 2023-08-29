import os
import urllib.request
from datetime import datetime
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool


def match_date(date_str):
    # 从字符串中匹配年月日
    pattern = r"\d{4}-\d{2}-\d{2}"
    match = re.search(pattern, date_str)
    if match:
        datetime_str = match.group(0)
    else:
        datetime_str = '0000-00-00'
    return datetime_str

#获取单页内容
def get_per_page(news_titles, news_dir):
    links = ["http://guba.eastmoney.com/o"+s.find('a')['href'] for s in news_titles]
    assert len(links) == len(news_titles)
    news = [] #用于存放资讯和时间
    stop_flag = False # 判断下一资讯列表页是否需要继续爬
    for title, link in zip(news_titles, links):
        title = title.a.attrs['title']
        # print(title, link)
        try:
            req = urllib.request.Request(link, headers={'UserWarning-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.48'})
            resp = urllib.request.urlopen(req, timeout=5)
            html = resp.read().decode('utf-8')
        except:
            print("{} 404 Not Found".format(link))
            continue
        soup = BeautifulSoup(html, 'lxml')
        # 获取新闻时间
        # date = soup.find('div', class_='zwfbtime').text.strip().split(' ')[1:-1]
        # date = ' '.join(date)
        date = soup.find('div', class_='zwfbtime').text.strip()
        date = match_date(date)
        datetime_object = datetime.strptime(date, "%Y-%m-%d")
        print(date + '\t' + title)
        if datetime_object.year < 2020: # 遇到2020年以前的新闻, 停止采集, 提前终止
            stop_flag = True
            break
        # 获取信息内容
        content = soup.find('div', {'id': 'zw_body'})
        if content is not None:
            content = [c.text.strip() for c in content.find_all('p', attrs={'style': None})]
            content = '\n'.join(content)
        else:
            try:
                content = soup.find('div', {'id': 'autowrite-box'}).text
            except AttributeError:
                continue

        # 保存新闻文本
        try:
            with open(os.path.join(news_dir, title.replace('/', '')+'.txt'), 'w') as f:
                f.writelines(content)
        except OSError as NameToLongError:
            title = title.replace('/', '')[:64] # 新闻标题太长时截断到64个字
            with open(os.path.join(news_dir, title+'.txt'), 'w') as f:
                f.writelines(content)
        news.append(date + '\t' + title)

    return (news, stop_flag)

#获取多页内容
def get_pages(stock_code, news_dir):
    news = []
    i = 0
    max_pages = 100
    while True:
        if i==1:
            url = 'http://guba.eastmoney.com/list,' + stock_code + ',1,f.html'
        else:
            url = 'http://guba.eastmoney.com/list,' + stock_code + ',1,f_'+str(i)+'.html'
        
        try:
            req = urllib.request.Request(url, headers={'UserWarning-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.48'})
            resp = urllib.request.urlopen(req)
            html = resp.read().decode('utf-8')
        except:
            print("===== x ===== " * 5, '\n' * 3)
            print("股票{} 页面{} 请求失败".format(stock_code, i+1))
            print('\n' * 3, "===== x ===== " * 5)
            continue
        soup = BeautifulSoup(html, 'lxml')
        news_titles = soup.find_all('span', {'class':'l3 a3'})[1:]
        # 空页面, 超过实际最大页码数
        if len(news_titles) == 0:
            break
        page_news, stop_flag = get_per_page(news_titles, news_dir)
        news = news + page_news
        i += 1
        # 遇到2020年以前的新闻, 停止采集下一资讯页新闻
        if stop_flag:
            break
        if i > max_pages:
            break

    return news

#保存
def collect_news(stock_codes):
    for code in stock_codes:
        print("> 采集股票 {}".format(code))
        news_dir = os.path.join(news_base, code)
        os.makedirs(news_dir, exist_ok=True)
        news_txt = os.path.join(news_base, '{}.txt'.format(code))
        if os.path.exists(news_txt):
            print("! 股票 {} 已采集过".format(code))
            continue
        news = get_pages(code, news_dir)
        # 正常采集到数据集 (解决全部咨询业请求失败问题)
        if len(news) != 0:
            with open(news_txt, 'w') as f:
                f.writelines(l + '\n' for l in news)

if __name__ == "__main__":
    # stock_codes = pd.read_csv("/data1/brian/Quant/dataset/all_codes.csv")['code'].astype(str).str.zfill(6).tolist()
    stock_codes = pd.read_csv("/data1/brian/Quant/dataset/sup_codes.csv")['code'].astype(str).str.zfill(6).tolist()
    stock_codes = list(np.array_split(stock_codes, 4)[3])
    news_base = "/data1/brian/Quant/news_sentiment_analysis/1-news_collecting/stock_news"
    
    NUM_PROCS = 16
    pool = ProcessPool(NUM_PROCS)
    split_codes = np.array_split(stock_codes, NUM_PROCS)
    pool.map(collect_news, split_codes)
    # collect_news(stock_codes)
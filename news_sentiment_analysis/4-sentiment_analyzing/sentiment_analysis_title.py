import os
import math
import torch
import pandas as pd
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification, AutoConfig


if __name__ == '__main__':
    max_len = 128
    gpu = 0
    bs = 2048
    # 加载步骤3中微调过的情感预测模型
    model_path = "/data1/brian/Quant/news_sentiment_analysis/3-sentiment_finetuning/outputs/2023-05-24-19:18:16/albert_finetuned_ep27"
    model_config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    summarized_news_base = "/data1/brian/Quant/news_sentiment_analysis/2-news_summarizing/summarized_stock_news"
    sentiment_analysis_base = "/data1/brian/Quant/news_sentiment_analysis/4-sentiment_analyzing/sentiment_outputs_title"
    os.makedirs(sentiment_analysis_base, exist_ok=True)
    for code_file in sorted(os.listdir(summarized_news_base)):
        code = code_file.split('.')[0]
        print("> 对股票 {} 新闻进行情感分析和表征".format(code))
        code_file = pd.read_csv(os.path.join(summarized_news_base, code_file), sep='\t')
        code_file = code_file.dropna(how='any', axis=0)

        sentiment_analysis_dir = os.path.join(sentiment_analysis_base, code)
        if os.path.exists(sentiment_analysis_dir):
            print("情感表征和分析已完成".format(code))
            continue
        os.makedirs(sentiment_analysis_dir)
        sentiment_emb_dir = os.path.join(sentiment_analysis_dir, 'embeddings')
        sentiment_stats_dir = os.path.join(sentiment_analysis_dir, 'stats')
        os.makedirs(sentiment_emb_dir)
        os.makedirs(sentiment_stats_dir)

        dates, titles = code_file['date'].tolist(), code_file['news_title'].tolist()
        # 文本编码
        inputs = tokenizer.batch_encode_plus(titles, max_length=max_len, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 摘要生成
        with torch.no_grad():
            outputs = model(**inputs)
        logits = torch.softmax(outputs.logits.detach().cpu(), dim=-1)
        logits_df = pd.DataFrame(logits.numpy(), columns=[0, 1])
        logits_df.to_csv(os.path.join(sentiment_stats_dir, 'logits.csv'), index=False)
        pred = torch.argmax(logits, dim=-1)
        pred_df = pd.DataFrame(pred.numpy(), columns=['sentiment'])
        pred_df.to_csv(os.path.join(sentiment_stats_dir, 'pred.csv'), index=False)
        mean_hidden_states = outputs.hidden_states[-1].mean(axis=1).detach().cpu()
        for (date, mean_hidden_state) in zip(dates, mean_hidden_states):
            date = date.replace('-', '')
            torch.save(mean_hidden_state, os.path.join(sentiment_emb_dir, date+'.pt'))
        
        print("=== x === " * 5)
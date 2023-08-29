import os
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    # sentiment_outputs = "/data1/brian/Quant/news_sentiment_analysis/4-sentiment_analyzing/sentiment_outputs"
    sentiment_outputs = "/data1/brian/Quant/news_sentiment_analysis/4-sentiment_analyzing/sentiment_outputs_title"
    senti_positive_ratio = []
    for code in tqdm(os.listdir(sentiment_outputs)):
        senti_polar_preds = pd.read_csv(os.path.join(sentiment_outputs, code, 'stats', 'pred.csv'))['sentiment'].astype(int)
        positive_ratio = sum(senti_polar_preds) / len(senti_polar_preds)
        senti_positive_ratio.append([code, positive_ratio])
    
    senti_positive_rank = pd.DataFrame(senti_positive_ratio, columns=['code', 'positive_ratio'])
    senti_positive_rank['code'] = senti_positive_rank['code'].astype(str).str.zfill(6)
    senti_positive_rank = senti_positive_rank.sort_values(by='positive_ratio', ascending=False)
    # senti_positive_rank.to_csv("/data1/brian/Quant/news_sentiment_analysis/4-sentiment_analyzing/senti_postive_rank.csv", index=False)
    senti_positive_rank.to_csv("/data1/brian/Quant/news_sentiment_analysis/4-sentiment_analyzing/senti_postive_rank_title.csv", index=False)

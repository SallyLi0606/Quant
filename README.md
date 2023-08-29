# A deep learning approach with extensive sentiment analysis for quantitative investment

## Section 1: News Sentiment Analysis
Inside the `news_sentiment_analysis` folder, including:
* News collecting from East Money (www.eastmoney.com)
* News contents summarizing with pretrained Pegasus
* Sentiment model finetuning with ChnSentiCorp dataset
* Sentiment analyzing for merged news titles and summarized contents


## Section 2: Deep Learning Model Building
Inside the `modeling` folder:

### 1. Deep Learning Model for Experiments
There are six subfolders, each corresponding to one of the six experimental groups, as follows:

|     | LSTM  | Transformer |
|  ----  | ----  |----|
| Title + Content | `MAS_lstm_enhancement/` | `MAS_transformer_enhancement/` |
| Title | `MAS_lstm_enhancement_title/` | `MAS_transformer_enhancement_title/`|
|  Vanilla | `MAS_lstm/` | `MAS_transformer/` |

### 2. Subfolder and File Meanings for Each Experimental Group
Inside each experimental group subfolder:

- `backtrader_sequence_model.py`: Code for deep learning model to predict stock price movements.
- `run.py`: Script to run `backtrader_sequence_model.py`.
- `output/`: Results of the deep learning model predictions.
- `backtrader_mystrategy.ipynb`: Code for backtesting strategies.
- `results/`: Results of strategy backtesting.
- `plot.py`: Code to plot strategy returns and benchmark comparisons, with image paths in `figs/`.
- `procedure/`: Records training process metrics for the top 50 stocks.

### 3. Additional Files
Inside the `MAS-2023-code/` path:
- `train_procedure.py`: Code for plotting metrics for the top 50 stocks.
- `evaluate.py`: Code for calculating metrics for the top 50 stocks.

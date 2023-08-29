import pandas as pd
import quantstats
import qstock as qs
import seaborn as sns
sns.set(context='paper', style='white')
sns.set_color_codes()

returns = pd.read_csv('results/my_strategy_returns.csv')
returns = pd.Series(returns['return'].values, index=returns['index'])
returns.index = pd.to_datetime(returns.index)

benchmark_df = qs.get_data('000300', '2022-06-01', '2022-12-12')  # 通过qstock获取
# benchmark_df = pd.read_csv('../dataset/benchmark_stock_000300.csv')  # 保存好了在本地
benchmark_df.index = pd.to_datetime(benchmark_df.index)
benchmark_df = benchmark_df[['open', 'high', 'low', 'close', 'volume']]

quantstats.plots.snapshot(returns, savefig={'fname':'figs/quantstats_snap.jpg', 'dpi':1024})
fig = quantstats.plots.returns(returns, benchmark=benchmark_df, savefig={'fname':'figs/quantstats_contrast.jpg', 'bbox_inches':'tight', 'dpi':1024})
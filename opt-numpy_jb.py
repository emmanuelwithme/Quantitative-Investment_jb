# 使用numpy+joblib平行運算，把多核心CPU發揮到極致
import numpy as np
import pandas as pd
import openpyxl
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

def getStrategyReturn(df_original, M, a, N, RSI_days, rsi_lower, rsi_upper, stop_loss_rate):
    df = df_original.copy(deep=True)
    
    # 計算前M日最高量
    df['前M日最高量'] = df['Volume'].shift().rolling(window=M).max()
    
    # 計算今日是否為真實新高
    df['今日為真實新高'] = df['Volume'] > a * df['前M日最高量']
    
    # 計算近N日是否出現真實最高
    df['近N日出現真實最高'] = df['今日為真實新高'].rolling(window=N, min_periods=1).apply(lambda x: np.any(x)).astype(bool)

    # 計算RSI指標
    def calculate_rsi(df, window):
        close_prices = df['Close']
        price_diff = close_prices.diff()
        upward_move = np.where(price_diff > 0, price_diff, 0)
        downward_move = np.abs(np.where(price_diff < 0, price_diff, 0))
        avg_gain = pd.Series(upward_move).rolling(window).mean()
        avg_loss = pd.Series(downward_move).rolling(window).mean()
        relative_strength = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + relative_strength))
        return rsi

    df['RSI'] = calculate_rsi(df, RSI_days)
    df['RSI'] = np.round(df['RSI'], 1)
    df['RSI多空'] = np.where(df['RSI'] > rsi_upper, -2, np.where(df['RSI'] <= rsi_lower, 2, 0))

    # 初始化部位流量、部位存量、停損價格、帳戶餘額和對準市值
    df['部位流量'] = 0
    df['部位存量'] = 0
    df['停損價格'] = 0
    df['帳戶餘額'] = 0
    df['對準市值'] = 0

    for i in range(1, len(df)):
        prev_day_flow = df.at[i - 1, '部位流量']
        prev_inventory = df.at[i - 1, '部位存量']
        current_inventory = prev_day_flow + prev_inventory

        if current_inventory == 0:
            df.at[i, '停損價格'] = 0
        elif prev_inventory in [-1, 0] and current_inventory == 1:
            df.at[i, '停損價格'] = df.at[i, 'Open'] * (1 - stop_loss_rate)
        elif prev_inventory in [0, 1] and current_inventory == -1:
            df.at[i, '停損價格'] = df.at[i, 'Open'] * (1 + stop_loss_rate)
        else:
            df.at[i, '停損價格'] = df.at[i - 1, '停損價格']

        if current_inventory == 0:
            if df.at[i, '近N日出現真實最高'] and df.at[i, 'RSI多空'] == 2:
                df.at[i, '部位流量'] = 1
            elif df.at[i, '近N日出現真實最高'] and df.at[i, 'RSI多空'] == -2:
                df.at[i, '部位流量'] = -1
        elif current_inventory == 1:
            if df.at[i, '近N日出現真實最高'] and df.at[i, 'RSI多空'] == -2:
                df.at[i, '部位流量'] = -2
        elif current_inventory == -1:
            if df.at[i, 'Close'] >= df.at[i, '停損價格']:
                df.at[i, '部位流量'] = 1
            elif df.at[i, '近N日出現真實最高'] and df.at[i, 'RSI多空'] == 2:
                df.at[i, '部位流量'] = 2

        account_balance = df.at[i - 1, '帳戶餘額'] - (prev_day_flow * df.at[i, 'Open'])
        target_value = account_balance + (current_inventory * df.at[i, 'Close'])

        df.at[i, '部位存量'] = current_inventory
        df.at[i, '帳戶餘額'] = account_balance
        df.at[i, '對準市值'] = target_value

    return df.iloc[-1]['對準市值']


df1 = pd.read_excel('data/三個樣本期.xlsx', sheet_name='Sample1', header=1, engine='openpyxl')
df2 = pd.read_excel('data/三個樣本期.xlsx', sheet_name='Sample2', header=1, engine='openpyxl')
df3 = pd.read_excel('data/三個樣本期.xlsx', sheet_name='Sample3', header=1, engine='openpyxl')

df1 = df1.rename(columns={'日期': 'Date', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low', '收盤價': 'Close', '成交量': 'Volume'})
df2 = df2.rename(columns={'日期': 'Date', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low', '收盤價': 'Close', '成交量': 'Volume'})
df3 = df3.rename(columns={'日期': 'Date', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low', '收盤價': 'Close', '成交量': 'Volume'})

def getBandHReturn(df):
    first_open = df.iloc[0]['Open']
    last_close = df.iloc[-1]['Close']
    return last_close - first_open

BandHReturn1 = getBandHReturn(df1)
BandHReturn2 = getBandHReturn(df2)
BandHReturn3 = getBandHReturn(df3)

param_grid = {'M_RSI_days': list(range(10, 130, 5)),
              'a': [1.0, 1.05, 1.1, 1.15, 1.2],
              'N': list(range(1, 11)),
              'rsi_lower': list(range(15, 50, 5)),
              'rsi_upper': list(range(55, 90, 5)),
              'stop_loss_rate': [0.1, 0.15, 0.2, 0.25]
              }

# 測試用參數組合
# test_param_grid = {'M_RSI_days': np.arange(10, 130, 50),
#                    'a': np.array([1]),
#                    'N': np.arange(1, 11),
#                    'rsi_lower': np.arange(15, 50, 30),
#                    'rsi_upper': np.arange(55, 90, 50),
#                    'stop_loss_rate': np.array([0.1])
#                   }

grid = ParameterGrid(param_grid)

# 寫法一 start
results1 = Parallel(n_jobs=-1)(delayed(getStrategyReturn)(df1, params['M_RSI_days'], params['a'], params['N'], params['M_RSI_days'], params['rsi_lower'], params['rsi_upper'], params['stop_loss_rate']) for params in tqdm(grid))
results2 = Parallel(n_jobs=-1)(delayed(getStrategyReturn)(df2, params['M_RSI_days'], params['a'], params['N'], params['M_RSI_days'], params['rsi_lower'], params['rsi_upper'], params['stop_loss_rate']) for params in tqdm(grid))
results3 = Parallel(n_jobs=-1)(delayed(getStrategyReturn)(df3, params['M_RSI_days'], params['a'], params['N'], params['M_RSI_days'], params['rsi_lower'], params['rsi_upper'], params['stop_loss_rate']) for params in tqdm(grid))
# 寫法一 end

# 寫法二－加入總進度條 start
# ...
# 寫法二－加入總進度條 end

results1_df = pd.DataFrame(grid)
results1_df['StrategyReturn'] = results1
results1_df['BandHReturn1'] = BandHReturn1

results2_df = pd.DataFrame(grid)
results2_df['StrategyReturn'] = results2
results2_df['BandHReturn2'] = BandHReturn2

results3_df = pd.DataFrame(grid)
results3_df['StrategyReturn'] = results3
results3_df['BandHReturn3'] = BandHReturn3

results1_df = results1_df.sort_values(by='StrategyReturn', ascending=False)
results2_df = results2_df.sort_values(by='StrategyReturn', ascending=False)
results3_df = results3_df.sort_values(by='StrategyReturn', ascending=False)

# 複製 "M_RSI_days" 欄位並將其命名為 "RSI_days"
results1_df['RSI_days'] = results1_df['M_RSI_days'].copy()
results2_df['RSI_days'] = results2_df['M_RSI_days'].copy()
results3_df['RSI_days'] = results3_df['M_RSI_days'].copy()

# 將 "M_RSI_days" 欄位更名為 "M"
results1_df = results1_df.rename(columns={'M_RSI_days': 'M'})
results2_df = results2_df.rename(columns={'M_RSI_days': 'M'})
results3_df = results3_df.rename(columns={'M_RSI_days': 'M'})

# 按照指定順序重新排列欄位，欄位順序：M, a, N, RSI_days, rsi_lower, rsi_upper, stop_loss_rate, StrategyReturn, BandHReturn1
results1_df = results1_df.reindex(columns=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate', 'StrategyReturn', 'BandHReturn1'])
results2_df = results2_df.reindex(columns=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate', 'StrategyReturn', 'BandHReturn2'])
results3_df = results3_df.reindex(columns=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate', 'StrategyReturn', 'BandHReturn3'])

results1_df.to_csv('strategy_result20230523_jb/Sample1.csv', index=False)
results2_df.to_csv('strategy_result20230523_jb/Sample2.csv', index=False)
results3_df.to_csv('strategy_result20230523_jb/Sample3.csv', index=False)

top_10_df1 = results1_df.head(10)
top_10_df2 = results2_df.head(10)
top_10_df3 = results3_df.head(10)

print('樣本一報酬最高前10:\n', top_10_df1)
print('樣本二報酬最高前10:\n', top_10_df2)
print('樣本三報酬最高前10:\n', top_10_df3)
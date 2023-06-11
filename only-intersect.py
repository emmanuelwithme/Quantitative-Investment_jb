import pandas as pd
# --------------下面為cross-intersection------------
# 讀取結果 CSV 檔案
results1_df = pd.read_csv('strategy_result20230523_jb/Sample1.csv')
results2_df = pd.read_csv('strategy_result20230523_jb/Sample2.csv')
results3_df = pd.read_csv('strategy_result20230523_jb/Sample3.csv')

# 獲取每個 DataFrame 的行數
num_rows1 = results1_df.shape[0]
num_rows2 = results2_df.shape[0]
num_rows3 = results3_df.shape[0]

# 計算行數的最小值
min_rows = min(num_rows1, num_rows2, num_rows3)

# num_params為已經找到交集的參數組合的數量
num_params = 0

# num_top_params為一開始要從多少筆資料開始交集
num_top_params = 100

# max_params為要尋找的多少組最大參數組和的數量
max_params = 100

# 使用行數的最小值作為 num_top_params 的最大值
max_top_params = min_rows

# 初始化 intersection_params
intersection_params = pd.DataFrame()

# num_params >= max_params已找到足夠的參數組
# num_top_params > max_top_params已經找到最後一筆資料
while num_params < max_params and num_top_params <= max_top_params:
    # 選取前 num_top_params 個參數組
    top_params1 = results1_df.head(num_top_params)
    top_params2 = results2_df.head(num_top_params)
    top_params3 = results3_df.head(num_top_params)

    # 獲取參數組的交集
    # 樣本一前XX名跟樣本二前XX名取交集，前提是兩個data都要['M','a','N','RSI_days','rsi_lower','rsi_upper','stop_loss_rate']這些欄位一樣
    # 另外樣本一的StrategyReturn增加suffixes='1'，變成StrategyReturn1。另外樣本二的StrategyReturn增加suffixes='2'，變成StrategyReturn2。
    intersection_params = pd.merge(top_params1, top_params2, how='inner', on=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate'], suffixes=('1', '2'))
    intersection_params = pd.merge(intersection_params, top_params3, how='inner', on=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate'])

    # 重新命名 StrategyReturn 欄位
    intersection_params = intersection_params.rename(columns={
        'StrategyReturn': 'StrategyReturn3'
    })

    # 删除重复的参数组合
    intersection_params = intersection_params.drop_duplicates()

    # 計算交集中的參數組數量
    num_params = len(intersection_params)

    # 如果找不到足夠的參數組，則增加 num_top_params
    if num_params < max_params:
        num_top_params += 40

# 印出找到的參數組
print("找到的參數組如下：")
print(intersection_params)


# 創建一個新的 list 存儲三個樣本共同較好的參數組合
common_params_list = []

# 遍歷每個共同參數
for index, row in intersection_params.iterrows():
    # 獲取該參數在每個樣本中的報酬
    strategy_return1 = row['StrategyReturn1']
    strategy_return2 = row['StrategyReturn2']
    strategy_return3 = row['StrategyReturn3']

    # 計算策略相對於 BnHReturn 的報酬
    relative_return1 = strategy_return1 - row['BandHReturn1']
    relative_return2 = strategy_return2 - row['BandHReturn2']
    relative_return3 = strategy_return3 - row['BandHReturn3']

    # 儲存結果
    common_params_list.append({
        'M': row['M'],
        'a': row['a'],
        'N': row['N'],
        'RSI_days': row['RSI_days'],
        'rsi_lower': row['rsi_lower'],
        'rsi_upper': row['rsi_upper'],
        'stop_loss_rate': row['stop_loss_rate'],
        'Sample1_StrategyReturn': strategy_return1,
        'Sample2_StrategyReturn': strategy_return2,
        'Sample3_StrategyReturn': strategy_return3,
        'Sample1_BnHReturn': row['BandHReturn1'],
        'Sample2_BnHReturn': row['BandHReturn2'],
        'Sample3_BnHReturn': row['BandHReturn3'],
        'Sample1_RelativeReturn': relative_return1,
        'Sample2_RelativeReturn': relative_return2,
        'Sample3_RelativeReturn': relative_return3
    })

# 將列表轉換為 DataFrame
common_params_df = pd.DataFrame(common_params_list)

# 儲存 DataFrame 為 CSV 檔案
common_params_df.to_csv('strategy_result20230523_jb/common_params.csv', index=False)
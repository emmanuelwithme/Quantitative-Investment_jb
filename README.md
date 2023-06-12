# Quantitative-Investment_jb
量化交易分析－第四組－台指期交易模組三建置、回測與最佳化參數，並且加入CPU平行計算加快7倍速度
## 模組三

### 精神
模組三我們使用python程式最佳化策略的參數，我們總共有7個參數，`M`和`RSI_days`是同步變更:

- M
- a
- N
- RSI_days
- rsi_lower
- rsi_upper
- stop_loss_rate

這些參數分別用於兩種策略：

1. 成交量爆量策略：檢查過去 `N` 天的當日成交量是否是近 `M` 天最大成交量的 `a` 倍。
2. 價格低買高賣策略：根據 RSI (RSI 計算範圍天數 `RSI_days`) 進行交易，當 RSI <= `rsi_lower` 時買入，當 RSI > `rsi_upper` 時賣出，其他情況則不進行交易。
3. 同時，在做空時如果損失超過停損 `stop_loss_rate`，則出清做空部位。

條件一「**成交量爆量**」為必要條件，一定要符合才會做多或做空。因為我們觀察認為成交量突然放大是利空出盡的訊號，買賣方從一致的意見轉為對峙，這時很有機會反轉。

------------

### 回測
我們將原本的 Excel 回測交易策略模型改寫成了 Python 函式 `getStrategyReturn`：

```python
getStrategyReturn(df_original,M,a,N,RSI_days,rsi_lower,rsi_upper,stop_loss_rate)
#省略
#得到策略總報酬
return df.iloc[-1]['對準市值']
```

------------

### 最佳化參數 (Grid Search)
#### 找到最好的參數
接下來，我們使用網格搜索（Grid Search）來進行最佳化參數的選擇。網格搜索是一種通過窮舉所有可能的參數組合來尋找最佳解的方法。

在樣本一中，我們嘗試了 52,920 種參數組合，並根據策略報酬對這些參數組合進行排名。我們找到了最好的參數組合：
- M = 30
- a = 1.1
- N = 5
- RSI_days = 30
- rsi_lower = 35
- rsi_upper = 60
- stop_loss_rate = 0.1

此時的策略報酬為 10,252 元。

#### 但是發現參數不適用於其他樣本
然後我把樣本一52920種參數組合中績效前1000好的參數組合拿出來放入樣本二跟樣本三跑，計算報酬後發現沒有一個參數組合同時在樣本一、樣本二、樣本三超過B&H報酬，所以可以推測可能是over-fitting了，雖然在樣本一中最佳的參數可以跑到10252元報酬，是B&H報酬的3倍左右，當初跑出來的時候直接嚇到下巴掉下來，覺得報酬不可思議，但是依然在樣本二、樣本三跑的不如理想。

#### 我想到解決辦法
為了解決這個問題，後來我加入cross-intersection的策略，這是我自己想出來的策略，我可以在樣本一、二、三都進行相同的52920種參數組合測試績效，然後把各樣本績效前2000好的參數組合取交集，如果有找到10筆，代表那10筆都在樣本一、二、三排名前2000名，所以是好的參數；如果沒有找到交集的參數組合，就繼續尋找前2500名、前3000名、前4000名、前4500名...，一直找下去，相信這個方法可以找到雖然報酬不是最高，但是是在各個樣本中都是前幾高的，更通用、適合各種樣本時段的參數。

#### 速度瓶頸
我的速度會陷入瓶頸，我把參數組合提升到 **`235200` (20多萬)** 種之後，運算要**``235200*360*5/7*10*3 = 1814400000 (18億次iterations)``**，會需要超過100個小時，陷入了效能瓶頸。於是我加入了以下3方法，改寫程式碼，加速運算速度:
1. 「CPU多核心平行運算」: `joblib` 庫中的 `Parallel` 和 `delayed` 用於並行計算。
2. 「RSI計算改用NumPy加速」: `NumPy`的資料結構運算效率比`Pandas.Dataframe()`高。

#### 解釋改良過後的程式碼
我加入「cross-intersection」、「CPU多核心平行運算」、「RSI計算改用NumPy加速」改良程式碼。
#### 計算20多萬種參數組合在三個樣本中的績效
計算績效: `opt-numpy_jb.py`
接下來我要解釋我的程式如何進行網格搜索（Grid Search）最佳化參數:

步驟:
1. **匯入所需的函式庫和模組**
```python
import numpy as np
import pandas as pd
import openpyxl
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
```
2. **定義策略函數 `getStrategyReturn()`**
```python
def getStrategyReturn(df_original, M, a, N, RSI_days, rsi_lower, rsi_upper, stop_loss_rate):
    # 省略函數內容
    return df.iloc[-1]['對準市值']
```
3. **讀取數據並進行前處理**
```python
df1 = pd.read_excel('data/三個樣本期.xlsx', sheet_name='Sample1', header=1, engine='openpyxl')
df2 = pd.read_excel('data/三個樣本期.xlsx', sheet_name='Sample2', header=1, engine='openpyxl')
df3 = pd.read_excel('data/三個樣本期.xlsx', sheet_name='Sample3', header=1, engine='openpyxl')
# 重新命名欄位名稱
df1 = df1.rename(columns={'日期': 'Date', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low', '收盤價': 'Close', '成交量': 'Volume'})
df2 = df2.rename(columns={'日期': 'Date', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low', '收盤價': 'Close', '成交量': 'Volume'})
df3 = df3.rename(columns={'日期': 'Date', '開盤價': 'Open', '最高價': 'High', '最低價': 'Low', '收盤價': 'Close', '成交量': 'Volume'})
```
4. **定義獲取買並持有策略回報的函數** `getBandHReturn()`
```python
def getBandHReturn(df):
    first_open = df.iloc[0]['Open']
    last_close = df.iloc[-1]['Close']
    return last_close - first_open
```
5. **定義參數網格** `param_grid`: 我定義了一個參數空間 `param_grid`，其中包括了所有可能的參數組合。每個參數都有一個範圍或列表，例如，`M_RSI_days` 的範圍是從10到120，步長為10。
```python
param_grid = {
    'M_RSI_days': list(range(10, 130, 5)),  # RSI計算的天數範圍，從10到130，步長為5
    'a': [1.0, 1.05, 1.1, 1.15, 1.2],  # 用於計算真實新高的乘數範圍
    'N': list(range(1, 11)),  # 近N天是否出現真實最高的天數範圍，從1到10
    'rsi_lower': list(range(15, 50, 5)),  # RSI下界範圍，從15到50，步長為5
    'rsi_upper': list(range(55, 90, 5)),  # RSI上界範圍，從55到90，步長為5
    'stop_loss_rate': [0.1, 0.15, 0.2, 0.25]  # 停損比例範圍，包括0.1、0.15、0.2和0.25
}
```
6. **創建參數網格** `grid`: 使用 `scikit-learn`中的 `ParameterGrid` 模組從參數空間創建一個參數網格，其中包含所有可能的參數組合。
```python
grid = ParameterGrid(param_grid)
```
7. **CPU並行計算20萬種參數組合績效**: 我們對參數網格中的每一個參數組合進行迭代嘗試（使用 `tqdm` 來顯示進度條），並計算每組參數的策略回報。我使用 `Parallel` 函數和 `delayed` 函數進行`CPU平行計算`，並且將 `getStrategyReturn` 函數應用於每個參數組合和樣本數據。這將同時運行三個獨立的作業，每個作業都遍歷參數網格，並將相應的參數傳遞給 `getStrategyReturn` 函數。
```python
results1 = Parallel(n_jobs=-1)(delayed(getStrategyReturn)(df1, params['M_RSI_days'], params['a'], params['N'], params['M_RSI_days'], params['rsi_lower'], params['rsi_upper'], params['stop_loss_rate']) for params in tqdm(grid))
results2 = Parallel(n_jobs=-1)(delayed(getStrategyReturn)(df2, params['M_RSI_days'], params['a'], params['N'], params['M_RSI_days'], params['rsi_lower'], params['rsi_upper'], params['stop_loss_rate']) for params in tqdm(grid))
results3 = Parallel(n_jobs=-1)(delayed(getStrategyReturn)(df3, params['M_RSI_days'], params['a'], params['N'], params['M_RSI_days'], params['rsi_lower'], params['rsi_upper'], params['stop_loss_rate']) for params in tqdm(grid))
```
8. **將計算結果整理成 `DataFrame`**:
```python
results1_df = pd.DataFrame(grid)
results1_df['StrategyReturn'] = results1
results1_df['BandHReturn1'] = BandHReturn1
results2_df = pd.DataFrame(grid)
results2_df['StrategyReturn'] = results2
results2_df['BandHReturn2'] = BandHReturn2
results3_df = pd.DataFrame(grid)
results3_df['StrategyReturn'] = results3
results3_df['BandHReturn3'] = BandHReturn3
```
9. **對績效進行排序和調整欄位名稱和順序**: 在這一步，我們按照策略回報（StrategyReturn）進行排序，並調整欄位名稱名稱和順序。
```python
# 按照績效排序
results1_df = results1_df.sort_values(by='StrategyReturn', ascending=False)
results2_df = results2_df.sort_values(by='StrategyReturn', ascending=False)
results3_df = results3_df.sort_values(by='StrategyReturn', ascending=False)
# 拷貝M_RSI_days欄位，放到新創建的RSI_days欄位
results1_df['RSI_days'] = results1_df['M_RSI_days'].copy()
results2_df['RSI_days'] = results2_df['M_RSI_days'].copy()
results3_df['RSI_days'] = results3_df['M_RSI_days'].copy()
# 重新命名M_RSI_days為M
results1_df = results1_df.rename(columns={'M_RSI_days': 'M'})
results2_df = results2_df.rename(columns={'M_RSI_days': 'M'})
results3_df = results3_df.rename(columns={'M_RSI_days': 'M'})
# 調整欄位名稱順序
results1_df = results1_df.reindex(columns=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate', 'StrategyReturn', 'BandHReturn1'])
results2_df = results2_df.reindex(columns=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate', 'StrategyReturn', 'BandHReturn2'])
results3_df = results3_df.reindex(columns=['M', 'a', 'N', 'RSI_days', 'rsi_lower', 'rsi_upper', 'stop_loss_rate', 'StrategyReturn', 'BandHReturn3'])
```
10. **保存20萬種參數組合績效結果**
```python
# 存檔20萬種參數組合，在三個樣本中的績效
results1_df.to_csv('strategy_result20230523_jb/Sample1.csv', index=False)
results2_df.to_csv('strategy_result20230523_jb/Sample2.csv', index=False)
results3_df.to_csv('strategy_result20230523_jb/Sample3.csv', index=False)
# 取出各樣本績效前10名
top_10_df1 = results1_df.head(10)
top_10_df2 = results2_df.head(10)
top_10_df3 = results3_df.head(10)
# 印出各樣本績效前10名
print('樣本一報酬最高前10:\n', top_10_df1)
print('樣本二報酬最高前10:\n', top_10_df2)
print('樣本三報酬最高前10:\n', top_10_df3)
```

#### Cross-Intersection 交叉交集
請先執行完計算績效程式: `opt-numpy_jb.py`再執行此程式。
此程式為交叉交集: `only-intersect.py`。
利用Cross-Intersection找出符合三個樣本通用的參數組合。
1. **讀取績效結果 CSV 檔案:**
```python
results1_df = pd.read_csv('strategy_result20230523_jb/Sample1.csv')
results2_df = pd.read_csv('strategy_result20230523_jb/Sample2.csv')
results3_df = pd.read_csv('strategy_result20230523_jb/Sample3.csv')
```
2. **獲取每個Sample `DataFrame` 的行數：** 計算每個樣本的結果數據框中的行數，以便確定後續交集計算的範圍
```python
num_rows1 = results1_df.shape[0]
num_rows2 = results2_df.shape[0]
num_rows3 = results3_df.shape[0]
```
3. **計算行數的最小值：** 找到三個樣本結果`DataFrame`中行數的最小值，以確保後續運算不會超出該樣本的行數。
```python
min_rows = min(num_rows1, num_rows2, num_rows3)
```
4. **初始化參數和 intersection_params DataFrame：** 用於存儲交集結果。
```python
# num_params為已經找到交集的參數組合的數量
num_params = 0
# num_top_params為一開始要從多少筆資料開始交集
num_top_params = 100
# max_params為要尋找的多少組最大參數組和的數量
max_params = 100
# 使用行數的最小值作為 num_top_params 的最大值
max_top_params = min_rows
```
5. **使用迴圈選取參數組合並進行交集計算:**
```python
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
```
6. **打印找到的共同參數組合：**
```python
# 印出找到的參數組
print("找到的參數組如下：")
print(intersection_params)
```
7. **重新命名欄位並且整理成common_params_list並儲存:**
```python
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
```
#### 改良過後找到更好的參數
我們嘗試了 235,200 種參數組合，並根據策略報酬對這些參數組合進行排名，並找到三個樣本中綜合表現最好的參數。我們找到最好的參數組合：
- M = 10
- a = 1.2
- N = 4 or 5
- RSI_days = 10
- rsi_lower = 0.15
- rsi_upper = 0.80
- stop_loss_rate = 0.2 or 0.25

此策略報酬:

|           | 策略報酬(元) | B&H報酬(元) |
|-----------|-------------|-------------|
| 樣本一     | 7,430       | 3,236       |
| 樣本二     | 6,326       | 614         |
| 樣本三     | 10,151      | 4,075       |

|           | 總交易次數 | 賺錢交易次數 | 虧錢交易次數 | 勝率   | 單次交易最大獲利 | 單次交易最大損失 | 獲利交易中的平均獲利 | 損失交易中的平均損失 | 賺賠比 | 最長的連續性獲利的次數 | 最長的連續性損失的次數 |
|-----------|-------------|---------------|---------------|--------|-----------------|-----------------|----------------------|---------------------|--------|----------------------|----------------------|
| 樣本一     | 3           | 3             | 0             | 100%   | 2976            | 0               | 2476.666667          | 算不出來             | 算不出來 | 3                    | 0                    |
| 樣本二     | 4           | 3             | 1             | 75%    | 4001.15         | -869.96         | 2398.753333          | -869.96            | 2.76    | 3                    | 1                    |
| 樣本三     | 2           | 2             | 0             | 100%   | 7048.26         | 0               | 5075.42              | 算不出來             | 算不出來 | 2                    | 0                    |


------------
請參考上述程式碼和說明，以了解模組三的詳細內容和運作原理。

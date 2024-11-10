import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import r2_score
import pytorch_lightning as pl
from glob import glob
from natsort import natsorted
import mplfinance as mpf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.loggers import CSVLogger
import joblib
import ta
from datetime import datetime , timedelta
from api.config import params
import argparse
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import datetime


#グローバル変数としてハイパーパラメータを設定
input_size = params['input_size']
batch_size = params['batch_size']
hidden_size = params['hidden_size']
learning_rate = params['learning_rate']
window_size = params['window_size']
layers_size =  params['layers_size']
weight_decay_size = params['weight_decay_size']
dropout_size = params['dropout_size']


#OHLCの学習モデル作成
def main(select_button):
   warnings.filterwarnings('ignore') # 警告非表示
   
   #保存するディレクトリの指定
   project_root = os.path.dirname(os.path.abspath(__file__))
   save_dir = os.path.join(project_root, '..', 'models_file')  # 一つ上のディレクトリにある 'data' を参照
   csv_dir = os.path.join(project_root, '..', 'data')  # 一つ上のディレクトリにある 'data' を参照
   
   if select_button == '30min':
      interval = '30m'

   elif select_button == '60min':
      interval = '1h'

   else:
      interval = '5m'
   
   #ドル/円の為替指定   
   ticker = yf.Ticker("USDJPY=X")
   
   #OHLCデータを取得
   if interval == '30m':
      OHLC_data = ticker.history(interval=interval, period='1mo')
   elif interval == '1h':
      end_date = datetime.datetime.now()
      start_date = end_date - datetime.timedelta(days=60)  # 60日間（約2ヶ月）
      OHLC_data = ticker.history(start=start_date, end=end_date, interval=interval)
   else:
      OHLC_data = ticker.history(interval=interval, period='5d')
      
   # DataFrameに変換してインデックスを最初の列に設定
   df = OHLC_data.reset_index()  # インデックスをリセットして列に変換
   df.set_index(df.columns[0], inplace=True)  # 最初の列をインデックスに設定
   
   print(df)
   
   #不要列の削除
   df = df.drop(['Volume','Dividends' ,'Stock Splits'] ,axis=1)

   split_index = int(0.8 * len(df))
   # 最初の80%
   df_train = df[:split_index]
   # 残りの20%
   df_test  = df[split_index:] 
   
   #パスの指定
   csv_save_path_train = os.path.join(csv_dir, 'usd_jpy.csv')   
   csv_save_path_test= os.path.join(csv_dir, 'usd_jpy_test.csv')   
   
   #csvで保存
   df_train.to_csv(csv_save_path_train)
   df_test.to_csv(csv_save_path_test)
   
   #特徴量を追加
   df_candle = feature_engineering(df)
   
   #インスタンス化
   scaler = MinMaxScaler()
   #データフレームの値を正規化
   candle_norm_data = scaler.fit_transform(df_candle)
   # 保存するファイルパスを指定
   scaler_save_path = os.path.join(save_dir, 'scaler.pkl')
   joblib.dump(scaler, scaler_save_path)
   
   #正規化後再度dfに戻す
   df_candle_csv = pd.DataFrame(candle_norm_data , columns = df_candle.columns)
   
   # 保存するCSVファイルのパスを生成
   csv_save_path_train = os.path.join(csv_dir, 'train_norm.csv')      
   
    # データをCSVファイルに保存する
   df_candle_csv.to_csv(csv_save_path_train,index= False)
   
   OHLC = 4

   #OHLC4つのモデルを作成
   for i in range(OHLC):
      
      train_loader , val_loader ,_ ,_ ,_ ,_ = train_val(i,window_size,batch_size,candle_norm_data)
      
      if i == 0:
         col = 'open'
      elif i == 1:
         col = 'high'
      elif i == 2:
         col = 'low'
      elif i == 3:
         col = 'close'
         
      # 再現性の確保
      pl.seed_everything(0)

      #インスタンス化
      model = LSTMRegressor(input_size=input_size , hidden_size=hidden_size , learning_rate=learning_rate,
                        layers_size = layers_size, weight_decay_size = weight_decay_size , dropout_size = dropout_size)

      #学習
      logger = CSVLogger(save_dir='logs', name='my_exp')
      trainer = pl.Trainer(max_epochs=15, deterministic=True, logger=logger)
      trainer.fit(model, train_loader,val_loader) 

      #モデルを保存
      torch.save(model.state_dict(), os.path.join(save_dir, f'model_{col}.pth'))
      #torch.save(model.state_dict(), f'model_{col}.pth')
      #torch.save(model.state_dict(), f'model.pth')
      #パス内のファイルの最新↑で作成したファイルを表示
      filepaths = natsorted(glob('logs/my_exp/*'))
         
      # metrics.csv(評価指標)の読み込み
      log = pd.read_csv(f'{filepaths[-1]}/metrics.csv')

      # データの確認
      print(log.head())

      log[['train_loss_epoch', 'epoch']].dropna(how='any', axis=0).reset_index()['train_loss_epoch'].plot();
      log[['val_loss', 'epoch']].dropna(how='any', axis=0).reset_index()['val_loss'].plot();
      
   # 凡例を表示
   #plt.legend()
   #plt.show()
   ##plt.plot(t, label='AirPassengers')      
   #plt.plot(now_pred, label='prediction')
   #plt.legend();
   #plt.show();

csv_dir = 'data'  

#特徴量追加
#ローソク足のパターンを取得
def feature_engineering(df):

   #変化率を算出
   df_change = df.pct_change() * 100

   #変化率の列名を変更
   df_change.columns = [f'{col}_change' for col in df_change.columns]

   #読み込んだdfと変化率のdfを結合
   df = pd.concat([df,df_change],axis =1)

   #最初の行の変化率が欠損値となっているためそこを削除する
   df = df.dropna()
   #タイムゾーンを標準時間に変換(utc=True)
   df.index = pd.to_datetime(df.index ,utc=True)
   
   k = 1.5  # 大陰 or 陽線の倍率基準
   m = 0.5  # 小陰 or 陽線の倍率基準
   df_candle = df.copy()

   #ローソク足の大きさをとる
   df_candle['candle_size'] = (df_candle['Close'] - df_candle['Open']).abs()
   df_candle['candle_HL_size'] = (df_candle['High'] - df_candle['Low']).abs()

   #10期間移動平均
   df_candle['avg_candle_size'] = ta.trend.SMAIndicator(df_candle['Close'], window=10).sma_indicator()
   
   #移動平均および移動標準偏差を計算
   rolling_mean = df_candle['avg_candle_size'].rolling(window=10).mean()
   rolling_std = df_candle['avg_candle_size'].rolling(window=10).std()
   
   #移動平均に2倍の標準偏差を加えるもしくは引いた値
   upper_bound = rolling_mean + (2 * rolling_std)
   lower_bound = rolling_mean - (2 * rolling_std)
   
   #移動平均から2標準偏差以上離れている値を外れ値として除外
   df_candle = df_candle[(df_candle['avg_candle_size'] >= lower_bound) & (df_candle['avg_candle_size'] <= upper_bound)]
   
   #上ひげの長さ
   df_candle['upper_wick'] = df_candle['High'] - df_candle[['Close', 'Open']].max(axis=1)
   
   #下ひげの長さ
   df_candle['lower_wick'] = df_candle[['Close', 'Open']].min(axis=1) - df_candle['Low'] 
   
   # 大陽線(10期間の平均より1.5倍のローソク足と定義)
   df_candle['big_bullish'] = ((df_candle['Close'] > df_candle['Open']) & (df_candle['candle_size'] > k * df_candle['avg_candle_size']))
   
   # 小陽線(10期間の平均より0.5倍のローソク足と定義)
   df_candle['small_bullish'] = ((df_candle['Close'] > df_candle['Open']) & (df_candle['candle_size'] > m * df_candle['avg_candle_size']))
   
   # 大陰線(10期間の平均より1.5倍のローソク足と定義)
   df_candle['big_bearish'] =  ((df_candle['Open'] > df_candle['Close']) & (df_candle['candle_size'] > k * df_candle['avg_candle_size']))

   # 小陰線(10期間の平均より0.5倍のローソク足と定義)
   df_candle['small_bearish'] =  ((df_candle['Open'] > df_candle['Close']) & (df_candle['candle_size'] > m * df_candle['avg_candle_size']))
   
   
   # ハンマー（下ヒゲが本体より長く、本体が小さい）
   df_candle['hammer'] = ((df_candle['candle_size'] < df_candle['lower_wick']) & \
      (df_candle['upper_wick'] < df_candle['lower_wick']) & \
         (df_candle['candle_size'] <= df_candle['candle_HL_size']*0.3))

   # ピンバー（上ヒゲまたは下ヒゲが非常に長い）
   df_candle['pin_bar'] = ((df_candle['upper_wick'] > df_candle['candle_size'] * 2) | \
      (df_candle['lower_wick'] > df_candle['candle_size'] * 2) & \
         (df_candle['candle_size'] <= df_candle['candle_HL_size']*0.3))
   
   #カラサカ(上ひげがなく、下ひげが非常に長いかつ本体が小さい)
   df_candle['karasaka'] = ((df_candle['High'] - df_candle[['Close','Open']].max(axis = 1)) <= df_candle['candle_HL_size'] * 0.2) &\
         (df_candle['lower_wick'] > df_candle['candle_size'] * 2) & \
            (df_candle['candle_size'] <= df_candle['candle_HL_size']*0.3)
   
   #トンカチ(下ひげがなく、上ひげが非常に長いかつ本体が小さい)
   df_candle['tonkachi'] = ((df_candle['High'] - df_candle[['Close','Open']].max(axis = 1)) <= df_candle['candle_HL_size'] * 0.2) &\
         (df_candle['upper_wick'] > df_candle['candle_size'] * 2) & \
            (df_candle['candle_size'] <= df_candle['candle_HL_size']*0.3)
            
   #ピアス
   #前日が陰線
   #当日の始値が前日の安値より低い
   ## 終値が前日の陰線の半分以上
   #当日が陽線
   df_candle['piercing'] = (df_candle['Close'].shift(1) < df_candle['Open'].shift(1)) & \
      (df_candle['Open'] < df_candle['Low'].shift(1)) &  \
         (df_candle['Close'] > (df_candle['Close'].shift(1) + df_candle['Open'].shift(1)) / 2) & \
            (df_candle['Close'] > df_candle['Open'])
   #含み足
   df_candle['inside_bar'] = (df_candle['High'] < df_candle['High'].shift(1)) & \
      (df_candle['Low'] > df_candle['Low'].shift(1))
   
   #包み足
   df_candle['outside_bar'] = (df_candle['High'] > df_candle['High'].shift(1)) & \
      (df_candle['Low'] < df_candle['Low'].shift(1))
      
   # 赤三兵（Three White Soldiers）
   df_candle['three_white'] = (df_candle['Close'] > df_candle['Open']) & \
      (df_candle['Close'].shift(1) > df_candle['Open'].shift(1)) & \
         (df_candle['Close'].shift(2) > df_candle['Open'].shift(2)) & \
            (df_candle['Close'] > df_candle['Close'].shift(1)) & \
               (df_candle['Close'].shift(1) > df_candle['Close'].shift(2))

# 黒三兵（Three Black Crows）
   df_candle['three_black'] = (df_candle['Close'] < df_candle['Open']) & \
      (df_candle['Close'].shift(1) < df_candle['Open'].shift(1)) & \
         (df_candle['Close'].shift(2) < df_candle['Open'].shift(2)) & \
            (df_candle['Close'] < df_candle['Close'].shift(1)) & \
               (df_candle['Close'].shift(1) < df_candle['Close'].shift(2))
               
   # モーニングスター（Morning Star)
   #二つ前が陽線
   #1つ前が小陽線 or 小陽線
   #現在が陽線
   #終値の1つ前が現在の終値より高い
   #終値が2つ前の始値を超える
   df_candle['morning_star'] = (df_candle['Close'].shift(2) < df_candle['Open'].shift(2)) & \
      (df_candle['small_bullish'].shift(1) | df_candle['small_bearish'].shift(1)) & \
         (df_candle['Close'] > df_candle['Open']) & \
            (df_candle['Close'] > df_candle['Close'].shift(1)) & \
               (df_candle['Close'] > df_candle['Open'].shift(2)) 
               
   # イブニングスター（Evening Star)
   #二つ前が陰線
   #1つ前が小陽線 or 小陽線
   #現在が陰線
   #終値の1妻絵が現在の終値より低い
   #終値が2つ前の始値を下回る
   df_candle['evening_star'] = (df_candle['Close'].shift(2) > df_candle['Open'].shift(2)) & \
      (df_candle['small_bullish'].shift(1) | df_candle['small_bearish'].shift(1)) & \
         (df_candle['Close'] < df_candle['Open']) & \
            (df_candle['Close']< df_candle['Close'].shift(1)) & \
               (df_candle['Close'] < df_candle['Open'].shift(2)) 
   
   #パターンの列だけのdfにする
   #不要な列を削除する
   df_feature = df_candle.iloc[:,-11:]
   #Boolean(True False)を1 0 に変換
   df_feature = df_feature.astype(int) 
   df_feature = pd.concat([df,df_feature],axis =1)
   #df_candle = df_candle.dropna()
   
   #40期間移動平均(中期間)
   df_feature['MA_40'] = ta.trend.SMAIndicator(df_candle['Close'], window=40).sma_indicator()
   
   #100期間移動平均(長期間)
   df_feature['MA_100'] = ta.trend.SMAIndicator(df_candle['Close'], window=100).sma_indicator()
   
   
   
   #ポリジャーバンド
   #upper(買われすぎ)およびlower(売られすぎ)を取得する
   #BBANDSは3つの値を返されるため、middle_bandはダミーを入れておく
   bollinger = ta.volatility.BollingerBands(close=df_feature['Close'], window=10, window_dev=2)
   df_feature['upper_band'] = bollinger.bollinger_hband()
   df_feature['lower_band'] = bollinger.bollinger_lband()
   
   bollinger_mid = ta.volatility.BollingerBands(close=df_feature['Close'], window=20, window_dev=2)
   df_feature['upper_band_mid'] = bollinger_mid.bollinger_hband()
   df_feature['lower_band_mid'] = bollinger_mid.bollinger_lband()
   
   bollinger_long = ta.volatility.BollingerBands(close=df_feature['Close'], window=10, window_dev=2)
   df_feature['upper_band_long'] = bollinger_long.bollinger_hband()
   df_feature['lower_band_long'] = bollinger_long.bollinger_lband()
   
   #ポリジャーハンドと同じように買われすぎ、売られすぎを取得する指標
   df_feature['RSI'] = ta.momentum.RSIIndicator(close=df_feature['Close'], window=10).rsi()
   df_feature['RSI_mid'] = ta.momentum.RSIIndicator(close=df_feature['Close'], window=20).rsi()
   df_feature['RSI_long'] = ta.momentum.RSIIndicator(close=df_feature['Close'], window=50).rsi()
   
   # MACDの計算
   macd = ta.trend.MACD(df_feature['Close'], window_slow=26, window_fast=12, window_sign=9)
   df_feature['MACD'] = macd.macd()
   df_feature['MACD_signal'] = macd.macd_signal()
   df_feature['MACD_hist'] = macd.macd_diff()
   
   macd = ta.trend.MACD(df_feature['Close'], window_slow=50, window_fast=21, window_sign=9)
   df_feature['MACD_mid'] = macd.macd()
   df_feature['MACD_signal_mid'] = macd.macd_signal()
   df_feature['MACD_hist_mid'] = macd.macd_diff()
   
   macd = ta.trend.MACD(df_feature['Close'], window_slow=100, window_fast=40, window_sign=9)
   df_feature['MACD_long'] = macd.macd()
   df_feature['MACD_signal_long'] = macd.macd_signal()
   df_feature['MACD_hist_long'] = macd.macd_diff()
   
   
   #欠損値の計算をしてしまっているdataを取り除く
   df_avg = df_candle['avg_candle_size']
   df_feature = pd.concat([df_feature,df_avg],axis =1)
   df_feature = df_feature.dropna()
   
   #欠損値を削除後再度不要な列を削除
   df_candle = df_candle.drop('avg_candle_size' ,axis= 1)
   print(df_feature)

   return df_feature

#学習用データを作成する
def train_val(OHLC_col,window_size,batch_size,candle_norm_data):
   window = window_size

   #入力データと教師データ用のからのlistを定義
   x,t = [],[]

   #dataframe全部からwindowを引いた分をループさせる

   for i in range(len(candle_norm_data)-window):
      
      #各要素にをwindowずつのデータを格納する
      x.append(candle_norm_data[i:i+window])

      #各要素をwindow+1のデータを格納する
      #windowサイズの次のデータを予測させるための値
      t.append(candle_norm_data[i+window,OHLC_col])
      
   #list型をndarray型に変換する
   x = np.array(x)
   t = np.array(t)
   
   
   print(x.shape, t.shape)

   #入力および教師データの1つ目の要素の確認
   #print(f'{x[0] ,t[0]}') 
   
   #サイズの確認
   #データ全体
   #入力変数(特徴量)の数
      
   #LSTMで使えるようにするためにndarray型をtensor型に変換
   x = torch.tensor(x, dtype=torch.float32)
   #xとtの形状を合わせる
   t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
   
   #OHLCで出力したいため、ターゲットの次元数を最初の4つにする
   #t = t[:,:4] 
   
   
   #=======================================================================
   #お試し
   #=======================================================================
   #LSTM(input_size,hidden_size ,first =True)
   #input_size：特徴量の数
   #hidden_size：隠れ層の数(大きくすれば記憶・処理する能力が上がるが、計算コストも上がる
   #bacth_first： [バッチサイズ, シーケンスの長さ, 特徴量の数] の形に整えるためのこれをする
   #↑をする理由として、[シーケンスの長さ,バッチサイズ, 特徴量の数]の形となってしまい、次元を間違えたりなんども変換しなくてよくなるため
   #lstm = nn.LSTM(input_size, 24, batch_first=True)

   #out：batch_size,シーケンス長さ,隠れ層の数
   #h：num_layers(lstmの層の数)*num_direction(単方向 or 双方向) ,batch_size, 隠れ層の数→短期記憶(シーケンス全体の要約)
   #c：num_layers(lstmの層の数)*num_direction(単方向 or 双方向) ,batch_size, 隠れ層の数→長期記憶(そのシーケンス内での重要なことを覚えておくこと)
   #単方向：過去から未来を予測したい場合に使用
   #双方向：文脈を理解するような場合に使用
   #out, (h, c) = lstm(x)
   #print(out.shape, h.shape, c.shape)   
      
   # 各データセットのサンプル数を決定
   n_train = int(len(x) * 0.8)

   #頭から8割までのデータを学習用データとする
   #残り2割を検証用データとする
   x_train, t_train = x[:n_train], t[:n_train]
   x_val, t_val = x[n_train:], t[n_train:]
   
   # ひとつのオブジェクトにまとめる
   train = torch.utils.data.TensorDataset(x_train, t_train)
   val = torch.utils.data.TensorDataset(x_val, t_val)

   print(len(train) , len(val))
   # ランダムに分割を行うため、シードを固定して再現性を確保
   pl.seed_everything(0)

   # バッチサイズの定義
   batch = batch_size

   #Dataloaderを準備
   train_loader = torch.utils.data.DataLoader(train, batch , shuffle = True, drop_last=True)
   val_loader = torch.utils.data.DataLoader(val, batch)  
   
   return train_loader,val_loader,x_train,x_val,t_train,t_val

#PyTorch Lightningを使ってモデルを実装
class LSTMRegressor(pl.LightningModule):

   def __init__(self,input_size, hidden_size, learning_rate, layers_size , weight_decay_size , dropout_size):
      super().__init__()

      #バッチ正規化
      self.batch_norm = nn.BatchNorm1d(input_size)
 
      #lstm層の定義
      self.lstm = nn.LSTM(input_size, hidden_size, layers_size , bidirectional=False , batch_first=True)
    
      #ドロップアウト層の定義
      #過学習を防ぐため
      self.dropout = nn.Dropout(dropout_size)
      
      #全結合層の定義
      #最終出力4(Close)
      self.fc = nn.Linear(hidden_size, 1)
      self.learning_rate = learning_rate
      self.weight_decay = weight_decay_size
   

   def forward(self, x):
      #(batch_size,input_dim,seq_len)の形に変更
      if x.dim() == 2:  # SHAPから渡された2次元テンソルの場合
        x = x.unsqueeze(1)  # (samples, 1, features)
      
      x = x.permute(0, 2, 1)

      x = self.batch_norm(x)
      
      # x の shapeを(batch_size, seq_len, input_dim)の形に変更 
      x = x.permute(0, 2, 1)
   
      #lstm層の処理
      out, (h, c) = self.lstm(x)
      # h の shape : (1, batch_size, hidden_size) 
      # 全結合層に入れるために (batch_size, hidden_size) に変更する
      #h = h.view(h.size(1), -1)
      #最後の隠れ層の状態を取得する
      h = h[-1]
      #h = self.batch_norm(h)
      h = self.dropout(h)
      h = self.fc(h)
      return h


   # 学習データに対する処理
   def training_step(self, batch, batch_idx):
      x, t = batch
      y = self(x)
      loss = F.smooth_l1_loss(y, t)
      self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
      r2= r2_score(y, t)
   
      self.log('train_r2', r2, on_step=True, on_epoch=True, prog_bar=True)
      return loss


   # 検証データに対する処理
   def validation_step(self, batch, batch_idx):
      x, t = batch
      y = self(x)
      loss = F.smooth_l1_loss(y, t)
      self.log('val_loss', loss, on_step=False, on_epoch=True)
      if t.size(0) > 1:
         r2= r2_score(y, t)
         self.log('val_r2', r2, on_step=False, on_epoch=True , prog_bar=True)
      return loss


# 最適化手法
   def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters() , lr=self.learning_rate , weight_decay = self.weight_decay)
      return optimizer  
   
#main('30min')


#XGBでの予測
#アンサンブル学習が必要な場合に有効化
def XGB_Regression():
   
   csv_load_path = os.path.join(csv_dir, 'usd_jpy.csv')
   df = pd.read_csv(csv_load_path, index_col='Datetime', parse_dates=True)
   #不要な列を削除
   df = df.drop(['Adj Close' , 'Volume'] , axis= 1)
   
   df_candle = feature_engineering(df)
   
   #インスタンス化
   scaler = MinMaxScaler()
   #データフレームの値を正規化
   candle_norm_data = scaler.fit_transform(df_candle)
   
   _ , _ , x_train , x_val , t_train , t_val = train_val(3,window_size,batch_size,candle_norm_data)
   
   #XGboostによる予測
   model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=100,
    subsample=1
)
   
   
   # 3次元データ (samples, time_steps, features) を2次元にフラット化
   xgb_x_train_flat = x_train.reshape(x_train.shape[0], -1)
   xgb_t_train = t_train.reshape(-1)

   # 同じようにxgb_x_valもフラット化
   xgb_x_val_flat = x_val.reshape(x_val.shape[0], -1)

   # フラット化したデータを使ってXGBoostのトレーニング
   model.fit(xgb_x_train_flat, xgb_t_train)

   # 検証データで予測を行う
   t_pred = model.predict(xgb_x_val_flat)
   
   t_pred = t_pred.reshape(-1,1)
   t_pred = np.column_stack([t_pred, np.full((t_pred.shape[0], 39), np.nan)])
   t_val = np.column_stack([t_val, np.full((t_val.shape[0], 39), np.nan)])
   #逆正規
   t_pred = scaler.inverse_transform(t_pred)
   t_val = scaler.inverse_transform(t_val)
   print(t_pred)
   
   t_pred = t_pred[:, 0]
   t_val = t_val[:, 0]
   
   
   # 評価 (MAE)
   mae = mean_absolute_error(t_val, t_pred)
   print(f"Mean Absolute Error: {mae}")
   
   # 時系列データとしてプロットするために、実際の時間の範囲を取得
   time_steps = np.arange(len(t_val))

   # 実データのプロット
   actual_trace = go.Scatter(
      x=time_steps,
      y=t_val,
      mode='lines',
      name='Actual',
      line=dict(color='blue')
   )

   # 予測データのプロット
   predicted_trace = go.Scatter(
      x=time_steps,
      y=t_pred,
      mode='lines',
      name='Predicted',
      line=dict(color='red', dash='dash')
   )

   # グラフのレイアウト
   layout = go.Layout(
      title='Actual vs Predicted Time Series',
      xaxis=dict(title='Time Steps'),
      yaxis=dict(title='Values'),
      legend=dict(x=0, y=1)
   )

   # 図の作成
   fig = go.Figure(data=[actual_trace, predicted_trace], layout=layout)

   # グラフを表示
   fig.show()

#XGB_Regression()
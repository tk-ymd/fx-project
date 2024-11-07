import os
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from typing import Union
from natsort import natsorted
from glob import glob
import mplfinance as mpf
from models.model import LSTMRegressor
from models.model import feature_engineering
from api.config import params
from collections import defaultdict

def run_prediction(select_button):
    
    #モデル/scalerをloadするフォルダ
    load_dir = 'models_file'
    
    #csvを保存するフォルダ
    csv_dir = 'data'
    #テストデータの読み込み
    csv_load_path = os.path.join(csv_dir, 'usd_jpy_test.csv')
    df_test = pd.read_csv(csv_load_path, index_col='Datetime', parse_dates=True)
    print(df_test)
    #df_test = pd.read_csv('usd_jpy_1h＿test.csv',index_col='Datetime', parse_dates=True)
    #df_test = df_test.drop(['Adj Close' , 'Volume'] , axis= 1)

    #変化率およびローソク足の取得
    df_test = feature_engineering(df_test)
    
    print(df_test)
    
    selected_button = select_button
    model = LSTMRegressor(
        
        input_size = params['input_size'],
        hidden_size = params['hidden_size'],
        learning_rate = params['learning_rate'],
        layers_size = params['layers_size'],
        weight_decay_size = params['weight_decay_size'],
        dropout_size = params['dropout_size']
        
        )

    def load_and_pred(df_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        
        #それぞれのモデルを格納する空の配列を定義
        all_dic = defaultdict(list)
        pred_dic = {}
        #time_step = 24
        OHLC = 4
        
        #スケーラーのロード
        scaler_load_path = os.path.join(os.path.join(load_dir, 'scaler.pkl'))
        #scaler_load_path = os.path.join(load_dir, 'scaler.pkl')
        print(f"Loading scaler from: {scaler_load_path}")
        scaler = joblib.load(scaler_load_path)
        #正規化
        norm_data = scaler.transform(df_test)
        
        
        pred_arr = []
        pred_dic = {}
        #OHLC(4回)ループ
        for n in range(OHLC):
            #配列とdictionaryを都度初期化
            #ループ事の変数を定義
            if n == 0:
                col = 'open'
                list_name = 'pred_open'
                
            elif n == 1:
                col = 'high'
                list_name = 'pred_high'
                
            elif n == 2:
                col = 'low'
                list_name = 'pred_low'
                
            elif n == 3:
                col = 'close'
                list_name = 'pred_close'
                
            model_path = os.path.join(load_dir, f'model_{col}.pth')
            model.load_state_dict(torch.load(model_path))
            model.eval()  # モデルを推論モードに設定
            model.cpu() # CPU での推論
            
            #input_seq = torch.tensor(norm_data.params['window_size'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                
                #最後のシーケンスを取り出す
                input_tensor = torch.tensor(
                    norm_data[-params['window_size']:,:], dtype=torch.float32
                ).unsqueeze(0)
                
                
                #次の予測結果取得
                pred_data = model(input_tensor)
                
                #次のタイムステップにデータを追加
                
                pred_arr.append(pred_data.item())
                pred = np.array(pred_arr).reshape(-1, 1)

                # 予測データを19列に合わせる（他の25列は NaN で埋める）
                #pred_full = np.column_stack([pred, np.full((pred.shape[0], 25), np.nan)])  # shape = (6166, 25)
            
                # 逆スケーリングを行う
                #inverse_pred_full = scaler.inverse_transform(pred_full)
                #最初の列だけを取得
                #inverse_pred = inverse_pred_full[:, 0]

                #dictionaryに格納
                pred_dic[list_name] = pred[-1].flatten()
        
        pred_df = pd.DataFrame(pred_dic)
        pred_df.columns = ['Open','High','Low','Close']
        
        #最後の71行を取得 
        last_row = norm_data[-72:, :4]

        # 取得したデータをDataFrameに変換（pred_dfと同じカラム名を使用）
        last_df = pd.DataFrame(last_row, columns=pred_df.columns)
        #特徴量エンジニアリングをするために72行を追加
        pred_df = pd.concat([pred_df, last_df], ignore_index=True)
        print(pred_df.shape)
        pred_df = feature_engineering(pred_df)
        
        pred_data = np.array(pred_df)
        #元々の正規化したデータと予測したデータを結合
        norm_data =np.vstack((norm_data,pred_data))
        print(pred_df.shape)
        for key, value in pred_dic.items():
            all_dic[key].append(value)
                
        #dictionaryをdfに変換
        #pred_df = pd.DataFrame(all_dic)
        #print(pred_df)
        #pred_df.columns = ['Open','High','Low','Close']
        
        #特徴量エンジニアリング
        #pred_df = feature_engineering(pred_df)
        #ndarray型に変更
        #pred_data = np.array(pred_df)
        #元々の正規化したデータと予測したデータを結合
        #norm_data =np.vstack((norm_data,pred_data))
        
        # 逆スケーリングを行う
        #inverse_pred_full = scaler.inverse_transform(pred_df)
        #最初の列だけを取得
        #inverse_pred = inverse_pred_full
        
        #print(inverse_pred)      
        return all_dic

    def result(df_test: Union[pd.DataFrame, np.ndarray]) -> go.Figure:
        
        # 現在の時刻を取得
        now = datetime.now()
        if selected_button == '30min':
            select_button_text = '30分'
            # 取得する結果を判定
            if now.minute < 30:
                # 現在の時刻がXX:00〜XX:29の場合、前の時間（例：13:00の結果）
                result_time = now.replace(minute=30, second=0, microsecond=0)
            else:
                # 現在の時刻がXX:30〜XX:59の場合、次の時間（例：14:00の結果）
                result_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        elif select_button == '60min':
            select_button_text = '1時間'
            #次の時間
            result_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        else:
            select_button_text = '5分'
            minute_adjustment = (5 - (now.minute % 5)) % 5
            result_time = (now + timedelta(minutes=minute_adjustment)).replace(second=0, microsecond=0)
        
        window_size = params['window_size']
        
        # 予測結果のdictionaryを格納
        pred_dic = load_and_pred(df_test)
        
        pred_df = pd.DataFrame(pred_dic)
        # スカラーに戻す
        pred_df = pred_df.explode(['pred_open', 'pred_high', 'pred_low', 'pred_close'])
        pred_df = pred_df.apply(pd.to_numeric, errors='coerce')
        pred_df.columns = ['Open','High','Low','Close'] 
        
        # 逆正規化するためにNaNの列を22列追加
        nan_columns = pd.DataFrame(np.nan, index=pred_df.index, columns=[f'NaN_{i}' for i in range(36)])
        
        # OHLCとNaN22列を結合
        pred_df = pd.concat([pred_df, nan_columns], axis=1)
        
        # スケーラーのロード
        scaler_load_path = os.path.join(os.path.join('models_file', 'scaler.pkl'))
        #scaler_load_path = os.path.join(load_dir, 'scaler.pkl')
        print(f"Loading scaler from: {scaler_load_path}")
        scaler = joblib.load(scaler_load_path)
        
        pred_df = scaler.inverse_transform(pred_df)
        pred_df = pred_df[:, :4]
        pred_df = pd.DataFrame(pred_df , columns=['Open','High','Low','Close'])
        pred_df.index = [result_time]
        print(pred_df)
        # 実際の値を取得
        actual = df_test['Close'].values[window_size:]

        fig = go.Figure(data=[go.Candlestick(
            x=pred_df.index,
            open=pred_df['Open'],
            high=pred_df['High'],
            low=pred_df['Low'],
            close=pred_df['Close'],
            name="Predicted OHLC"
        )])

        # グラフのタイトルやラベルを追加
        fig.update_layout(
            title=f'予測結果ローソク足(※{select_button_text}毎に随時更新)',
            title_font_size=16,
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
            
        )
        # チャートを表示
        #fig.show()
        fig = fig.to_json()
        # HTMLに保存（任意）
        #fig.write_html('candlestick_with_actual.html')
        
        # figオブジェクトを返す
        return fig
        
    prediction_chart = result(df_test)
    return prediction_chart
    
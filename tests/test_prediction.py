import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.metrics import mean_absolute_error
import shap


#モデル/scalerをloadするフォルダ
load_dir = 'models_file'
    
#csvを保存するフォルダ
csv_dir = 'data'

#テストデータの読み込み
csv_load_path = os.path.join(csv_dir, 'usd_jpy_test.csv')
df_test = pd.read_csv(csv_load_path, index_col='Datetime', parse_dates=True)

#変化率およびローソク足の取得
df_test = feature_engineering(df_test)

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
    pred_dic = {}
    
    OHLC = 4
    
    #for n in range(OHLC):
        #配列を都度初期化
    pred_arr = []
    #    if n == 0:
    #    col = 'open'
    #        list_name = 'pred_open'
            
    #    elif n == 1:
    #        col = 'high'
    #        list_name = 'pred_high'
            
    #    elif n == 2:
    #        col = 'low'
    #        list_name = 'pred_low'
            
    #    elif n == 3:
    #        col = 'close'
    #        list_name = 'pred_close'
            
    #モデルのロード
    model_path = os.path.join(load_dir, f'model_close.pth')
    print(f"Loading scaler from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    model.eval()  # モデルを推論モードに設定
    model.cpu() # CPU での推論
    
    #スケーラーのロード
    scaler_load_path = os.path.join(os.path.join(load_dir, 'scaler.pkl'))
    scaler_load_path = os.path.join(load_dir, 'scaler.pkl')
    print(f"Loading scaler from: {scaler_load_path}")
    scaler = joblib.load(scaler_load_path)
    
    norm_data = scaler.transform(df_test)
    #正規化したデータをdfに再変換
    df_test_norm = pd.DataFrame(norm_data, columns=df_test.columns )
    
    print(df_test_norm.shape)
    
    #正規化されたtrainデータの読み込み
    csv_load_path = os.path.join(csv_dir, 'train_norm.csv')
    df_train = pd.read_csv(csv_load_path)
    df_train = pd.DataFrame(df_train, columns=df_test.columns )
    
    x_train_tensor = torch.tensor(df_train.values, dtype=torch.float32)
    x_train_tensor = x_train_tensor.view(x_train_tensor.size(0), -1)
    
    
    
    # テストデータをTensorに変換
    
    x_test_tensor = torch.tensor(df_test_norm.values, dtype=torch.float32)
    x_test_tensor = x_test_tensor.view(x_test_tensor.size(0), -1)
    

    print("X_train_tensor shape:", x_train_tensor.shape)
    print("X_test_tensor shape:", x_test_tensor.shape)
    print("Feature names:", df_test.columns)
    
    
    # SHAPのDeepExplainerを使用して特徴量の重要度を確認
    explainer = shap.DeepExplainer(model, x_train_tensor)

    # テストデータに対するSHAP値の計算
    shap_values = explainer.shap_values(x_test_tensor,check_additivity=False)
    
    print("shap_values shape:", shap_values.shape)
    
    shap_values = np.array(shap_values).squeeze(axis=2)

    # 特徴量の影響度をプロット
    shap.summary_plot(shap_values, df_test_norm, feature_names=df_test.columns.tolist() , max_display=40)
    
    with torch.no_grad():
        for i in range(len(norm_data) - params['window_size']):
            
            input_tensor = torch.tensor(
                norm_data[i : i + params['window_size']], dtype=torch.float32
            ).unsqueeze(0)
            
            pred_data = model(input_tensor)
            
            pred_arr.append(pred_data.item())
            pred = np.array(pred_arr).reshape(-1, 1)
            # 予測データの列を合わせる
            pred_full = np.column_stack([pred, np.full((pred.shape[0], 39), np.nan)])  # shape = (6166, 25)
        
            # 逆スケーリングを行う
            inverse_pred_full = scaler.inverse_transform(pred_full)
            #最初の列だけを取得
            inverse_pred = inverse_pred_full[:, 0]
        
            #pred_dic[list_name] = inverse_pred 
            pred_dic = inverse_pred 
    #for key, value in pred_dic.items():
    #    print(f"{key}: {len(value)}")
    
    #return pred_dic
    return inverse_pred

def result(df_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

    window_size = params['window_size']
    
    #予測結果のdictionaryを格納
    #pred_dic = load_and_pred(df_test)
    #pred_df = pd.DataFrame(pred_dic)
    pred = load_and_pred(df_test)
    #pred =pred_df['pred_close']
    
    #windowsize分遅延
    
    #pred = np.roll(pred,-5)
    
    #実際の値を取得
    actual = df_test['Close'].values[window_size:]
    
    #平均絶対誤差
    mse = mean_absolute_error(pred , actual)
    
    y_true_direction = np.sign(np.diff(actual))  # 実際の値の変化方向
    y_pred_direction = np.sign(np.diff(pred))  # 予測された値の変化方向
    
    correct_direction = y_true_direction == y_pred_direction
    # Hit Rateの計算
    hit_rate = np.mean(correct_direction)   
    
    print(mse)
    print(hit_rate)
    
    # 結果の可視化（plotlyを使用）
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode="lines", name="Actual Close"))
    fig.add_trace(go.Scatter(y=pred, mode="lines", name="Predicted Close"))
    
    fig.write_html('actual.html')
    #graph_html = fig.to_html(full_html=False)
    
    return actual
        
        


graph_html = result(df_test)

    #return HTMLResponse(content=graph_html, status_code=200) 
    
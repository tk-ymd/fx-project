from fastapi import APIRouter
from alpha_vantage.foreignexchange import ForeignExchange
import threading
import time
from models.model import main
from utils.prediction import run_prediction
from pydantic import BaseModel
import requests
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# ルーターを作成
router = APIRouter()
# 為替レートのグローバル変数
exchange_rate = {"rate": None}  
#ローソク足のグローバル変数
exchange_chart = {"chart": None}  

# バックグラウンドで動作させる関数：為替レートの更新
def update_exchange_rate():
    global exchange_rate
    while True:
        # Yahoo FinanceからUSD/JPYの為替レートを取得
        ticker = yf.Ticker("USDJPY=X")
        data = ticker.history(period="1d")
        if not data.empty:
            # 最新の為替レートを取得
            exchange_rate = exchange_rate = data['Close'].iloc[-1]
            print(f"Current USD/JPY Exchange Rate: {exchange_rate}")
        else:
            print("Failed to fetch data")
        
        # 60秒ごとに更新
        time.sleep(60)
        
        
def init_chart():
    global exchange_chart
    # 過去1日の30分足データを取得
    ticker = yf.Ticker("USDJPY=X")
    data = ticker.history(interval="30m", period="1d")  # 1日分の30分足データ
    
    if not data.empty:
        # Plotly用のチャートデータを準備
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,  # 日付と時間
                open=data['Open'], 
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )
        ])
        
        # チャートのレイアウト設定
        fig.update_layout(
            title='USD/JPY 分足ローソク足チャート',
            xaxis_title='Date',
            yaxis_title='Price (JPY)',
            xaxis_rangeslider_visible=False
        )
        
        exchange_chart = fig.to_json()
        print(exchange_rate)
    else:
        print("Failed to fetch data")
        


def updata_chart(select_button):
    
    if select_button == '30min':
        interval = '30m'
        period = '1d'
        text = '30分足'
    elif select_button == '60min':
        
        interval = '60m'
        period = '1d'
        text = '1時間足'
    else:
        interval = '5m'
        period = '1d'
        text = '5分足'
    print(f'interval:{interval}')

    # 過去1日のローソク足を取得
    ticker = yf.Ticker("USDJPY=X")
    data = ticker.history(interval=interval, period=period)  
    
    if not data.empty:
        # Plotly用のチャートデータを準備
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,  # 日付と時間
                open=data['Open'], 
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )
        ])
        
        # チャートのレイアウト設定
        fig.update_layout(
            title=f'USD/JPY{text}ローソク足チャート',
            xaxis_title='Date',
            yaxis_title='Price (JPY)',
            xaxis_rangeslider_visible=False
        )
        
        #chart = fig.to_json()
        
        return fig
    else:
        print("Failed to fetch data")

#為替レートを返すエンドポイント
@router.get("/rate")
def get_rate():
    return {"USD/JPY rate": exchange_rate}

#ローソク足を返すエンドポイント
@router.get("/chart")
def get_chart():
    return {"USD/JPY rate": exchange_chart}

#ボタンに基づく情報を返すエンドポイント
@router.get("/button_info")
async def get_chart(selected_button: str):
    try:
        chart = updata_chart(selected_button)
        chart = chart.to_json()
        return {f"USD/JPY_{selected_button} chart": chart}
    except Exception as e:
        print(f"Error in /button_info: {e}")
        return {"error": str(e)}
# データモデルを定義
class SelectButton(BaseModel):
    selected_button: str

# 予測モデルを呼び出すエンドポイント
@router.post("/prediction")
async def get_chart(button_info: SelectButton):
    main(button_info.selected_button)
    fig = run_prediction(button_info.selected_button)
    return {"USD/JPY chart": fig}
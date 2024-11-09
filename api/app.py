import os
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import time
import json
from copy import deepcopy


#start_background_tasks()

# FastAPI サーバーのURL
# Renderの環境変数からAPI_URLを取得
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/")

st.title('FX予測')

# 為替レートを取得する
def get_exchange_rate():
    try:
        response = requests.get(f"{API_URL}rate")
        if response.status_code == 200:
            data = response.json()
            return data.get("USD/JPY rate", "データがありません")
        else:
            return "エラー: データ取得に失敗しました"
    except Exception as e:
        return f"エラー: {e}"
    
#為替チャートを取得する 
def get_exchange_chart():
    try:
        response = requests.get(f"{API_URL}chart")
        if response.status_code == 200:
            data = response.json()
            return data.get("USD/JPY rate", "データがありません")
        else:
            return "エラー: データ取得に失敗しました"
    except Exception as e:
        return f"エラー: {e}"
        
col1,col2 = st.columns(2)

#最新レートを表示用
with col1:
    #コンテナの作成
    real_rate = st.empty()

# CSSを使用してst.buttonのスタイルをカスタマイズ
st.markdown("""
    <style>
    
    .rate-card {
                display: flex;
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .button-container {
        display: flex;
        justify-content: flex-start; /* 左寄せ */
        gap: 20px;
        margin-top: 20px;
    }
    
    div.stButton > button {
        background-color: transparent; /* 背景を透明に */
        color: orange; /* テキストの色をオレンジに */
        border: 2px solid orange; /* 枠線をオレンジに */
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        color: orange; /* テキストの色をオレンジに */
        border: 2px solid orange; /* 枠線をオレンジに */
        background-color: #fffacd; /* ホバー時に背景色を変更 */
    }
    </style>
    
    """, unsafe_allow_html=True)

col1, col2 ,col3 = st.columns([1,1,4])


real_chart = st.empty()

# 初期状態の設定
if 'initialized' not in st.session_state:
    # 初回起動時だけ30分足を設定
    st.session_state["selected_button"] = '30min'
    st.session_state['initialized'] = True  # 初期化済みフラグ
    print('初期化')

if "previous_button" not in st.session_state:
    st.session_state.previous_button = None  # 初期状態としてNone
    
if 'fig' not in st.session_state or st.session_state.fig is None:
    st.session_state.fig = go.Figure()  


# ボタンが押された時に対応する処理を実行
def handle_button_click(button_name):
    print(st.session_state['selected_button'])
     # `previous_button` を現在の `selected_button` に更新する
    st.session_state['previous_button'] = st.session_state['selected_button']
    # `selected_button` を新しいボタンの名前に更新する
    st.session_state['selected_button'] = button_name
    print(st.session_state['selected_button'])
    print(st.session_state['previous_button'])
    
# 5分足
with col1:
    if st.button("5分足"):
        handle_button_click('5min')
# 30分足
with col2:
    if st.button("30分足"):
        handle_button_click('30min')
        
#1時間足
with col3:
    if st.button("1時間足"):
        handle_button_click('60min')


previous_button = st.session_state['previous_button']
select_button = st.session_state['selected_button']

# ローソク足予測部分
col1, col2 = st.columns(2)


    
#予測結果のグラフを取得
def new_chart():
    response = requests.post(f"{API_URL}prediction",json={"selected_button": st.session_state['selected_button']})
    if response.status_code == 200:
        prediction = response.json()
        # 受け取ったJSONデータをplotlyのFigureに変換
        fig_json = prediction["USD/JPY chart"]
        # JSONデータをplotlyのFigureに変換
        fig = pio.from_json(fig_json)  
        return fig


def add_chart():
    new_data = new_chart()
    
    #前回と押されたbuttonが違う場合、新しく描画
    #同じbuttonの場合はグラフを追加
    if st.session_state['selected_button'] != st.session_state['previous_button']:
    # ボタンが変わった場合、新しいFigureを作成
        print('new_figure')
        fig = go.Figure()
        st.session_state['previous_button'] = st.session_state['selected_button']
        st.session_state.fig = new_data
    else:
        if new_data:
            print('trace_figure')
            for trace in new_data.data:
                st.session_state.fig.add_trace(trace)  # 新しいトレースを追加
    return st.session_state.fig

with col1:
    #コンテナの作成
    pred_chart = st.empty()

#with col2:

    # MAEとhitrateを表示しようと思う


#sleepする時間の定義
sleep_intervals = {
    '5min': 300,   # 5分（5分足の場合）
    '30min': 1800,  # 30分（30分足の場合）
    '60min': 3600,  # 1時間（60分足の場合）
    
}

# スリープ時間を取得する関数
def get_sleep_time():
    selected_button = st.session_state['selected_button']
    # 選択されたボタンのスリープ時間を取得。デフォルトは30分。
    return sleep_intervals.get(selected_button, 1800)

#リアルタイム更新ループ
#為替レートおよびローソク足チャート、予測結果を定期更新する
while True:
    
    # 最新の為替レートを取得
    with real_rate.container():
        rate = get_exchange_rate()
        
        # レートを表示
        real_rate.write(f"現在のUSD/JPY為替レート: {rate}")
    
    #最新の為替チャートを描画
    response = requests.get(f"{API_URL}button_info", params={"selected_button": st.session_state['selected_button']})
    if response.status_code == 200: 
        chart_data = response.json().get(f"USD/JPY_{select_button} chart")
    with real_chart.container():
        fig = pio.from_json(chart_data)
        # レートを表示
        real_chart.plotly_chart(fig)

    #予測結果を描画
    with pred_chart.container():
        
        with st.spinner('モデルの読み込み/予測を実行しています...'):
            
            pred_fig = add_chart()
            if isinstance(pred_fig, dict):
                pred_fig = go.Figure(pred_fig)
    
        #st.write('次の価格の予測結果')
        pred_chart.plotly_chart(pred_fig , use_container_width=True)
        #st.plotly_chart(pred_fig, use_container_width=True)
    #指定した時間で再実行
    time.sleep(get_sleep_time())
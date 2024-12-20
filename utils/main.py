import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
from api.routers import router as fx_router
from api.routers import update_exchange_rate , init_chart

#アプリ起動時バックグラウンドでrateとchartを自動更新し続ける
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # update_exchange_rate をバックグラウンドで実行
    update_rate_thread = threading.Thread(target=update_exchange_rate, daemon=True)
    update_rate_thread.start()
    
    # init_chart をバックグラウンドで実行
    update_chart_thread = threading.Thread(target=init_chart, daemon=True)
    update_chart_thread.start()
    
    # アプリケーションが起動したら次の処理へ
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "https://fx-project-streamlit.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターを登録
app.include_router(fx_router)

# ルートエンドポイント
@app.get("/")
def read_root():
    return {"message": "FX API is running"}

# 開発環境用の起動スクリプト
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if port == 10000 else "127.0.0.1"
    uvicorn.run("main:app", host=host, port=port, reload=True)
    
## プロジェクト名
   my-project

## プロジェクト概要
   このプロジェクトは、FX(為替)の自動売買システムを実装するための時系列予測結果を確認するためのプロジェクトです。

## 機能一覧
   -時間足の選択機能
   ユーザーがボタンで異なる時間足（5分足、30分足、1時間足）を選択でき、選択した時間足に基づいて最新の予測結果を表示します。
   -リアルタイム予測表示
   時間足を変更するごとに、AIモデルによる新しい予測結果が自動的に計算され、画面に表示されます。

## デモ
   ![表示画面のスクリーンショット](images/screenshot1.png)

## 使用技術/ライブラリ

**プログラミング言語**
   Python

**フレームワーク/ライブラリ**
   PyTorch Lightning: モデルのトレーニングと管理の効率化
   Pandas: データ操作と分析
   LSTM (Long Short-Term Memory): 時系列予測のためのニューラルネットワークアーキテクチャ
   FastAPI: APIサーバーの構築とデプロイ
   Plotly: インタラクティブなグラフと可視化
   NumPy: 数値計算や行列操作
   yfinance: 金融データの取得
   scikit-learn: データの前処理と評価指標
   Streamlit: インタラクティブなWebアプリケーションの構築

## インストール手順

1. **リポジトリをクローン**  
    プロジェクトをダウンロードするには、以下のコマンドを使用してリポジトリをクローンします。

        ```bash
        git clone https://github.com/tk-ymd/my-project.git
        cd my-project
        ```

2. **Pythonのインストール**
    [Python公式サイト](https://www.python.org/downloads/)からPythonをインストールしてください（**Python 3.8以上**推奨）。

3. **Visual Studio Codeのインストール**
   [VS Code公式サイト](https://code.visualstudio.com/)からVisual Studio Codeをインストールします。

4. **仮想環境の作成**
   プロジェクトのディレクトリで仮想環境を作成し、有効化します。
     ```bash
     python -m venv myenv
     myenv\Scripts\activate   # Windows
     source myenv/bin/activate  # Mac/Linux
     ```

5. **依存関係のインストール**
   `requirements.txt`からプロジェクトの依存関係をインストールします。
     ```bash
     pip install -r requirements.txt
     ```

6. **アプリケーションの起動**
   FastAPIサーバーを起動する場合:
     ```bash
     uvicorn app:app --reload
     ```
   Streamlitアプリを起動する場合:
     ```bash
     streamlit run app.py
     ```

## 使用方法
   時間足の選択と予測の表示

   アプリケーションの画面で「30分」「1時間」など、異なる時間足のボタンを選択できます。
   時間足を選択すると、それに応じた予測結果が表示されます。
   でータは時間ごとに自動更新されます。

## プロジェクト構成
   '''my-project/
   ├── api/                 # API関連のコード（FastAPIなどのエンドポイント）
   ├── data/                # データファイルを格納するディレクトリ
   ├── images/              # READMEなどで使用する画像ファイル
   ├── logs/                # ログファイルを保存するディレクトリ
   ├── models/              # 機械学習モデルの保存場所
   ├── models_file/         # モデルの定義ファイルや関連ファイル
   ├── tests/               # テストコードを格納するディレクトリ
   ├── utils/               # ユーティリティ関数やヘルパースクリプト
   ├── .gitignore           # Gitで無視するファイルやディレクトリを指定
   ├── README.md            # プロジェクトの説明や手順を記載したREADME
   ├── actual.html          # HTMLレポートや出力ファイル
   └── requirements.txt     # 必要なライブラリの一覧



- **api/**: APIのエンドポイントやリクエストハンドラのコード。
- **data/**: データセットや事前処理済みデータを格納するディレクトリ。
- **images/**: READMEやドキュメントで使用する画像ファイルを格納。
- **logs/**: ログファイルが保存されるディレクトリ。
- **models/**: 訓練済みの機械学習モデルを格納。
- **models_file/**: モデル定義やモデル関連の補助ファイルを格納。
- **new_env/**: 仮想環境（Gitで追跡しないよう.gitignoreに追加を推奨）。
- **tests/**: テストコードを格納し、コードの動作を検証。
- **utils/**: 補助的な関数や共通処理をまとめたスクリプト。
- **.gitignore**: Gitで無視するファイルやディレクトリをリスト化。
- **actual.html**: 出力結果のHTMLファイルやレポート。
- **README.md**: プロジェクトの概要、インストール手順、使い方を記載したドキュメント。
- **requirements.txt**: プロジェクトの依存ライブラリをリストしたファイル。

## 今後の改善点

   **予測結果の向上**
      モデルのハイパーパラメータの調整や、別のアルゴリズムの適用で予測精度を高めます。

   **インターフェースの改善**
      より使いやすく、ユーザーフレンドリーなUIに改良します。

   **DQNを用いた自動売買の実装**
      将来的には、DQN（Deep Q-Network）を用いて自動売買機能を実装し、取引の意思決定を強化学習アルゴリズムで最適化します。これにより、過去のデータに基づいた学習を通じて、より効率的で自動化された取引が可能になることを目指しています。

## 作者情報
   名前:山田 貴大
   GitHub: [\[GitHubプロフィールリンク\]](https://github.com/tk-ymd)
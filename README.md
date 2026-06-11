# 日本株 1〜5日スイング候補ツール

日本株の監視銘柄を日足で分析し、1〜5営業日の短期スイング候補を「買い候補」「監視」「触らない」に分類するStreamlitアプリです。yfinanceで株価を取得し、移動平均、RSI、出来高、押し目、反発兆候、テーマ性を使ってMVP用のスコアを計算します。

## 注意事項

- このツールは投資判断の補助ツールです。
- 自動売買は行いません。
- 最終判断と売買責任は利用者本人にあります。
- yfinanceのデータは遅延、欠損、取得失敗があり得ます。
- APIキー、`.env`、`.streamlit/secrets.toml` はGitHubへコミットしないでください。

## ファイル構成

```text
stock_swing_tool/
├─ app.py
├─ scanner.py
├─ data_fetcher.py
├─ indicators.py
├─ scoring.py
├─ ai_judge.py
├─ notifier.py
├─ backtest.py
├─ config.py
├─ watchlist.csv
├─ trades.csv
├─ requirements.txt
├─ runtime.txt
├─ .env.example
├─ .gitignore
├─ .streamlit/config.toml
├─ .streamlit/secrets.example.toml
└─ README.md
```

## ローカルセットアップ

```bash
cd stock_swing_tool
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

OpenAI APIキーは任意です。空のままでもルールベースのコメントで動きます。

## ローカル起動

```bash
streamlit run app.py
```

ブラウザで以下を開きます。

```text
http://localhost:8501
```

## 同じWi-Fi内のスマホから見る

PC側で以下のように起動します。

```bash
streamlit run app.py --server.address 0.0.0.0
```

別ターミナルでPCのローカルIPを確認します。

```bash
ipconfig
```

スマホを同じWi-Fiにつなぎ、ブラウザで以下の形式にアクセスします。

```text
http://PCのローカルIP:8501
```

WindowsファイアウォールでPythonまたはStreamlitの通信許可が必要になる場合があります。

## watchlist.csvの編集

`watchlist.csv`を以下の形式で編集します。

```csv
code,name,theme,market
6501,日立製作所,AI/電力/データセンター,prime
```

日本株コードはアプリ側で自動的に`.T`を付けてyfinanceから取得します。

Streamlit Community Cloudで銘柄を変えたい場合は、GitHub上の`watchlist.csv`を編集してコミットします。Cloud側でアプリが自動再起動され、変更が反映されます。

## trades.csvの扱い

`trades.csv`は通知候補や手動検証のメモを保存するためのCSVです。

ローカル実行ではファイルに追記されます。Streamlit Community Cloudでも一時的に書き込める場合がありますが、アプリの再起動、再デプロイ、スリープ復帰などで内容が消える可能性があります。Cloud上では永続保存としては扱わず、重要な検証履歴は画面からコピーして別途保存してください。

将来的に永続保存したい場合は、Google Sheets、Supabase、SQLiteを永続ディスクに置ける環境、または外部DBへの保存を追加します。

## Streamlit Community Cloudへデプロイする手順

外出先のスマホからURLで見るには、Streamlit Community Cloudへデプロイするのが一番簡単です。

### 1. GitHubリポジトリを作る

GitHubで新しいリポジトリを作成します。初心者向けには、`stock_swing_tool`フォルダの中身をリポジトリ直下に置く構成がおすすめです。

おすすめのGitHub上の配置:

```text
your-repo/
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ watchlist.csv
├─ trades.csv
└─ ...
```

現在のフォルダごと置く場合は、Streamlit CloudのMain file pathに以下を指定します。

```text
stock_swing_tool/app.py
```

その場合、依存関係ファイルもCloudに認識されるよう、必要に応じて`requirements.txt`をリポジトリ直下にも置いてください。

### 2. GitHubへアップロードする

`.env`、`.streamlit/secrets.toml`、APIキーを書いたファイルはアップロードしません。このプロジェクトの`.gitignore`では以下を除外しています。

```text
.env
.env.*
.streamlit/secrets.toml
secrets.toml
.venv/
__pycache__/
streamlit*.log
```

`.env.example`はサンプルなのでコミットして問題ありません。

### 3. Streamlit Community Cloudでアプリを作成する

1. [Streamlit Community Cloud](https://share.streamlit.io/) にアクセスします。
2. GitHubアカウントでログインします。
3. `Create app` または `New app` を押します。
4. Repositoryに作成したGitHubリポジトリを選びます。
5. Branchは通常 `main` を選びます。
6. Main file pathを指定します。

リポジトリ直下に`app.py`がある場合:

```text
app.py
```

`stock_swing_tool`フォルダごと置いた場合:

```text
stock_swing_tool/app.py
```

7. Deployを押します。

デプロイが終わると、外出先のスマホからアクセスできるURLが発行されます。

## Streamlit CloudのSecrets設定

初期状態ではAPIキーなしで動きます。OpenAI APIを使いたい場合だけSecretsを設定してください。

Streamlit Cloudのアプリ画面で以下を開きます。

```text
App settings → Secrets
```

以下のように入力します。

```toml
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"
JQUANTS_EMAIL = ""
JQUANTS_PASSWORD = ""
NOTIFICATION_CHANNEL = "console"
PRICE_PERIOD = "9mo"
```

OpenAI APIを使う場合は、`OPENAI_API_KEY`に実際のキーを入れます。使わない場合は空欄のままで構いません。

Secretsに入れた値はGitHubには保存されません。GitHubのREADME、CSV、PythonファイルへAPIキーを直接書かないでください。

## Cloudで動かすときのポイント

- `watchlist.csv`はGitHubで管理します。
- `trades.csv`はCloud上では一時保存扱いです。
- `.env`はCloudでは使いません。Secretsを使います。
- `runtime.txt`でPython 3.12を指定しています。
- `requirements.txt`に必要な依存関係をまとめています。
- 株価取得はyfinanceを使うため、時間帯や通信状況によって一部銘柄が取得失敗になる場合があります。

## 画面でできること

- 今日の買い候補をカード形式で確認
- 監視銘柄の「何待ちか」をカード形式で確認
- 銘柄ごとのスコア、買い条件、損切り、利確目安、期待値を確認
- チャートとスコア内訳を確認
- ChatGPTへ貼り付けやすい通知文をコピー
- 候補を`trades.csv`へ記録

## バックテスト

簡易バックテストはローカルで以下を実行します。

```bash
python backtest.py
```

過去データ上でスコア70点以上の候補を抽出し、翌日、3日後、5日後の終値リターンを集計します。MVPの簡易検証なので、約定価格、手数料、スリッページ、寄り付きギャップは未考慮です。

## 将来の拡張案

- J-Quants APIへのデータ取得差し替え
- Google Sheetsや外部DBへの検証履歴保存
- 日経平均、TOPIX、米国指数、先物を使った地合いフィルター
- Discord、Gmail、LINE系通知
- 売買履歴の自動評価とチャート上の売買ポイント表示
- スコア条件の画面編集

## MVPで未実装または簡易実装の部分

- 外部通知は未実装です。
- バックテストは簡易実装です。
- J-Quants API接続は未実装です。
- 地合いフィルターは個別銘柄トレンドによる仮スコアです。
- Cloud上の`trades.csv`は永続保存ではありません。
- OpenAI APIが使えない場合はルールベースコメントに自動フォールバックします。

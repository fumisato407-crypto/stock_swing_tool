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
├─ intraday_scanner.py
├─ entry_rules.py
├─ alert_builder.py
├─ ai_judge.py
├─ paper_trader.py
├─ virtual_trade_store.py
├─ outcome_tracker.py
├─ pattern_stats.py
├─ stock_personality.py
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

OpenAI APIキーは任意です。`OPENAI_API_ENABLED=false`がデフォルトなので、APIキーが入っていても通常の株価スキャン、スコア計算、候補コメント生成、ルールベース仮想ログ保存ではOpenAI APIを呼びません。

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
data_cache/
alerts_log.csv
virtual_trades.db
virtual_trades.db-*
replay_trades.db
replay_trades.db-*
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

## Streamlit Cloudに変更が反映されないとき

PCローカル版とStreamlit Cloud版のUIがズレている場合は、まずGitHubに最新commitがpushされているか確認します。

```bash
git status
git log --oneline -3
git fetch origin
git status --short --branch
```

`main...origin/main` の後ろに `ahead` が出ている場合は、ローカルのcommitがGitHubへpushされていません。

```bash
git push origin main
```

GitHubへpush済みなのにCloud版が古い場合は、Streamlit Community Cloud側で再起動します。

1. Streamlit Community Cloudで対象アプリを開きます。
2. 右下または右上の `Manage app` を開きます。
3. `Reboot app` を押します。
4. それでも変わらない場合は、同じ画面から `Deploy` または `Redeploy` を実行します。
5. `Logs` で最新commitが読み込まれているか、エラーが出ていないか確認します。

Cloudの設定で確認する項目:

- Repositoryが正しいGitHubリポジトリになっていること
- Branchが `main` になっていること
- Main file pathが、リポジトリ直下なら `app.py`、フォルダごと置いた場合は `stock_swing_tool/app.py` になっていること
- Cloudのログで古いcommitではなく最新commitが使われていること

スマホ側に古い画面キャッシュが残る場合があります。Cloudを更新した後は、スマホで以下も試してください。

- ブラウザの更新ボタンを押す
- URL末尾に `?v=2` のような適当なクエリを付けて開く
- 別ブラウザで開く
- シークレットモードまたはプライベートブラウズで開く
- Streamlit画面右上のメニューから `Rerun` を押す

最新UIでは、トップ画面には件数サマリーと案内文だけが表示され、買い候補カードは直接並びません。買い候補の詳細は `買い候補` タブ内の折りたたみを開いて確認します。

## Streamlit CloudのSecrets設定

初期状態ではAPIキーなしで動きます。OpenAI APIやDiscord通知を使いたい場合だけSecretsを設定してください。

Streamlit Cloudのアプリ画面で以下を開きます。

```text
App settings → Secrets
```

以下のように入力します。

```toml
OPENAI_API_KEY = ""
OPENAI_API_ENABLED = false
OPENAI_MODEL = "gpt-4o-mini"
AI_VIRTUAL_MODEL = "gpt-5.5"
JQUANTS_EMAIL = ""
JQUANTS_PASSWORD = ""
NOTIFICATION_CHANNEL = "console"
DISCORD_WEBHOOK_URL = ""
DISCORD_MENTION_ID = ""
PRICE_PERIOD = "6mo"
```

OpenAI APIを使う場合は、`OPENAI_API_KEY`に実際のキーを入れます。使わない場合は空欄のままで構いません。

Discord通知を使う場合は、`DISCORD_WEBHOOK_URL`にDiscordのWebhook URLを入れます。Webhook URLはPythonファイル、README、CSVには直接書かず、ローカルでは`.env`、Streamlit CloudではSecretsにだけ保存してください。

スマホで通知に気づきやすくしたい場合は、任意で`DISCORD_MENTION_ID`にDiscordユーザーIDを入れます。空欄ならメンションなしで送信します。ID本体は画面やログに表示しません。

Secretsに入れた値はGitHubには保存されません。GitHubのREADME、CSV、PythonファイルへAPIキーを直接書かないでください。

## Cloudで動かすときのポイント

- `watchlist.csv`はGitHubで管理します。
- `trades.csv`はCloud上では一時保存扱いです。
- `virtual_trades.db`はAI仮想取引専用の保存先です。Cloud上では一時保存扱いです。
- Streamlit Cloud上のSQLiteは永続保存に向きません。本命の学習DBはローカルPCで運用してください。
- `.env`はCloudでは使いません。Secretsを使います。
- `runtime.txt`でPython 3.12を指定しています。
- `requirements.txt`に必要な依存関係をまとめています。
- 株価取得はyfinanceを使うため、時間帯や通信状況によって一部銘柄が取得失敗になる場合があります。

## 画面でできること

- 今日の買い候補をカード形式で確認
- 監視銘柄の「何待ちか」をカード形式で確認
- 銘柄ごとのスコア、買い条件、損切り、利確目安、期待値を確認
- チャートとスコア内訳を確認
- 場中の5分足から「買い検討OK」「監視強化」「見送り」を確認
- GPTによるAI仮想取引の判断、仮想成績、銘柄別クセ、パターン別勝率を確認
- 取得失敗時に正規化後シンボル、エラー種別、取得行数を確認
- ChatGPTへ貼り付けやすい通知文をコピー
- 候補を`trades.csv`へ記録

## 場中エントリー監視

`場中エントリー監視`タブでは、日足ベースのスイング候補とは別に、5分足を使って「今買える形になったか」を判定します。自動売買は行わず、画面上の通知と通知文生成だけを行います。

使い方:

1. `場中エントリー監視`タブを開きます。
2. 対象を選びます。
   - `watchlist.csv全銘柄を対象`
   - `買い候補・監視銘柄を対象に含める`
   - 手動入力コード
3. `場中データを更新`を押します。
4. `買い検討OK`、`監視強化`、`見送り`の折りたたみを開いて詳細を確認します。

取得テスト:

- `場中エントリー監視`タブの`取得テスト`から、任意の銘柄コードでyfinance取得状況を確認できます。
- 初期値は`5803`です。
- 入力コード、yfinance用シンボル、日足取得行数、5分足取得行数、最新Close、エラー内容を表示します。
- `5803`、`5803.T`、`5803.0`、`5803.T.T`のような入力は共通の正規化処理で`5803.T`に変換します。

判定に使う主な要素:

- 現在値が短期線・中期線・長期線の上にあるか
- 当日安値から反発しているか
- 直近安値を切り上げているか
- 50円刻み、100円刻みなどの節目を突破・維持しているか
- 出来高が直近平均より増えているか
- 損切りを近く置けるか
- 第1利確までの値幅が損切り幅以上あるか

通知条件:

- `intraday_score >= 75` かつハードな見送り条件がない場合: `買い検討OK`
- `65 <= intraday_score < 75` かつハードな見送り条件がない場合: `監視強化`
- それ以外: `見送り`

以下の場合はスコアに関係なく`買い検討OK`にしません。

- 前日安値割れ
- 当日安値更新中
- 出来高が極端に少ない
- 損切り幅が利確幅より大きすぎる
- 第1利確までの距離が近すぎる
- 移動平均線の下で推移

画面通知:

- `買い検討OK`または`監視強化`の銘柄は、Streamlit画面上に通知されます。
- 同じ銘柄で同じ条件の通知が連続しないように、セッション内で通知済みキーを管理します。
- 通知履歴はローカルでは`alerts_log.csv`に保存されます。

Discord通知:

- ローカルでは`.env`、Streamlit CloudではSecretsに`DISCORD_WEBHOOK_URL`を設定します。
- 任意で`DISCORD_MENTION_ID`を設定すると、通知本文の先頭にDiscordメンションを付けます。
- `場中エントリー監視`タブで`Discord通知ON/OFF`をONにすると、条件を満たした`買い検討OK`だけをDiscordへ送信します。
- `見送り`はDiscord通知しません。
- `テスト通知`ボタンでWebhook設定を確認できます。
- 同じ銘柄・同じ判定はセッション内で重複通知しないようにしています。
- 通知の成功/失敗は画面に表示され、`alerts_log.csv`にも`discord_sent`、`discord_error`として保存されます。

本番通知ルール:

- `Discord通知ON/OFF`がONのときだけ送信します。
- 自動通知は市場時間内のみです。テスト通知は市場時間外でも送れます。
- `自動監視ON/OFF`がONの間、ブラウザでStreamlit画面を開いているセッションだけが60秒、120秒、300秒の選択間隔で場中スキャンを再実行します。
- 通知対象は場中エントリー監視で`買い検討OK`、かつ`intraday_score >= 70`の銘柄だけです。
- `監視強化`や60点台の監視候補はDiscord通知しません。
- 初回スキャン時の既存候補は一括通知しません。
- 同じ銘柄は最低30分あけてから再通知します。
- Webhook URLとDiscordユーザーIDは画面やログに表示しません。
- PCスリープ中、Streamlit停止中、ブラウザを閉じている間は自動監視できません。

注意:

- yfinanceの5分足が取得できない時間帯や銘柄では、該当銘柄だけ`取得失敗`として表示します。
- 全銘柄が取得失敗になった場合は、銘柄コード形式、yfinance接続、watchlist.csv形式を確認するためのデバッグ表を表示します。
- Streamlit Cloud上の`alerts_log.csv`は永続保存ではありません。
- 場中エントリー監視は、既存の日足スイング候補抽出とは別判定です。
- 最終判断は必ず人間が行ってください。

## AI仮想取引

`AI仮想取引`タブでは、既存スコアで抽出した候補をルールベースまたはGPTでpaper trading / virtual trading専用の仮想判断として保存します。実売買、発注、自動売買、Discord本番通知への混在は行いません。

判断結果:

- `virtual_buy`: 仮想ポジションとしてopen保存します。
- `virtual_watch`: 判断ログとして保存します。
- `virtual_avoid`: 判断ログとして保存します。

保存先:

- AI仮想取引は`virtual_trades.db`に保存します。
- `trades.csv`や実トレード記録とは混ぜません。
- `virtual_trades.db`は`.gitignore`で除外しています。
- Streamlit Cloud上の`virtual_trades.db`は永続保存ではありません。
- Streamlit Cloud上のSQLiteは再起動や再デプロイで消える可能性があります。検証を積み上げる本命DBはローカルPCで運用してください。
- AI仮想取引ログには、判断元を示す`judge_source`、使用モデルを示す`model_used`、GPT生成かどうかを示す`is_ai_generated`を保存します。
- `judge_source`は、GPT判断なら`gpt`、ルールベースfallbackなら`fallback`、DB疎通テストなら`test`になります。
- AI仮想取引ログの時刻はJST基準で保存・表示します。互換性のため既存の`timestamp`も残し、新規保存では`timestamp_jst`と`created_at_jst`も保存します。

使い方:

1. `AI仮想取引`タブを開きます。
2. まず`株価スキャンを実行`で候補を作ります。
3. `相場中ルール買いログ自動保存`をONにするか、`手動で1回保存`で仮想ログを保存します。
4. 手動で候補を保存したい場合は、`手動の仮想判断`を開いて`AI仮想判断を実行`を押します。
5. `仮想成績`タブでopen仮想取引の結果を後追い更新します。

AI仮想取引タブの見方:

- `現在の運用状態`では、OpenAI API使用ON/OFF、判定方式、API calls、仮想ログ件数、open件数、closed件数を確認します。
- `仮想買い時刻`は、その仮想取引を買った扱いにした時刻です。画面表示と保存時刻はJST基準です。
- `買値`は仮想エントリー価格、`損切り`は撤退目安、`利確目標`は利益確定の目安です。
- `open`または`保有中`はまだ検証中、`closed`または`終了`は結果確定済みです。
- `判定方式`が`ルールベース`ならOpenAI APIを使っていません。
- `API calls`が0なら、そのセッションではOpenAI API呼び出しが発生していないためAPI料金は発生していません。
- `詳細デバッグ情報`は接続テスト、DB疎通テスト、内部状態確認用です。通常運用では見なくてかまいません。

AIに渡す情報:

- GPT判断時点の銘柄名、価格、固定スコア、entry_type、買い条件、損切り、利確目安、判定理由などを`market_snapshot_json`として保存します。
- 未来データは判断時点では渡しません。
- 1時間後、当日終値、1営業日後、3営業日後、5営業日後の検証は、後から`outcome_tracker.py`が行います。
- 同一ローソク内で利確と損切りが両方到達した場合は、保守的に損切り優先で判定します。

OpenAI API:

- ローカルでは`.env`、Streamlit CloudではSecretsに`OPENAI_API_KEY`を設定します。
- `OPENAI_API_ENABLED=false`がデフォルトです。
- APIキーが入っていても、`OPENAI_API_ENABLED=false`または画面の「OpenAI API使用」がOFFならGPT判断は呼びません。
- ルールベース仮想ログ保存ではOpenAI APIを呼ばないため、API料金は発生しません。
- GPT判断を使う時だけ、Secretsまたは`.env`で`OPENAI_API_ENABLED=true`にするか、画面の「OpenAI API使用」をONにします。
- ChatGPT Plusの月額料金とOpenAI APIの従量課金は別です。GPT判断をONにするとAPI側の課金対象になります。
- AI仮想取引のモデルは`AI_VIRTUAL_MODEL`で指定できます。初期値は`gpt-5.5`です。
- 通常の株価スキャン、スコア計算、候補コメント生成ではOpenAI APIを呼びません。
- OpenAI APIを呼ぶのは、OpenAI API使用がONの状態で`OpenAI接続テスト`、または`AI仮想判断を実行`を押した時だけです。
- 画面上の`openai_call_count`で、現在のセッション内のOpenAI API呼び出し回数を確認できます。
- 初期表示直後は`openai_call_count=0`です。
- APIキーがない、またはAPI呼び出しに失敗した場合は、アプリが落ちないようにルールベースの仮想判断へフォールバックします。
- APIキーや秘密情報は画面、ログ、DBに表示しません。
- APIキーはGitHubにコミットしないでください。`.env`と`.streamlit/secrets.toml`は`.gitignore`で除外します。
- OpenAI APIキーなしで保存した仮想判断は、`judge_source=fallback`、`model_used=rule_based_fallback`、`is_ai_generated=0`として記録されます。

相場中ルール買いログ自動保存:

- 基本運用はOpenAI API使用OFFです。OFFならAPIキーが入っていてもGPT判断は呼ばず、API料金は発生しません。
- `AI仮想取引`タブの`相場中ルール買いログ自動保存`をONにすると、Streamlit画面を開いている間だけ自動実行します。
- 実行間隔は1分、3分、5分、10分から選べます。
- 対象は`買い候補のみ`または`買い候補＋監視`から選べます。
- 最小スコアと最大保存件数を設定し、条件に合う上位候補を`virtual_trades.db`へpaper tradingログとして保存します。
- 自動保存は市場時間中だけ動きます。平日の前場09:00〜11:30、後場12:30〜15:30が対象で、昼休み、時間外、土日は停止します。祝日は今後対応予定です。
- 自動保存は安全運用のため`rule_based_fallback`で実行します。`judge_source=fallback`、`model_used=rule_based_fallback`、`is_ai_generated=0`、`fallback_reason=openai_disabled`として保存されます。
- 同一銘柄、同一entry_type、同一judge_sourceの短時間重複保存は30分抑制します。
- これは実売買・自動売買・発注ではありません。検証用のpaper tradingです。
- `trades.csv`とは混ぜません。AI仮想取引ログは`virtual_trades.db`に保存します。
- Streamlit Cloud上のSQLiteは永続保存に向きません。本命の長時間運用は、ローカルPC常駐の`market_runner.py`を使います。
- 平日場中を待たずにCloud上で確認したい場合は、`手動で1回保存`を押します。市場時間判定を無視して1回だけ同じ保存ロジックを実行します。

ローカル常駐の自動収集:

- 本命運用はローカルPCで`market_runner.py`を動かし、`virtual_trades.db`を継続的に育てる形です。
- Streamlit Cloudの自動保存は、画面を開いている間だけ動く簡易自動実行です。Cloud上のSQLiteは再起動や再デプロイで消える可能性があるため、本命保存先にしないでください。
- `market_runner.py`も実売買・自動売買・発注ではありません。paper tradingログを`virtual_trades.db`へ保存するだけです。
- OpenAI API使用はOFF固定の運用です。APIキーがあっても`--use-openai off`でAPI料金は発生しません。
- 市場時間外、昼休み、土日は自動待機します。終了するときはターミナルで`Ctrl+C`を押します。

起動例:

```bash
python market_runner.py --interval-minutes 5 --target buy --min-score 70 --max-candidates 3
```

1回だけ動作確認:

```bash
python market_runner.py --once --force-market-open --dry-run
python market_runner.py --once --force-market-open
```

Windows用bat:

```bat
run_market_runner.bat
```

`run_market_runner.bat`はプロジェクトフォルダへ移動し、`.venv`があれば有効化して、5分間隔で`market_runner.py`を起動します。処理ログは`logs/market_runner.log`に残ります。

Windowsタスクスケジューラの例:

1. タスクスケジューラを開きます。
2. `基本タスクの作成`を選びます。
3. 名前を`stock_swing_tool market runner`などにします。
4. トリガーを`毎週`にし、月曜から金曜の8:55に設定します。
5. 操作は`プログラムの開始`を選びます。
6. プログラムに`C:\Users\fumi\Documents\New project\stock_swing_tool\run_market_runner.bat`を指定します。
7. 実行後は市場時間外なら待機し、市場時間中だけスキャンとpaper tradingログ保存を行います。

成績表示:

- `仮想成績`タブで仮想取引の件数、open件数、勝率、平均リターンを確認できます。
- `パターン別勝率`タブでentry_type別、銘柄別、地合い別の勝率、平均利益、平均損失、期待値を確認できます。
- `銘柄別クセ`タブで銘柄ごとの得意パターン、苦手パターン、最適保有期間、注意点を確認できます。
- サンプル数が30未満なら「仮説」、50未満なら「参考」、50以上で「信頼度 中」、100以上で「信頼度 高」と表示します。

## 過去リプレイ検証

`過去リプレイ検証`タブでは、過去データを使ってルール買いのpaper trading / backtestを行います。リアルタイム監視とは別に、指定した銘柄、期間、時間足で過去相場を再生し、仮想買いポイントとその後の成績を確認できます。

基本方針:

- 実売買、発注、自動売買ではありません。
- OpenAI APIは使いません。ルールベースのみで検証します。
- Discord本番通知は行いません。
- `trades.csv`や`virtual_trades.db`とは混ぜず、結果は`replay_trades.db`へ保存します。
- 買い判定では、その時点以前のローソク足だけを使います。未来の高値、安値、終値は買い判断に使いません。
- 未来データは、利確、損切り、期限到達などの結果検証にだけ使います。

使い方:

1. `過去リプレイ検証`タブを開きます。
2. 銘柄コード、期間、時間足を選び、`過去データ取得`を押します。
3. 取得データ件数、最初の日時、最後の日時を確認します。
4. 開始日時、終了日時、最小スコア、対象ルール、最大仮想買い件数を設定します。
5. `リプレイ検証を実行`を押します。
6. 勝率、平均リターン、最大利益率平均、最大下落率平均、損切り到達件数、利確到達件数を確認します。
7. チャートでは終値、仮想買いポイント、損切りライン、利確ラインを確認できます。

100株想定損益:

- `想定株数`は100株、200株、300株、500株、1000株から選べます。初期値は100株です。
- `損益円`は`(exit_price - entry_price) * 想定株数`で計算します。
- `必要資金`は`entry_price * 想定株数`です。
- `累計利益`は勝ちトレードの利益合計です。
- `累計損失`は負けトレードの損失合計です。
- `純損益`は利益と損失を合算した最終損益です。
- `損益比`は平均利益 ÷ 平均損失の絶対値です。
- `累計損益カーブ`では、終了済みトレードだけを時系列で積み上げます。
- open中、検証中、exit_priceがないトレードは確定損益に含めません。
- 手数料、税金、スリッページはMVPでは未考慮です。
- この表示は過去リプレイ検証用のpaper tradingであり、実売買ではありません。

データについて:

- yfinanceの無料データを使うため、取得できる期間や時間足には制限があります。
- まずは`5分足`、`1時間足`、`日足`で試してください。
- 取得済みデータは`data_cache/`へCSV保存し、次回以降はキャッシュを再利用します。
- `data_cache/`、`replay_trades.db`、`replay_trades.db-*`は`.gitignore`で除外しています。
- リアルタイム運用の`market_runner.py`で貯めるログと、過去リプレイ検証は補完関係です。過去でルールの癖を見て、場中運用で実際の候補ログを貯める流れを想定しています。

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
- Gmail、LINE系通知
- 売買履歴の自動評価とチャート上の売買ポイント表示
- スコア条件の画面編集

## MVPで未実装または簡易実装の部分

- Discord通知は実装済みです。Gmail、LINE系通知は未実装です。
- バックテストは簡易実装です。
- J-Quants API接続は未実装です。
- 地合いフィルターは個別銘柄トレンドによる仮スコアです。
- Cloud上の`trades.csv`、`alerts_log.csv`、`virtual_trades.db`は永続保存ではありません。
- OpenAI APIが使えない場合はルールベースコメントに自動フォールバックします。

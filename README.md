# PMT (yfinance × Django)

銘柄コード（ティッカー）を入力すると `yfinance` から価格を取得し、**月次リターン**で以下を可視化します。

- 初期資金（デフォルト 10,000）を起点とした**月次リターンの蓄積（資産推移）**
- **月次リターンの比較**
- 指標: CAGR / 年率Vol / Sharpe / VaR(月次) / ES(月次)

## 起動

```bash
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

ブラウザで `http://127.0.0.1:8000/` を開きます。

## 使い方

- 銘柄コードはカンマ区切りで入力します（例: `AAPL, MSFT, 7203.T, ^N225`）
- 期間は任意（空なら yfinance 既定の範囲）
- VaR/ES は月次リターンの**ヒストリカル法**（損失を正の数で表示）
- Sharpe は年率無リスク金利を月次に変換して年率化しています


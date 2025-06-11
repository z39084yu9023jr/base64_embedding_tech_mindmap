# base64_embedding_tech_mindmap

画像からマインドマップ形式のMarkdownを生成するPythonツール。

## 機能

- 高精度OCR理解: 技術書籍の文字・図表を正確に抽出
- 階層構造の自動認識: 章→節→小節の論理的整理
- 図表情報の統合: グラフ・イラスト・写真の内容を文章に統合
- 表形式データの変換: 比較情報や分類データの構造化表示
- 見開きページ対応: 複数ページの最適な合成処理
- 複数プロンプト対応: 用途に応じた分析手法の選択
- 出力ファイル名カスタマイズ: プレフィックス機能付き
- JSON化処理の追加: 画像から構造化テキスト情報(JSON)を抽出
- 要約生成の追加: JSON構造を元にした要約テキストの生成

## ファイル構成

```
tech-mindmap-generator/
├── script.py                    # メインスクリプト
├── requirements.txt             # 依存関係
├── README.md                   # このファイル
├── LICENSE                     # ライセンス
└── .gitignore                  # Git除外設定
```

## 必要環境

- **Python**: 3.8以上
- **Ollama**: ローカルで動作するLLMサーバー

## セットアップ方法

### 1. リポジトリのクローン

```bash
git clone https://github.com/z39084yu9023jr/base64_embedding_tech_mindmap.git
base64_embedding_tech_mindmap
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. Ollamaのセットアップ

```bash
# Ollamaのインストール（macOS/Linux）
curl -fsSL https://ollama.ai/install.sh | sh

# Ollamaサーバーの起動
ollama serve
```

## 使用方法

### 基本的な使用法

```bash
# 単一画像の処理
python script.py image.png

# ディレクトリ内の全画像を処理
python script.py ./book_pages/

# ワイルドカードを使用
python script.py "./pages/*.png"
```

### オプション

```bash
# レイアウト指定（縦並び/横並び）
python script.py ./pages/ --layout vertical
python script.py ./pages/ --layout horizontal

# プロンプト種類の選択
python script.py ./pages/ --prompt enhanced        # OCR理解・階層構造対応（デフォルト）
python script.py ./pages/ --prompt comprehensive  # 包括的分析
python script.py ./pages/ --prompt section        # セクション構造化
python script.py ./pages/ --prompt visual         # 視覚情報統合

# 出力ファイル名の指定
python script.py ./pages/ --output my_mindmap.md

# ファイル名プレフィックスの追加
python script.py ./pages/ --prefix chapter1 --output mindmap.md
# → 出力: chapter1_mindmap.md

# カスタムプロンプトの使用
python script.py ./pages/ --custom "あなた独自の分析指示"

# JSON構造の抽出と要約の保存を有効化
python script.py ./pages/ --json-summary
```

### 実用例

```bash
# 見開きページを横並びで処理、要約も保存
python script.py "./book/page*.png" --layout horizontal --prompt comprehensive --prefix chapter2 --json-summary

# 単一ページの詳細分析（要約付き）
python script.py single_page.jpg --prompt section --output detailed_analysis.md --json-summary

# 図表重視の分析
python script.py "./diagrams/*.png" --prompt visual --prefix diagrams
```

## 出力形式
<prefix>_mindmap.md # マインドマップ（Markdown形式）
<prefix>_summary.txt # 要約テキスト（プレーンテキスト）
<prefix>_structure.json # 構造情報（JSON形式）

### マインドマップ例（Markdown）

```markdown
# メインタイトル

## セクションタイトル
  - 重要ポイント1
    + 詳細情報
    + 具体例
  - 重要ポイント2
    + 関連キーワード

## 比較情報（表形式）

| カテゴリ | 特性1 | 特性2 | 特性3 |
|---------|------|------|------|
| 項目1   | 内容  | 内容  | 内容  |
| 項目2   | 内容  | 内容  | 内容  |

## 用語
  - `専門用語`: 定義・説明
  - **重要概念**: 強調表示
```

## トラブルシューティング

### Ollamaサーバーに接続できない場合

```bash
# サーバーの状態確認
curl http://localhost:11434/api/tags

# サーバーの再起動
ollama serve
```

### メモリ不足エラーの場合

- より軽量なモデルの使用を検討
- 画像サイズの調整（スクリプト内のMAX_WIDTH, MAX_HEIGHTパラメータ）

### OCR精度が低い場合

- 画像の解像度を上げる
- コントラストの改善
- 複数の小さな画像に分割して処理

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 関連リンク

- [Ollama公式サイト](https://ollama.ai/)

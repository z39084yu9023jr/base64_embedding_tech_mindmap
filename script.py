# -*- coding: utf-8 -*-
"""
File: script.py
Description:画像からマインドマップ形式のマークダウンを生成（強化版）
Author: z39084yu9023jr
Created: 2024
Version: 3.2 - OCR理解・階層構造・表形式対応強化版 + プレフィックスオプション追加 + 要約文章生成機能追加
Requirements: Pillow, requests
Usage: python enhanced_tech_mindmap_generator.py

Features:
- 高精度OCR理解によるテキスト抽出
- 図表・イラスト情報の統合
- 階層構造の自動認識と整理
- 表形式データの適切な変換
- 比較情報の構造化表示
- 見開きページ対応の最適化
- 出力ファイル名プレフィックス機能
- JSON形式OCRデータからの要約文章生成
"""

from PIL import Image, ImageEnhance, ImageFilter
import base64
import os
import sys
import glob
import json
import time
from io import BytesIO
from pathlib import Path
import requests

# 設定（技術書籍用に最適化）
MAX_WIDTH = 2048
MAX_HEIGHT = 1536
MAX_FILE_SIZE = 1024 * 1024
INITIAL_QUALITY = 90
OLLAMA_IMAGE_MODEL = "yourmodel"
OLLAMA_TEXT_MODEL = "yourmodel"
OLLAMA_MODEL = "yourmodel"
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 300


# ===== 追加部分1: プロンプト定数の追加 (ファイル先頭の設定部分に追加) =====
SUMMARY_PROMPT = """
あなたは内容をやさしく説明するアシスタントです。
以下の内容を踏まえて、誰にでも理解しやすい形で丁寧に要約してください。

【要約の条件】
- 内容のタイトルを文章の先頭に付けて下さい
- 難しい専門用語は使わず、もし使う場合はわかりやすく説明を入れてください
- 一つひとつの概念を順を追って説明してください
- 丁寧に例や背景を含めて説明してください
- 初心者でも理解できるような配慮を心がけてください
- あまりにも幼稚な文章表現は避けて下さい

【出力形式】
- 文章形式（マークダウン不要）
- 説明的な文体、ある程度親しみはあるが幼稚ではない表現
- 推論型のLLMの場合、推論を行う過程(<think>⋯</think>)は不要

以下は、画像や動画のOCRから得られたテキスト情報です。この内容をわかりやすく説明してください：
"""


def enhance_image_for_technical_text(image):
    """
    技術書籍用の画像前処理（OCR精度重視）
    """
    # より強いコントラスト強化（薄い印刷・図表対応）
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # シャープネス強化（文字・図表の境界明確化）
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.4)
    
    # 明度調整（影・グラデーション補正）
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    # 高精度シャープニングフィルター
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=1))
    
    return image


def resize_image_for_technical(image, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    リサイズ（OCR可読性重視）
    """
    width, height = image.size
    
    # アスペクト比を保持しつつ、OCR最適サイズに調整
    if width <= max_width and height <= max_height:
        # 小さすぎる場合は拡大（OCR精度向上）
        if width < 1000 or height < 700:
            scale = min(1000 / width, 700 / height)
            new_size = (int(width * scale), int(height * scale))
            return image.resize(new_size, Image.LANCZOS)
        return image
    
    ratio = min(max_width / width, max_height / height)
    new_size = (int(width * ratio), int(height * ratio))
    return image.resize(new_size, Image.LANCZOS)


def combine_images_for_technical_book(image_paths, layout='vertical'):
    """
    画像合成
    """
    images = []
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")
        try:
            img = Image.open(path).convert("RGB")
            enhanced_img = enhance_image_for_technical_text(img)
            resized_img = resize_image_for_technical(enhanced_img)
            images.append(resized_img)
            print(f"ページ{i+1}処理完了: {path} ({img.size} -> {resized_img.size})")
        except Exception as e:
            raise Exception(f"画像の読み込みに失敗しました ({path}): {e}")
    
    if not images:
        raise ValueError("有効な画像が見つかりませんでした")
    
    if layout == 'vertical':
        # 縦並び（順序）
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        
        max_width = max(widths)
        total_height = sum(heights)
        
        # ページ間スペース（セクション区切り明確化）
        page_spacing = 50
        total_height_with_spacing = total_height + (len(images) - 1) * page_spacing
        
        new_img = Image.new("RGB", (max_width, total_height_with_spacing), (255, 255, 255))
        
        y_offset = 0
        for i, img in enumerate(images):
            # 横方向中央寄せ
            x_offset = (max_width - img.width) // 2
            new_img.paste(img, (x_offset, y_offset))
            y_offset += img.height
            
            # ページ区切り線を追加（視覚的分離）
            if i < len(images) - 1:
                line_y = y_offset + page_spacing // 2
                for x in range(max_width):
                    for line_width in range(-2, 3):  # 5px幅の線
                        if 0 <= line_y + line_width < new_img.height:
                            new_img.putpixel((x, line_y + line_width), (200, 200, 200))
                y_offset += page_spacing
    
    else:  # horizontal（見開き対応）
        # 横並び
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        
        total_width = sum(widths)
        max_height = max(heights)
        
        page_spacing = 80  # 見開き用の広いスペース
        total_width_with_spacing = total_width + (len(images) - 1) * page_spacing
        
        new_img = Image.new("RGB", (total_width_with_spacing, max_height), (255, 255, 255))
        
        x_offset = 0
        for i, img in enumerate(images):
            # 縦方向中央寄せ
            y_offset = (max_height - img.height) // 2
            new_img.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
            # ページ区切り線を追加
            if i < len(images) - 1:
                line_x = x_offset + page_spacing // 2
                for y in range(max_height):
                    for line_width in range(-2, 3):
                        if 0 <= line_x + line_width < new_img.width:
                            new_img.putpixel((line_x + line_width, y), (200, 200, 200))
                x_offset += page_spacing
    
    print(f"合成画像サイズ: {new_img.size} (レイアウト: {layout})")
    return new_img


def create_enhanced_mindmap_prompts():
    """
    強化されたマインドマップ生成用プロンプト（OCR理解・階層構造・表形式対応）
    """
    prompts = {
        'enhanced_mindmap': """あなたは、画像認識（OCRと図解理解）によって添付された画像から情報を抽出し、
その内容をもとに「マインドマップ」をMarkdown形式で出力する専門アシスタントです。

【出力フォーマット】
- トップノード：タイトルなど、最上位の見出し（Markdown見出し記号「#」を使用）
- サブノード：重要キーワード・要点（インデント2スペース＋「-」）
- 枝ノード：関連キーワードや具体例（インデント4スペース＋「+」や「*」）
- 補足説明が必要な場合は「（例：～）」「補足：～」を追加

【手順】
1. 画像入力からOCRでテキストを抽出する
2. テキストを階層構造に整理する
3. 図・表・イラストなどを要点として統合する
4. 各ノードに、キーワードや要点を箇条書きで追加する
5. 長い文章は1行の要約にまとめ、簡潔かつ読みやすいMarkdown形式に整形する

【表形式の活用】
- 複数の項目を比較する情報（例：特性比較、長所短所、分類表など）は表形式で表現してください
- リスト形式では伝わりにくい構造化情報は、適宜表形式を用いて視覚的に整理してください
- 表形式が適切な場合のフォーマット例：

| カテゴリ | 特性1 | 特性2 | 特性3 |
|---------|------|------|------|
| 項目1   | 内容  | 内容  | 内容  |
| 項目2   | 内容  | 内容  | 内容  |

【重要な指示】
- ノード間の関係性が明確になるように、階層の深さや配置を工夫してください
- 比較データや特性一覧などは、必要に応じてMarkdownのテーブル形式を使用して表現してください
- 用語は `バッククォート` で囲み、重要な概念は **太字** で強調してください
- 図表の情報も文章と同等に扱い、マインドマップに統合してください

画像を詳細に分析し、上記の形式でマインドマップを作成してください。""",

        'comprehensive_analysis': """内容を包括的に分析し、理解しやすい構造化マインドマップを作成してください。

【分析重点項目】
1. **階層構造の明確化**: 論理的な流れを整理
2. **視覚情報の統合**: 図表・グラフ・イラストの内容を文章と統合
3. **比較情報の表形式化**: 特性比較、分類表、仕様一覧などを表で表現
4. **用語の体系化**: 用語の定義と関連性を明記
5. **実装・応用例の整理**: 具体例やベストプラクティスを構造化

【出力要求】
- OCRによる正確なテキスト抽出
- 図表情報の自然な統合
- 階層構造の論理的整理
- 表形式による比較情報の視覚化
- 用語の適切な強調とカテゴライズ

マークダウン形式で、学習効果の高い構造化されたマインドマップを生成してください。""",

        'section_structured': """このセクションの内容を詳細に構造化し、以下の観点から分析してください：

【構造化分析】
1. **セクション概要**: 主要テーマと目的の明確化
2. **階層構造**: 見出し→小見出し→段落の論理的整理
3. **概念関係**: 各概念間の依存関係と相互作用
4. **視覚情報**: 図表・グラフ・イラストの詳細分析
5. **比較・分類**: 特性比較や分類情報の表形式整理

【技術的要素】
- コードスニペットの解説
- アルゴリズムの手順説明
- 設計パターンの構造
- システム構成の関係性

【出力形式】
- 階層的なマークダウン構造
- 適切な表形式による比較情報
- 用語の定義と関連付け
- 具体例と補足説明の統合

構造化された学習しやすいマインドマップを作成してください。""",

        'visual_integration': """画像内の視覚情報（図表・グラフ・イラスト・写真）を重点的に分析し、
テキスト情報と統合したマインドマップを作成してください。

【視覚情報分析】
1. **図表の詳細読み取り**: グラフの数値、表の項目、チャートの関係性
2. **イラスト・写真の説明**: 視覚的に示される概念や手順
3. **レイアウト情報**: ページ構成や強調表示の意図
4. **色彩・形状の意味**: 視覚的な分類や重要度の表現

【統合手法】
- 視覚情報を文章として自然に表現
- 図表データの表形式化
- イラストで示される概念の言語化
- レイアウトが示す階層関係の反映

【出力特徴】
- 視覚情報の詳細な言語化
- 図表データの構造化表示
- 画像とテキストの自然な統合
- 視覚的階層の論理的整理

画像の全ての情報を活用した包括的なマインドマップを生成してください。"""
    }
    return prompts


def compress_and_base64_for_technical(image):
    """
    技術書籍用の圧縮（OCR可読性重視）
    """
    quality = INITIAL_QUALITY
    while True:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= MAX_FILE_SIZE or quality <= 60:  # OCR用に品質下限を高めに設定
            break
        quality -= 5
    
    base64_str = base64.b64encode(data).decode("utf-8")
    print(f"最終画像サイズ: {len(data)} bytes, 品質: {quality}")
    return base64_str


def run_ollama_image_to_text(base64_image):
    """
    画像認識 → テキスト化
    """
    prompt_text = """
あなたは画像認識に長けたアシスタントです。

以下の2段階のタスクを行ってください：

---

【Step1：自然文としての情報整形】

まず、画像から読み取れる内容を以下の指針に基づいて**日本語の説明文**として自然に整形してください。

▼出力指針：
- 文章の構造をできるだけ保持してください。
- OCR結果はそのままではなく、助詞や接続詞を補って自然な日本語に整えてください。
- 図や表の内容も、自然文として言語化してください。
- 背景、因果関係、プロセスなど、文脈が読み取れる場合は補ってください。
- 箇条書きや記号、表などは、そのままではなく**文章として再構成**してください。
- 用語は必要に応じて説明を補足してください。

▼出力形式：
- 説明文形式（段落構造）
- 教科書やナレーションのような自然で平易な日本語

---

【Step2：JSON形式での構造化出力】

次に、Step1で生成した自然文をもとに、以下の構造に従って**情報をJSON形式**で整理してください：

```json
{
  "見出し": "メインタイトルや章タイトル",
  "キーワード": ["重要なキーワード1", "重要なキーワード2", "重要なキーワード3"],
  "関係性": "概念間の関係性や構造の説明（因果関係や順序など）",
  "詳細内容": "本文の要約や詳細な説明（自然文ベースで）",
  "図表情報": "図・表・フローチャートなどの情報を説明文形式で",
  "階層構造": "見出し・小見出し・箇条などの階層関係を自然言語で記述"
}
"""

    payload = {
        "model": OLLAMA_IMAGE_MODEL,
        "prompt": prompt_text,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.85,
            "num_predict": 2048,
            "repeat_penalty": 1.1,
            "top_k": 35,
            "num_ctx": 4096
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=REQUEST_TIMEOUT,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response']
            else:
                print(f"予期しない応答形式: {result}")
                return None
        else:
            print(f"HTTPエラー {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"タイムアウトしました（{REQUEST_TIMEOUT}秒）")
        return None
    except Exception as e:
        print(f"API実行エラー: {e}")
        return None


# ===== 追加部分2: JSON解析と文章生成関数 =====

def parse_json_content(json_text):
    """
    JSON形式のテキストを解析し、全キーと値を取得
    """
    try:
        import json
        # JSON文字列から辞書に変換を試行
        if isinstance(json_text, str):
            # JSON部分を抽出（```json...```やその他の装飾を除去）
            lines = json_text.strip().split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if '{' in line or in_json:
                    in_json = True
                    json_lines.append(line)
                    if '}' in line and line.strip().endswith('}'):
                        break
            
            if json_lines:
                json_str = '\n'.join(json_lines)
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    # JSON解析に失敗した場合は、テキストから手動で情報を抽出
                    return extract_info_from_text(json_text)
            else:
                return extract_info_from_text(json_text)
        else:
            return json_text
    except Exception as e:
        print(f"JSON解析エラー: {e}")
        return extract_info_from_text(json_text)


def extract_info_from_text(text):
    """
    JSON形式でない場合の情報抽出
    """
    return {
        "全体内容": text,
        "抽出方法": "テキスト全体からの情報抽出"
    }


def run_ollama_json_to_summary(json_data):
    """
    JSON形式のデータから要約文章を生成
    """
    # JSONデータから情報を構造化テキストに変換
    structured_text = format_json_to_text(json_data)
    
    combined_prompt = f"{SUMMARY_PROMPT}\n\n{structured_text}"
    
    payload = {
        "model": OLLAMA_TEXT_MODEL,
        "prompt": combined_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # 要約用に少し創造性を上げる
            "top_p": 0.9,
            "num_predict": 4096,  # 長い要約に対応
            "repeat_penalty": 1.1,
            "top_k": 40,
            "num_ctx": 4096
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=REQUEST_TIMEOUT,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response']
            else:
                print(f"予期しない応答形式: {result}")
                return None
        else:
            print(f"HTTPエラー {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"タイムアウトしました（{REQUEST_TIMEOUT}秒）")
        return None
    except Exception as e:
        print(f"API実行エラー: {e}")
        return None


def format_json_to_text(json_data):
    """
    JSON形式のデータを構造化テキストに変換
    """
    if isinstance(json_data, dict):
        formatted_parts = []
        for key, value in json_data.items():
            if isinstance(value, list):
                value_str = "、".join(str(v) for v in value)
                formatted_parts.append(f"【{key}】: {value_str}")
            elif isinstance(value, dict):
                nested_parts = []
                for k, v in value.items():
                    nested_parts.append(f"{k}: {v}")
                value_str = " / ".join(nested_parts)
                formatted_parts.append(f"【{key}】: {value_str}")
            else:
                formatted_parts.append(f"【{key}】: {value}")
        
        return "\n\n".join(formatted_parts)
    else:
        return str(json_data)


def save_summary_to_file(content, output_path):
    """
    要約文章をファイルに保存
    """
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"""<!-- Generated by Enhanced Technical Summary Generator -->
<!-- Created: {timestamp} -->
<!-- Format: Plain Text Summary -->

"""
        full_content = header + content
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        print(f"✓ 要約文章を保存しました: {output_path}")
        return True
    except Exception as e:
        print(f"✗ ファイル保存エラー: {e}")
        return False


def generate_summary_filename(output_file=None, prefix=None):
    """
    要約文章用の出力ファイル名を生成
    """
    if output_file:
        # 拡張子を.txtに変更
        base_filename = os.path.splitext(output_file)[0] + "_summary.txt"
    else:
        base_filename = f"enhanced_summary_{int(time.time())}.txt"

    if prefix:
        filename_parts = os.path.splitext(base_filename)
        base_name = filename_parts[0]
        extension = filename_parts[1] if filename_parts[1] else '.txt'
        prefixed_filename = f"{prefix}_{base_name}{extension}"
        return prefixed_filename
    else:
        return base_filename


def run_ollama_text_to_mindmap(text_input, prompt_text):
    """
    テキスト → マインドマップ生成
    """
    combined_prompt = f"""以下のテキスト情報をもとに、マインドマップ形式のMarkdownを生成してください：

入力テキスト：
{text_input}

{prompt_text}"""
    
    payload = {
        "model": OLLAMA_TEXT_MODEL,
        "prompt": combined_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.85,
            "num_predict": 3072,
            "repeat_penalty": 1.1,
            "top_k": 35,
            "num_ctx": 4096
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=REQUEST_TIMEOUT,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response']
            else:
                print(f"予期しない応答形式: {result}")
                return None
        else:
            print(f"HTTPエラー {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"タイムアウトしました（{REQUEST_TIMEOUT}秒）")
        return None
    except Exception as e:
        print(f"API実行エラー: {e}")
        return None


def run_ollama_for_enhanced_mindmap(prompt_text, base64_image):
    """
    強化されたマインドマップ生成用のOllama API実行
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.1,  # より一貫性重視
            "top_p": 0.85,
            "num_predict": 3072,  # より長い出力に対応
            "repeat_penalty": 1.1,
            "top_k": 35,
            "num_ctx": 4096  # コンテキスト長を増加
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=REQUEST_TIMEOUT,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response']
            else:
                print(f"予期しない応答形式: {result}")
                return None
        else:
            print(f"HTTPエラー {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"タイムアウトしました（{REQUEST_TIMEOUT}秒）")
        return None
    except Exception as e:
        print(f"API実行エラー: {e}")
        return None


def check_ollama_status():
    """
    Ollamaサーバーの状態チェック
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            if OLLAMA_MODEL in model_names:
                print(f"✓ Ollamaサーバー起動中、{OLLAMA_MODEL}モデル利用可能")
                return True
            else:
                print(f"✗ {OLLAMA_MODEL}モデルが見つかりません")
                print(f"利用可能なモデル: {model_names}")
                return False
        else:
            return False
    except Exception as e:
        print(f"✗ Ollamaサーバーに接続できません: {e}")
        return False


def get_image_files(input_spec):
    """
    画像ファイルリストを取得（順序を保持）
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    if isinstance(input_spec, list):
        return input_spec
    
    if isinstance(input_spec, str):
        if os.path.isdir(input_spec):
            image_files = []
            for ext in image_extensions:
                pattern = os.path.join(input_spec, f"*{ext}")
                files = glob.glob(pattern)
                image_files.extend(files)
                pattern = os.path.join(input_spec, f"*{ext.upper()}")
                files = glob.glob(pattern)
                image_files.extend(files)
            # 重複除去しつつ順序保持
            seen = set()
            unique_files = []
            for f in sorted(image_files):
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            return unique_files
        
        elif '*' in input_spec or '?' in input_spec:
            return sorted(glob.glob(input_spec))
        
        else:
            return [input_spec] if os.path.exists(input_spec) else []
    
    return []


def generate_output_filename(output_file=None, prefix=None):
    """
    出力ファイル名を生成（プレフィックス対応）
    """
    if output_file:
        # ユーザー指定のファイル名
        base_filename = output_file
    else:
        # デフォルトファイル名
        base_filename = f"enhanced_mindmap_{int(time.time())}.md"

    if prefix:
        # プレフィックスが指定されている場合
        # ファイル名と拡張子を分離
        filename_parts = os.path.splitext(base_filename)
        base_name = filename_parts[0]
        extension = filename_parts[1] if filename_parts[1] else '.md'
        
        # プレフィックスを追加
        prefixed_filename = f"{prefix}_{base_name}{extension}"
        return prefixed_filename
    else:
        return base_filename


def save_mindmap_to_file(content, output_path="enhanced_mindmap_output.md"):
    """
    マインドマップをファイルに保存（メタデータ付き）
    """
    try:
        # ヘッダー情報を追加
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"""<!-- Generated by Enhanced Technical Mindmap Generator -->
<!-- Created: {timestamp} -->
<!-- Format: Markdown Mindmap with OCR Integration -->

"""
        full_content = header + content
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        print(f"✓ 強化マインドマップを保存しました: {output_path}")
        return True
    except Exception as e:
        print(f"✗ ファイル保存エラー: {e}")
        return False


def generate_enhanced_technical_mindmap(image_input, layout='vertical', prompt_type='enhanced_mindmap', 
                                      output_file=None, custom_prompt=None, prefix=None):
    """
    強化されたマインドマップ生成メイン処理
    """
    print("=== 強化版マインドマップ生成ツール ===")
    print("OCR理解・階層構造・表形式対応版")
    
    if not check_ollama_status():
        return None
    
    image_paths = get_image_files(image_input)
    if not image_paths:
        print("エラー: 有効な画像ファイルが見つかりませんでした")
        return None
    
    print(f"処理対象ページ: {len(image_paths)}枚")
    for i, path in enumerate(image_paths, 1):
        print(f"  ページ{i}: {path}")
    
    try:
        print(f"\n画像の合成を開始... (レイアウト: {layout})")
        combined_image = combine_images_for_technical_book(image_paths, layout)
        
        print("Base64変換中...")
        base64_img = compress_and_base64_for_technical(combined_image)
        
        # プロンプト選択
        if custom_prompt:
            prompt = custom_prompt
            print("カスタムプロンプトを使用")
        else:
            prompts = create_enhanced_mindmap_prompts()
            prompt = prompts.get(prompt_type, prompts['enhanced_mindmap'])
            print(f"使用プロンプト: {prompt_type}")
        
        print("強化マインドマップ生成中...")
        print("- OCR理解によるテキスト抽出")
        print("- 図表情報の統合")
        print("- 階層構造の自動整理")
        print("- 表形式データの最適化")

        # 3段階処理：画像→テキスト→マインドマップ
        print("\n=== 3段階処理開始 ===")
        print("段階1: 画像認識 → テキスト化")
        extracted_text = run_ollama_image_to_text(base64_img)
        
        if not extracted_text:
            print("画像からのテキスト抽出に失敗しました")
            return None

        print("段階1完了 - 抽出されたテキスト:")
        print("-" * 50)
        print(extracted_text)
        print("-" * 50)
        
        # JSONデータを解析
        print("段階1-2: JSON解析と要約文章生成")
        json_data = parse_json_content(extracted_text)
        print("解析されたJSONデータ構造:")
        if isinstance(json_data, dict):
            for key in json_data.keys():
                print(f"  - {key}")
        
        # JSON情報から要約文章を生成
        summary_text = run_ollama_json_to_summary(json_data)
        
        if summary_text:
            print("\n" + "="*60)
            print("【JSONデータから生成された要約文章】")
            print("="*60)
            print(summary_text)
            print("="*60)
            
            # 要約文章をファイルに保存
            summary_output_path = generate_summary_filename(output_file, prefix)
            save_summary_to_file(summary_text, summary_output_path)
        else:
            print("要約文章の生成に失敗しました")

        print("段階2: テキスト → マインドマップ生成")
        response_jsn = run_ollama_text_to_mindmap(extracted_text, prompt)
    
        if not response_jsn:
            print("テキストからのマインドマップ生成に失敗しました")
            return None
        else:
            print("\n" + "="*50)
            print("【テキストから生成されたマインドマップ】")
            print("="*50)
            print(response_jsn)
            print("="*50)

            # ファイル保存（プレフィックス対応）
            output_path = generate_output_filename(output_file, prefix)
            save_mindmap_to_file(response_jsn, output_path)
        
        print("段階3: 画像 → マインドマップ生成")
        response = run_ollama_for_enhanced_mindmap(prompt, base64_img)
        
        if response:
            print("\n" + "="*70)
            print("【生成された強化マインドマップ】")
            print("="*70)
            print(response)
            print("="*70)
            
            # ファイル保存（プレフィックス対応）
            output_path = generate_output_filename(output_file, prefix)
            save_mindmap_to_file(response, output_path)
            
            return response
        else:
            print("強化マインドマップの生成に失敗しました")
            return None
        
    except Exception as e:
        print(f"処理エラー: {e}")
        return None


# メイン実行部分
if __name__ == "__main__":
    print("=== 強化版マインドマップ生成ツール ===")
    print("OCR理解・階層構造・表形式対応版\n")
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python script.py <画像パス> [オプション]")
        print("\nオプション:")
        print("  --layout vertical|horizontal     画像配置 (デフォルト: vertical)")
        print("  --prompt enhanced|comprehensive|section|visual  プロンプト種類")
        print("  --output <ファイル名>           出力ファイル名")
        print("  --prefix <プレフィックス>       出力ファイル名の先頭に付ける文字列")
        print("  --custom '<カスタムプロンプト>' カスタムプロンプト")
        print("\nプロンプト種類:")
        print("  enhanced     : OCR理解・階層構造・表形式対応（デフォルト）")
        print("  comprehensive: 包括的分析（学習効果重視）")
        print("  section      : セクション構造化（詳細分析）")
        print("  visual       : 視覚情報統合（図表重視）")
        print("\n例:")
        print("  python script.py ./pages/ --layout vertical --prompt enhanced")
        print("  python script.py '*.png' --output tech_mindmap.md --prompt comprehensive")
        print("  python script.py ./book_pages/ --layout horizontal --prompt visual")
        print("  python script.py ./pages/ --prefix chapter1 --output mindmap.md")
        print("    → 出力ファイル名: chapter1_mindmap.md")
        sys.exit(1)
    
    image_input = sys.argv[1]
    
    # オプション解析
    layout = 'vertical'
    prompt_type = 'enhanced_mindmap'
    output_file = None
    custom_prompt = None
    prefix = None
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--layout' and i + 1 < len(sys.argv):
            layout = sys.argv[i + 1]
            i += 2
        elif arg == '--prompt' and i + 1 < len(sys.argv):
            prompt_type = sys.argv[i + 1]
            i += 2
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg == '--prefix' and i + 1 < len(sys.argv):
            prefix = sys.argv[i + 1]
            i += 2
        elif arg == '--custom' and i + 1 < len(sys.argv):
            custom_prompt = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    print(f"入力: {image_input}")
    print(f"レイアウト: {layout}")
    print(f"プロンプト種類: {prompt_type}")
    if output_file:
        print(f"出力ファイル: {output_file}")
    if prefix:
        print(f"ファイル名プレフィックス: {prefix}")
        # 最終的な出力ファイル名を表示
        final_output = generate_output_filename(output_file, prefix)
        print(f"最終出力ファイル名: {final_output}")
    if custom_prompt:
        print(f"カスタムプロンプト使用")
    
    result = generate_enhanced_technical_mindmap(
        image_input, 
        layout=layout,
        prompt_type=prompt_type,
        output_file=output_file,
        custom_prompt=custom_prompt,
        prefix=prefix
    )

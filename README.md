# ローカルLLMエージェント性能評価レポート

**実施日**: 2025年10月4日  
**実験者**: Mr.Kondo  
**環境**: Google Colab (GPU T4 15.8GB)  

**Built with Llama**

---

## 1. 実験目的

量子化された小型LLM（Llama-3.2-3B-Instruct）を用いて、LangChainの異なるアーキテクチャ（Chain/Agent/Graph）におけるタスク実行性能を比較検証する。特にツール利用の有無がタスク成功率に与える影響を定量的に評価する。

---

## 2. 実験構成

### 2.1 モデル仕様
- **モデル**: meta-llama/Llama-3.2-3B-Instruct（ [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)で使用申請をした後、HF_TOKENを使ってダウンロード。）
- **量子化**: bitsandbytes 4bit NF4
- **GPU使用量**: 7.53GB (予約8.21GB)
- **生成パラメータ**: max_new_tokens=256, do_sample=False

### 2.2 評価タスク

| ID | タスク | 内容 | ツール要否 |
|----|--------|------|-----------|
| s0_format | JSON出力 | `{"ok": true}` | 不要 |
| s1_reasoning | 簡易算術 | 40+2 | 不要 |
| s3_tool | 複雑計算 | 231×47 + 5³ | **必須** |
| s4_context_qa | 日本語理解 | 古都名抽出 | 不要 |
| s5_extract_email | 情報抽出 | メールアドレス | 不要 |

各タスクを3試行、3つのRunnerで実行（計45タスク）。

### 2.3 Runner実装

#### Chain (LCEL)
- プロンプト → LLM → 直接出力
- ツール機能なし
- 最もシンプルな構成

#### Agent (カスタム実装)
- 正規表現で計算式を検出
- 検出時: calculator tool呼び出し
- 非検出時: LLM直接回答
- 最大3イテレーション、15秒制限

#### Graph (LangGraph)
- LLMノード → route関数 → tools/END
- 柔軟な終了条件（パターンマッチング）
- 再帰制限25回

---

## 3. 主要な技術的課題と解決策

### 3.1 ChatHuggingFaceラッパーの不具合

**問題**: 初期実装で全Runner成功率40%以下。LLMが指示を無視し余計な説明文を付加。

**原因**: `ChatHuggingFace`がLlama-3.2のチャット形式を正しく処理できず、プロンプトが破損。

**解決**: 
```python
# 手動でチャットテンプレート適用
def format_for_llama(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, 
                                   add_generation_prompt=True)
```

`HuggingFacePipeline`を直接使用することで安定性を確保。

### 3.2 Agent無限ループ問題

**問題**: LangChain標準の`hwchase17/react`プロンプトを使用したところ、全タスクでiteration limitに到達（成功率0%）。

**原因**: 3Bモデルには複雑なReActフォーマット（Thought/Action/Observation）の理解が困難。

**解決**: ReActフレームワークを廃止し、シンプルなパターンマッチングに置き換え：

```python
calc_pattern = r'(\d+\s*[\+\-\*/\^]\s*\d+(?:\s*[\+\-\*/\^]\s*\d+)*)'
if re.search(calc_pattern, prompt):
    expr = calc_match.group(1).strip()
    result = calculator.invoke(expr)
```

計算式を検出したら即座にツール呼び出し、それ以外はLLMに任せる二分岐戦略が有効。

### 3.3 Graph終了判定の失敗

**問題**: s0_format（JSON出力）とs4_context_qa（日本語1語）でrecursion limitエラー。

**原因**: route関数が`Final Answer:`プレフィックスのみを終了条件としており、それを含まない出力で無限ループ。

**解決**: タスク別の柔軟な終了条件を追加：

```python
cleaned = re.sub(r'[\s\n\r。、.,!？?]+', '', txt)

if txt == '{"ok": true}':  # JSON完全一致
    return END
if re.fullmatch(r'\d{1,5}', cleaned):  # 数字のみ
    return END
if '@' in txt and re.search(email_pattern, txt):  # メール検出
    return END
```

空白・句読点を除去した後のパターンマッチングで誤判定を防止。

---

## 4. 実験結果

### 4.1 最終成功率

| Runner | 成功数 | 成功率 | 平均実行時間 |
|--------|--------|--------|-------------|
| **Agent** | 15/15 | **100%** | ~1,500ms |
| **Graph** | 15/15 | **100%** | ~800ms |
| **Chain** | 12/15 | **80%** | ~400ms |
| **総合** | 42/45 | **93%** | - |

### 4.2 タスク別分析

| タスク | 成功率 | 備考 |
|--------|--------|------|
| s0_format | 100% | 全Runner成功 |
| s1_reasoning | 100% | 全Runner成功 |
| **s3_tool** | **67%** | **Chainのみ失敗** |
| s4_context_qa | 100% | 全Runner成功 |
| s5_extract_email | 100% | 全Runner成功 |

### 4.3 s3_tool（複雑計算）の詳細

| Runner | 出力 | 正解 | 結果 | ツール呼び出し |
|--------|------|------|------|---------------|
| Agent | 10982 | 10982 | ✓ | 1回 |
| Graph | 10982 | 10982 | ✓ | 1回 |
| Chain | -（token内で解答できず） | 10982 | ✗ | 0回 |

**Chain失敗の原因**:
- ツール機能を持たず、LLMが`231×47 + 5³`を暗算
- 3Bモデルでは複雑な算術計算が不可能
- プロンプト工夫（"no steps"指示）でも誤答

---

## 5. 重要な知見

### 5.1 小型LLMの限界と対策

**限界**:
- 複雑な算術計算（3桁×2桁レベル）は暗算不可能
- ReActのような多段階推論フォーマットの理解が困難
- `max_new_tokens`制限内で説明と解答を両立できない

**有効な対策**:
- 計算タスク: 正規表現で事前検出し、強制的にツール呼び出し
- プロンプト: 説明を禁止し、最終回答のみ要求
- 終了条件: タスク別のパターンマッチングで明示的に定義

### 5.2 アーキテクチャ別の適性

**Chain**:
- 最速（平均400ms）だが機能が限定的
- ツール不要タスクでは100%の成功率
- 複雑計算には不適

**Agent**:
- ツール利用により全タスク対応可能
- パターンマッチング式の実装で安定動作
- 標準ReActフレームワークは3Bモデルには過剰

**Graph**:
- 最も柔軟な制御が可能
- route関数の工夫で多様な終了条件に対応
- 実行速度とツール利用のバランスが最良

---

## 6. 結論

Llama-3.2-3B（4bit量子化）でも、適切なプロンプト設計とツール統合により93%の高い成功率を達成した。特に以下の点が明らかになった：

1. **ツールの必要性**: 複雑計算タスクではツール必須。LLMの暗算能力に依存すると失敗する。

2. **フレームワーク選択**: 小型モデルにはシンプルな実装が有効。LangChain標準のReActは過剰。

3. **明示的制御の重要性**: チャットテンプレート手動適用、パターンマッチング、タスク別終了条件など、LLMの曖昧性に頼らない設計が成功の鍵。

4. **Chain単体の限界**: ツール機能を持たないChainは、プロンプト工夫では解決できない本質的制約がある。Agent/Graphとの使い分けが重要。

本実験により、安価かつ簡便に、LLMエージェント開発における技術的知見が得られた。

---

**付録**: 完全なコードはGoogle Colab Notebookとして保存済み（results_yyyymmdd_hhmmss/にCSV出力）

---
marp: true
---

# Full-Duplex-Bench 紹介スライド アウトライン

---

## スライド 1: タイトル

**Full-Duplex-Bench: 全二重音声対話モデルの対話能力ベンチマーク**

- NTU Hung-yi Lee 研究室ら (arXiv:2503.04721, arXiv:2507.23159)
- v1.0: ターンテイキング評価 / v1.5: オーバーラップ処理評価
- GitHub: https://github.com/DanielLin94144/Full-Duplex-Bench

---

## スライド 2: 背景・モチベーション

**なぜこのベンチマークが必要か？**

- 従来の音声対話モデル評価は「応答の質」に偏りがち
- 実際の対話では **ターンテイキング**（発話交替）が重要
  - 相槌を適切に打てるか？
  - 沈黙が続いたとき勝手に話し始めないか？
  - ユーザーの割り込みに対応できるか？
- これらを **自動的・定量的** に評価する仕組みがなかった

---

## スライド 3: Full-Duplex-Bench v1.0 — 評価する4つの対話能力

| タスク | 評価内容 | 理想的な振る舞い |
|---|---|---|
| **Pause Handling** | ユーザーが一時的に沈黙した場面 | 勝手に話し始めない (TOR↓) |
| **Backchannel** | ユーザーの発話中の相槌 | 短い相槌を適切な頻度で打つ (TOR↓, Freq↑) |
| **Smooth Turn-Taking** | ユーザーの発話が終わった場面 | スムーズにターンを取る (TOR↑, Latency↓) |
| **User Interruption** | ユーザーが割り込んだ場面 | 割り込みに応じた適切な応答 (GPT-4o↑) |

- **TOR** = Turn-Over Rate（発話を取る/取らないの割合）
- タスクによって「高い方が良い」「低い方が良い」が異なる

---

## スライド 4: Full-Duplex-Bench v1.5 — オーバーラップ処理の評価

**v1.0に加えて、発話の重なりをどう処理するかを評価**

- 4つのオーバーラップシナリオ:
  - ユーザー割り込み / リスナー相槌 / サイド会話 / 環境音声
- 評価指標:
  - **Behavior** — 振る舞い分類（RESUME / STOP / YIELD など）
  - **General Before/After** — オーバーラップ前後の音声品質変化
    - UTMOS（知覚品質）、CPM（発話速度）、ピッチ、強度
  - **Timing** — 停止・応答のレイテンシ

---

## スライド 5: 日本語への適用（今回の取り組み）

**英語ベンチマークを日本語対話データ (j-chat) に適用**

- j-chatコーパス: ステレオWAV + 文字レベルタイムスタンプJSON
- 独自の前処理パイプライン:
  1. ステレオ音声をモノラルに分離（話者A/B）
  2. 対話イベントの自動検出（ポーズ、相槌、ターンテイキング等）
  3. 評価用サンプルの生成
- 日本語向けの調整:
  - 相槌パターン: 「うん」「はい」「なるほど」「へー」など
  - 文字数ベースの閾値調整（英語の単語数→日本語の文字数）
  - WPM → CPM（文字/分）

---

## スライド 6: テスト実行結果（j-chat 1ファイル）

### v1.0 評価

| タスク | 指標 | 値 | 解釈 |
|---|---|---|---|
| Pause Handling | — | サンプル0 | この対話にポーズ該当区間なし |
| Backchannel | TOR↓ | **0.00** | 相槌のみで本格発話なし → 良好 |
| | Freq↑ | **0.138** | 適度な相槌頻度 |
| Smooth Turn-Taking | TOR↑ | **1.0** | 全サンプルでターン取得 → 良好 |
| | Latency↓ | **0.0s** | 即座に応答 |
| User Interruption | GPT-4o↑ | **1.0/5** | 割り込みへの文脈対応が不適切 |

### v1.5 評価

| タスク | 指標 | 値 | 解釈 |
|---|---|---|---|
| General | UTMOS (post) | **1.36** | 音声品質: 低め（短い音声断片が多い） |
| | CPM (post) | **196.3** | オーバーラップ後の発話速度 |
| Behavior | C_RESUME | **100%** | 常に発話を再開（止まらない） |

※ 1ファイルのみのテストであり、統計的信頼性は限定的

---

## スライド 7: パイプラインの実行手順

```bash
# 1. 環境構築
conda create -n full-duplex-bench python=3.10
pip install -r requirements.txt

# 2. .envにAPIキーを設定（User Interruption / Behavior評価用）
echo 'export OPENAI_API_KEY="sk-..."' > .env

# 3. 前処理 + 評価を一括実行
source .env && bash scripts/run_jchat_eval.sh

# 前処理済みなら評価のみ
source .env && bash scripts/run_jchat_eval.sh --skip_preprocess
```

- GPU環境が必要（UTMOS等のモデル推論）
- OPENAI_API_KEY が必要（User Interruption, Behavior の GPT-4o評価）

---

## スライド 8: 今後の活用

- **自社モデルの対話能力評価**: モデル推論出力を用意すればベンチマーク可能
- **j-chatデータの拡充**: より多くの対話ファイルで安定した評価
- **モデル間比較**: 論文の結果（dGSLM, Moshi, Freeze-Omni, Gemini Live）と比較可能
- **日本語固有の課題**:
  - 相槌パターンの網羅性向上
  - JSD評価用のGT分布の構築

---

## 参考

- 論文: arXiv:2503.04721 (v1.0), arXiv:2507.23159 (v1.5)
- リポジトリ: https://github.com/DanielLin94144/Full-Duplex-Bench
- 評価対象モデル例: dGSLM, Moshi, Freeze-Omni, Gemini Live, GPT-4o Realtime

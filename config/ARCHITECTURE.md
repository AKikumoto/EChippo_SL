# EChipp_SL — マスタードキュメント
### アーキテクチャ + 実装計画（統合版）

> **注：このファイルは `config/ARCHITECTURE_ENG.md` の日本語版です。**
> 最新の変更は英語版に先に反映される。記述に矛盾がある場合は英語版を正とする。

> **このドキュメントの使い方：**
> コードを書く前に、必ず対応する論文のセクションを開いて読むこと。
> すべての数値とすべての式に引用（論文名、ページ、式番号）が付いている。
> 「なぜこの値を使うのか」を自分の言葉で説明できるようになってからコードに移ること。
> **各ステップの理解確認に答えてから次のステップに進むこと。**

---

## 1. プロジェクト目標

0. **目標は理解** — プログラムをステップごとに構築する
1. Schapiro et al. (2017) 海馬統計学習モデル (hip-sl) を PyTorch で再現する
2. MSP/TSP 解離を式と回路のレベルで理解する
3. EfAb R21 グラント（提出 2027年2月）の計算論的基盤を構築する
4. BasalGangliaACC と同じモジュール構造を使い、コードの比較・共有を容易にする

**なぜ PyTorch か？（emergent/Go ではなく）**
- emergent（元の C++ 実装）は廃止済み
- Go 再実装（schapirolab/hip-sl）は emer/emergent フレームワークが必要 — 拡張困難
- PyTorch により RSA パイプライン・SBI（BayesFlow）・BGACC コードベースとの統合が可能
- トレードオフ：真の ODE ダイナミクスなし → Euler 積分で近似（BGACC と同じ選択）

---

## 2. 対象モデル：Schapiro et al. (2017)

**論文：** Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017).
Complementary learning systems within the hippocampus. *Phil. Trans. R. Soc. B*, 372, 20160049.

**Go 再実装コード：** https://github.com/schapirolab/hip-sl

Go 再実装と元の emergent との主なパラメータ差異：
| パラメータ | emergent 元版 | Go 再実装 |
|-----------|--------------|----------|
| 抑制方式 | kWTA | FFFB |
| MSP 学習率 | 0.02 | 0.05 |
| TSP 学習率 | 0.2 | 0.4 |
| マイナスフェーズ ActM 記録 | サイクル 80 | サイクル 75 |
| プラスフェーズ ActP 記録 | サイクル 100 | サイクル 100 |

この PyTorch 実装は特に注記がない限り Go 再実装の値を採用する。

---

## 3. 全体アーキテクチャ

```
ECin ──────────────────────────────► CA1 ──► ECout
 │               [MSP：直接]          ▲
 │                                    │
 └──► DG ──────────► CA3 ────────────┘
       [TSP：エピソード的]
```

### 2つの相補的経路

| 経路 | ルート | 機能 | 学習 |
|------|--------|------|------|
| **MSP**（単シナプス経路） | ECin → CA1 直接 | 統計的規則性の学習；滑らかな重複表現 | 遅い Hebbian |
| **TSP**（三シナプス経路） | ECin → DG → CA3 → CA1 | エピソード的結合；固有のエピソードを保持 | 速い Hebbian |

**コアインサイト（Schapiro 2017 §2）：**
MSP の ECin → CA1 直接結合は、ゆっくり学習することで頻繁に共起するアイテムが類似した CA1 表現を持つよう発達する — コミュニティ構造を学習する。
TSP の DG 経由のパスはパターン分離を強制する：類似した入力でも DG では異なるパターンが活性化し、CA3 がエピソードごとに固有のアトラクターを形成できる。

> **理解確認：** なぜ MSP は遅く学習する必要があるのか？MSP が TSP と同じ速さで学習したら、コミュニティ構造学習に何が起きるか？

### 投射一覧（Schapiro 2017, Fig. 2）

| 送信側 | 受信側 | 種別 | 学習対象？ |
|--------|--------|------|----------|
| ECin | CA1 | フィードフォワード | ○（MSP） |
| ECin | DG | フィードフォワード | ○（TSP） |
| DG | CA3 | フィードフォワード | ○（TSP） |
| CA3 | CA3 | 回帰結合 | ○（TSP） |
| CA3 | CA1 | フィードフォワード | ○（TSP） |
| CA1 | ECout | フィードフォワード | ○ |
| ECout | CA1 | 逆投射 | ○（プラスフェーズ教師信号） |

---

## 4. 各層の説明

> **読む場所：** Schapiro (2017) §2.1; O'Reilly & Munakata (2000) Chapter 2

### L_ECin — 内嗅皮質入力

- ローカリスト表現：アイテムごとに 1 ユニット（n_items ユニット）
- 活動パターン：現在のアイテムのワンホット入力
- ECin 自体の学習なし；入力ドライバー
- マイナス・プラス両フェーズとも刺激パターンで固定駆動

### L_DG — 歯状回

- 役割：パターン分離 — アイテムごとの表現をできる限り異なるものにする
- 疎度：活性ユニット ~1%（他の層より大幅に疎）
- 機構：強力な kWTA — 上位 k（≈1%）のユニットのみが活性化
- 入力受信：ECin（フィードフォワード）
- 回帰結合なし
- 速い学習率（TSP 経路）

**なぜこれほど高い疎度が必要か？**
O'Reilly & Munakata (2000): 高い疎度により直交した表現が保証される → 干渉が最小化される → 各エピソードが固有の DG コードを持つ → CA3 が固有のアトラクターを形成できる（Schapiro 2017 §2.2）。

> **理解確認：** DG の疎度を 50%（CA3 と同程度）に下げたら、TSP が固有のエピソード記憶を形成する能力はどうなるか？

### L_CA3 — CA3 野

- 役割：パターン補完（アトラクターダイナミクス）
- 入力受信：DG（フィードフォワード）、CA3 自身（回帰結合）
- 回帰結合が鍵：部分的な手がかり → CA3 が活性化 → 回帰結合が完全なパターンを再現
- 疎度：~10%（DG より疎ではなく、パターン補完のための重複を許容）
- ステートフル：Euler 積分；`_activity` バッファを保持

**回帰ダイナミクス（O'Reilly & Munakata 2000, Ch. 2）：**
```
net_CA3(t) = W_DG→CA3 @ a_DG + W_CA3→CA3 @ a_CA3(t-1)
a_CA3(t)   = (1 - tau) * a_CA3(t-1) + tau * F_nxx1(net_CA3(t))
```

> **理解確認：** CA3 の回帰結合はパターン補完を可能にする。しかし新しいアイテムの最初の試行では、CA3 に保存済みパターンがない。このとき CA3 は何を出力するか？これは最初の試行の問題になるか？

### L_CA1 — CA1 野

- 役割：MSP（ECin から）と TSP（CA3 から）の信号の収束点
- 入力受信：ECin（MSP、遅い学習）、CA3（TSP、速い学習）、ECout（プラスフェーズ逆投射）
- MSP と TSP のどちらが支配的かは相対的な重みの強さと学習率に依存
- ステートフル：Euler 積分

**ネット入力：**
```
net_CA1 = W_ECin→CA1 @ a_ECin + W_CA3→CA1 @ a_CA3
```
プラスフェーズでは：ECout 逆投射も CA1 を駆動。

### L_ECout — 内嗅皮質出力

- 入力受信：CA1（フィードフォワード）
- 役割：入力パターンの再構成；CA1 へのプラスフェーズ教師信号を提供
- マイナスフェーズ：CA1 → ECout（ECout の逆投射はマイナスフェーズの活動を伝達）
- プラスフェーズ：ECout を目標アイテムのパターンにクランプ → 修正信号を提供

---

## 5. 活性化関数

> **読む場所：** O'Reilly & Munakata (2000) Chapter 2, Equations 2.11–2.14

### nxx1（NoisyXX1）

BasalGangliaACC・Frank モデルと同じ活性化関数。Leabra フレームワーク。

$$y_j = \frac{1}{1 + \frac{1}{\gamma \cdot [V_m - \theta]_+}}$$

- $\gamma$（gain）：シグモイドの急峻さ。デフォルト：600
- $\theta$（threshold）：発火閾値。デフォルト：0.25
- $[x]_+$：整流 — $x < 0$ のとき 0、それ以外は $x$

**デフォルトパラメータ（Schapiro 2017 は Leabra デフォルト値に従う）：**
| パラメータ | 値 | 出典 |
|-----------|-----|------|
| gain $\gamma$ | 600 | O'Reilly & Munakata (2000) |
| threshold $\theta$ | 0.25 | O'Reilly & Munakata (2000) |

注：BasalGangliaACC と異なり DA による gain 変調はない。gain と threshold は固定。

> **理解確認：** Vm = 0.25（ちょうど閾値）のとき nxx1 の出力は？Vm = 0.3 のときは？

### kWTA（k-Winners-Take-All）抑制

側方抑制：上位 k ユニットのみが活性を維持し、残りは抑制される。

| 層 | k 割合 | 疎度 |
|----|--------|------|
| DG | ~0.01 | ~1% |
| CA3 | ~0.10 | ~10% |
| CA1 | ~0.10 | ~10% |
| ECout | ~0.10 | ~10% |

---

## 6. CHL 学習則

> **読む場所：** O'Reilly & Munakata (2000) Chapter 4（Contrastive Hebbian Learning）; Schapiro (2017) §2.3

Contrastive Hebbian Learning（CHL）：Leabra のコア学習機構。

**1試行あたり2つのフェーズ：**

```
マイナスフェーズ：ECin 活性 → CA1 が MSP + TSP の予測で定常状態へ
                 ECout が CA1 の予測出力に定常状態（クランプなし）

プラスフェーズ：  ECin 活性（同じ）+ ECout を目標アイテムにクランプ
                 CA1 が目標に向けて再定常化；逆投射で修正
                 （プラスフェーズはマイナスフェーズの最終状態から継続；フェーズ間で reset() を呼ばない）
```

**重み更新：**

$$\Delta W_{ij} = \alpha \cdot (\hat{y}_j^+ \cdot \hat{y}_i^+ - \hat{y}_j^- \cdot \hat{y}_i^-)$$

- $\hat{y}^+$：プラスフェーズの活動
- $\hat{y}^-$：マイナスフェーズの活動
- $\alpha$：学習率（MSP = 0.05、TSP = 0.4；Go 再実装より）

**なぜ TSP は MSP より速く学習するのか：**
MSP は統計的規則性を蓄積するためにゆっくり学習しなければならない — 速い学習では各エピソードが前のものを上書きしてしまう。TSP は忘れられる前に個別のイベントを素早く結合する必要がある（Schapiro 2017 §2.3）。

> **理解確認：** なぜ TSP は MSP より速い学習率が必要か？TSP が MSP と同じ速さで学習したら、パターン補完に何が起きるか？

---

## 7. 統計学習課題

> **読む場所：** Schapiro (2017) §2.4, Fig. 1

### コミュニティグラフ構造

コミュニティ内遷移確率が高いグラフ上のアイテム：
- アイテム数：15（5コミュニティ × 3アイテム；設定変更可能）
- コミュニティ内遷移：高確率
- コミュニティ間（ボトルネック）遷移：低確率
- 各提示：現在アイテム → 次アイテムのペア

**学習手順：**
- コミュニティグラフ上のランダムウォーク
- 各ステップ：現在アイテムを ECin 入力として提示
- 目標：次アイテムを ECout 教師信号として提示
- CHL 実行（マイナスフェーズ → プラスフェーズ → 重み更新）

**期待される結果（Schapiro 2017 Fig. 3–4）：**
- 学習後、CA1 表現がコミュニティ単位でクラスタリング
- MSP：コミュニティ内で重複した滑らかな表現
- TSP：エピソードごとに明確な境界を持つ表現
- パターン補完（部分的な手がかり → CA3 が補完 → CA1 正解）

### 主要な評価指標

- **RSA**：CA1 ペアワイズ類似度行列 — コミュニティ内ペアはコミュニティ間より類似度が高いはず
- **パターン補完**：部分的な手がかり（アイテム特徴の 50%）→ CA3 が補完 → CA1 の正解率
- **コミュニティクラスタリング**：CA1 表現の t-SNE または PCA（コミュニティで色分け）

---

## 8. Euler 積分ポリシー

試行内ダイナミクスに参加するすべての settling 層は**ステートフル** — Euler 積分により `_activity` バッファをサイクルをまたいで保持する（BGACC と同様）。

```python
# Euler 更新（全 settling 層）
self._activity = (1 - self.tau) * self._activity + self.tau * new_act
```

**デフォルト tau：** 0.1（Leabra デフォルト；O'Reilly & Munakata 2000）

**reset ポリシー：**
- `reset()` は試行開始時に 1 回だけ呼ぶ（マイナスフェーズ前）
- プラスフェーズはマイナスフェーズの最終状態から継続 — フェーズ間で `reset()` を呼ばない
- `use_euler: bool = True` フラグ；`use_euler=False` でユニットテスト用のステートレス動作に戻る

**Euler 積分が必要な層：** `L_DG`、`L_CA3`、`L_CA1`、`L_ECout`
**ステートレスな層：** `L_ECin`（入力ドライバーのみ）

---

## 9. 実装計画（10ステップ）

> **ルール：**
> - 各ステップを独立したクラスまたは関数として書く
> - 書いたらすぐに対応するテストノートブックを書き、手計算で検証する
> - 「なぜこのアプローチか」を論文引用付きでコメントに残す（例：`# Schapiro 2017 p.X Eq.Y`）
> - 次のステップに進む前に各ステップの理解確認に答える

| ステップ | 構築するもの | ファイル | テスト | 状態 |
|---------|-------------|---------|--------|------|
| **1** | `F_nxx1`、`F_kWTA` 活性化関数 | `src/util.py` | `notebook/test_nxx1.ipynb` | 未着手 |
| **2** | `L_ECin`、`L_ECout` | `src/layer.py` | `notebook/test_layers.ipynb` | 未着手 |
| **3** | `L_DG`（疎な kWTA、パターン分離） | `src/layer.py` | `notebook/test_layers.ipynb` | 未着手 |
| **4** | `L_CA3`（回帰アトラクター、Euler） | `src/layer.py` | `notebook/test_layers.ipynb` | 未着手 |
| **5** | `L_CA1`（MSP + TSP 収束、Euler） | `src/layer.py` | `notebook/test_layers.ipynb` | 未着手 |
| **6** | `CommunityGraphEnv`、`CommunityGraphDataset` | `src/tasks.py` | `notebook/test_task.ipynb` | 未着手 |
| **7** | `M_HipSL` 組み立て + CHL 学習ループ | `src/model.py` | `notebook/test_full_model.ipynb` | 未着手 |
| **8** | Schapiro 2017 結果の再現（RSA、パターン補完） | notebooks | `notebook/test_full_model.ipynb` | 未着手 |
| **9** | `M_HipSL_SR`：SR(t) 拡張 | `src/model.py` | `notebook/test_sr.ipynb` | 未着手 |
| **10** | EfAb 行動指標（n_dynamic/n_stable、べき乗則） | `src/` | `notebook/test_sr.ipynb` | 未着手 |

---

## 10. 課題スイート

| 課題 | 目的 | 状態 | 対応ステップ |
|------|------|------|------------|
| コミュニティグラフ（Schapiro 2017） | MSP/TSP 解離の再現 | 未着手 | ステップ 6 |
| パターン補完テスト | CA3 アトラクターダイナミクスの検証 | 未着手 | ステップ 7–8 |
| コミュニティ RSA | CA1 表現のコミュニティクラスタリング | 未着手 | ステップ 8 |
| EfAb 課題（ルールベースアクション選択） | SR(t) 拡張の検証 | 未着手 | ステップ 9–10 |

---

## 11. ファイル構造

```
EChipp_SL/
├── src/
│   ├── __init__.py
│   ├── util.py       F_nxx1, F_kWTA, F_fffb（ステップ 1）
│   ├── layer.py      L_ECin, L_ECout, L_DG, L_CA3, L_CA1（ステップ 2–5）
│   ├── model.py      M_HipSL（ステップ 7–8）；M_HipSL_SR（ステップ 9）
│   └── tasks.py      CommunityGraphEnv, CommunityGraphDataset（ステップ 6）
├── notebook/
│   ├── test_nxx1.ipynb           ステップ 1 検証
│   ├── test_layers.ipynb         ステップ 2–5 検証
│   ├── test_task.ipynb           ステップ 6 検証
│   ├── test_full_model.ipynb     ステップ 7–8：Schapiro 2017 再現
│   └── test_sr.ipynb             ステップ 9–10：SR(t) 拡張
├── simulations/
├── scripts/
├── trained_models/
├── visualizations/
├── manuscript/
├── config/
│   ├── ARCHITECTURE.md     （このファイル — 日本語版）
│   ├── ARCHITECTURE_ENG.md （英語版マスタードキュメント）
│   └── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## 12. PyTorch の制限と回避策

| 制限 | 影響 | 回避策 |
|------|------|--------|
| ODE ソルバーなし；Euler 積分のみ | 定常化ダイナミクスは近似 | 小さい tau（0.1）；n_steps 以内に収束を確認 |
| CHL には 2 回の手動フォワードパスが必要 | Hebbian 重みに `loss.backward()` 使用不可 | 手動重み更新（BGACC と同様） |
| Hard kWTA は微分不可能 | kWTA を通じた誤差逆伝播不可 | ステップ 1–8 は hard kWTA；RSA フィッティングが加わった場合は soft kWTA |
| Hopfield アトラクターの組み込みなし | CA3 の回帰結合は明示的に実装必要 | `L_CA3._activity` の明示的な回帰バッファ |

---

## 13. パラメータ表

> **出典：** Schapiro (2017)；Go 再実装（hip.go_vs_hip-sl.go_param_changes.xlsx）；O'Reilly & Munakata (2000)

| パラメータ | 値 | 層 | 出典 |
|-----------|-----|-----|------|
| nxx1 gain γ | 600 | 全層 | O'Reilly & Munakata (2000) |
| nxx1 threshold θ | 0.25 | 全層 | O'Reilly & Munakata (2000) |
| Euler tau | 0.1 | 全ステートフル層 | Leabra デフォルト |
| DG 疎度 | ~1% | DG | Schapiro (2017) §2.2 |
| CA3 疎度 | ~10% | CA3 | Schapiro (2017) |
| CA1 疎度 | ~10% | CA1 | Schapiro (2017) |
| MSP 学習率 | 0.05 | ECin→CA1 | Go 再実装 |
| TSP 学習率 | 0.4 | ECin→DG, DG→CA3, CA3→CA1 | Go 再実装 |
| アイテム数 | 15 | 課題 | Schapiro (2017) Fig. 1 |
| コミュニティ数 | 5 | 課題 | Schapiro (2017) Fig. 1 |
| コミュニティあたりアイテム数 | 3 | 課題 | Schapiro (2017) Fig. 1 |
| マイナスフェーズ settling サイクル数 | 25 | 全層 | Go 再実装 |
| プラスフェーズ settling サイクル数 | 25 | 全層 | Go 再実装 |

---

## 14. 参考文献

- Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017). Complementary learning systems within the hippocampus. *Phil. Trans. R. Soc. B*, 372, 20160049.
- O'Reilly, R. C. & Munakata, Y. (2000). *Computational Explorations in Cognitive Neuroscience*. MIT Press.
- Kikumoto, A. et al. (2025). Conjunctive representational trajectories predict power-law improvement and overnight abstraction. *Cerebral Cortex*.
- Mylonas, D. et al. (2024). Hippocampus is necessary for micro-offline gains. *J. Neurosci.*

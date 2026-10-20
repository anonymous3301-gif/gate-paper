# Appendix A. Master log items 1-25 (verbatim)

# Items 1--25 (upstream master log) -- pasted

The table below is the upstream master log's items 1--25, pasted verbatim
(pasted 2026-10-20). It is kept here as the appendix provenance block; nothing in it is paraphrased or reformatted.

Column 3 ("where it appears") is reproduced verbatim from the upstream log, so
its section numbers refer to the upstream manuscript rather than to this
version.

**Note on row 25.** Row 25 is the threshold-provenance correction, which is
**item 26 in this document** (the upstream log's item 26 was merged into it), so
that the same correction is not listed twice.

| item | 何を修正したか | 記載箇所 |
|---|---|---|
| 1 | RMSNorm が delta を抑制 → norm_3cond で delta も改善 | §5, §9.2.1 |
| 2 | hebbian > delta → 三重アーティファクト | §4, §9.2.1 |
| 3 | delta が方向を失う（cos 0.075）→ v_prev バグ | §3, §9.2.1 |
| 4 | 遷移は d_mem≈24-28 でシャープ → d_mem=24 はタイ | §4, §9.2.1 |
| 5 | 2610.09056 の理論は直交性で測る → 原典は2次モーメント | §8.3, §9.2.2 |
| 6 | delta の junk 脆弱性は decay 忘れ → 棄却。Delta > GDN at J=8-32 | §7, §9.2.2 |
| 7 | 容量がクロスオーバーを決める → 主項は正規化のスケール | §4, §9.2.2 |
| 8 | κ(G) の爆発が精度破綻を招く → κ=10⁹ でも acc 安定 | §8.3, §9.2.3 |
| 9 | κ 爆発の主因は V=64 → 支配変数は V/d_mem 比 | §8.3, §9.2.3 |
| 10 | 内在次元8 → L2 モデル固有。Kaczmarz は 26 まで成長 | §8.3, §9.2.3 |
| 11 | OSDN は対角前処理の破綻の証拠 → OSDN は成功例 | §1.6, §9.2.2 |
| 12 | n_pairs 軸は平坦 → 浅い U が残る | §8.3, §9.2.3 |
| 13 | LMS 安定性が最適βを決める → raw-key regime 限定 | §4, §9.2.4 |
| 14 | 新アーキは CPU では無理 → モデルではなく設計原理 | §1.5, §9.2.6 |
| 15 | Kaczmarz の利点は分離の利用 → 共適応の非対称性が本質 | §5, §9.2.5 |
| 16 | ゲートはテスト時の無料の再重み付け → 訓練時の inductive bias | §5, §9.2.5 |
| 17 | curriculum が正規化を代替する → raw + curriculum は 0.211 | §6, §9.2.7 |
| 18 | 平均βが性能を決める → 二重解離で棄却 | §5, §9.2.5 |
| 19 | raw k の失敗はキーノルム爆発 → E‖k‖² 同一で loss=2.36e18 | §3, §9.2.6 |
| 20 | 形状感度比で非対称性を測る → fixed の分母がゼロに潰れた副作用 | §5, §9.2.5 |
| 21 | mean corrected to training value → gate_m の平均は 0.125 に rescaled | §5, §9.2.5 |
| 22 | GDN の decay が junk 耐性を悪化させる（新規）→ 既報 | §7, §9.2.2 |
| 23 | Wv bias=True で budget_match 開始 → bias=False で再起動 | §3, §9.2.1 |
| 24 | raw-k の3相は seed 普遍 → bimodal | §3, §9.2.1 |
| 25 | 閾値0.45は完全 blind → Delta 10シードには blind、GDN v15 には non-blind | §7, §9.2.4 |

# 评测结果（Eval Results）

> 记录每一轮 eval 的准确率与改进过程。新一轮调优后在末尾 append 一节，**保留历史可对比**——这本身就是 review 材料，证明准确率是"测出来 + 迭代出来"的，不是拍脑袋。

## 评测设置

- **评测集**：`eval/labeled/` 共 200 条（10 类目 × 20），每条标注 `violation: true/false` 等字段。
- **指标**：二分类违法判定的 Precision / Recall / F1 / Accuracy（micro 平均）。
- **detector**：`src/eval/rag_detector.py` — 条款 → 本地混合检索（dense + 关键词 RRF）→ 本地 LLM 判定。
- **全程本地、私有**：embedding = bge-small-zh（ONNX，CPU）；LLM = qwen2.5:3b（Ollama，GPU）。语料与查询不出本机。
- **复现**：
  ```
  python src/rag.py index                 # 建索引（若未建）
  python scripts/run_eval.py --detector rag
  ```

---

## v1 基线 · 2026-06-25 · reason-then-conclude prompt

**MICRO AVG（200 条）：Acc 0.615 · P 0.671 · R 0.524 · F1 0.588**

| 类目 | N | P | R | F1 | Acc | TP/FP/FN/TN |
|------|---|---|----|----|----|----|
| wage_composition | 20 | 1.000 | 0.818 | 0.900 | **0.900** | 9/0/2/9 |
| probation_period | 20 | 0.889 | 0.727 | 0.800 | **0.800** | 8/1/3/8 |
| non_compete | 20 | 0.800 | 0.800 | 0.800 | **0.800** | 8/2/2/8 |
| confidentiality_ip | 20 | 0.667 | 0.250 | 0.364 | 0.650 | 2/1/6/11 |
| social_insurance | 20 | 0.636 | 0.636 | 0.636 | 0.600 | 7/4/4/5 |
| penalty_clause | 20 | 0.625 | 0.500 | 0.556 | 0.600 | 5/3/5/7 |
| termination | 20 | 0.571 | 0.400 | 0.471 | 0.550 | 4/3/6/7 |
| job_change_rights | 20 | 0.500 | 0.091 | 0.154 | 0.450 | 1/1/10/8 |
| working_hours | 20 | 0.500 | 0.500 | 0.500 | 0.400 | 6/6/6/2 |
| service_period | 20 | 0.455 | 0.455 | 0.455 | 0.400 | 5/6/6/3 |

### 读数

- **有明确数字红线**的类目（最低工资、试用期档位、竞业期限）→ 0.80–0.90，模型搞得定。
- **要靠判断、无硬数字**的类目（工时、调岗、保密）→ 0.40 上下，且专崩在**召回**（把违法看成合规、漏判）。
- 系统性短板 = **召回 0.52**：保守 prompt（"拿不准时判合规"）在所有类目都矫枉过正。
- 最严重错例：`job_change_rights` 召回 0.091（11 条违法漏 10 条）—— **待解剖**：检索没给对法条？还是模型拿到了也判错？

---

## 下一轮（待填）

- [ ] 解剖 `job_change_rights` 的 10 条漏判：逐条判定是 **retrieval 问题** 还是 **reasoning 问题**
- [ ] 召回回调：prompt 改「拿不准 → 标记 + 建议人工复核」，看 R 是否回升、Acc 是否还稳
- [ ] `chunk()` 加 `第\d+条` 分隔符，让整条法条不被切碎（很可能直接救回若干 FN）
- 结果记录：______

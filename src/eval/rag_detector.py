# -*- coding: utf-8 -*-
"""RAG-backed violation detector -- the real detector evaluated against the stubs.

For each contract clause: retrieve the most relevant labor-law passages with the
local hybrid RAG (src/rag.py), then ask the local LLM whether the clause violates
the law, grounded only in those passages. The result is parsed into a
harness.Prediction so it plugs into the existing eval the same way the stubs do.

This is the file to ITERATE on. Accuracy is won in two places:
  1. PROMPT          -- how the judgment is framed for the (small) local model.
  2. _parse_verdict  -- turning the model's free text into a clean true/false.
Improving what gets retrieved is done in rag.py (IDF_MIN, chunking).

Run:  python scripts/run_eval.py --detector rag --category probation_period
"""
import json
import os
import sys

# Put src/ on the path so we can import the sibling rag module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import rag

# The index (chunks + vectors) is loaded once and reused across every clause.
_chunks = None
_matrix = None


def _index():
    global _chunks, _matrix
    if _chunks is None:
        if not (os.path.exists(rag.CHUNKS_FILE) and os.path.exists(rag.INDEX_FILE)):
            raise FileNotFoundError("No index found. Build it first: python src/rag.py index")
        _chunks = json.load(open(rag.CHUNKS_FILE, encoding="utf-8"))
        _matrix = np.load(rag.INDEX_FILE)["matrix"]
    return _chunks, _matrix


# Reason-then-conclude. A small model that must commit to a verdict on line 1
# just anchors on its prior ("there's a law -> must be a violation") and flags
# everything. Forcing it to find the rule and compare BEFORE concluding -- and
# stating the base rate ("most clauses are compliant") -- is what gives it a
# chance to say 合规. The verdict goes on the LAST line so reasoning text above
# it is never mistaken for the verdict.
PROMPT = (
    "你是劳动法合规审查助手。只依据下面给出的法律条文，判断该合同条款是否违法。\n\n"
    "请按步骤分析：\n"
    "1. 这条条款涉及什么（如试用期长度/违约金/工时/社保）？\n"
    "2. 相关法条规定的标准或上限是什么？引用块号。\n"
    "3. 条款是否超出或违反了该标准？\n"
    "注意：大多数合同条款是合规的，只有明确违反法条规定时才判“违法”；拿不准时判“合规”。\n\n"
    "最后单独一行输出结论，格式严格为：\n"
    "结论：违法   或   结论：合规\n\n"
    "相关法律条文：\n{context}\n\n"
    "待审查条款：{clause}"
)


def _parse_verdict(raw):
    """Read the verdict from the explicit 结论 line (or the last non-empty line if
    the model omits the label). Compliant markers are checked first so '不违法' is
    not mistaken for a violation. Defaults to True (flag for review) when unclear --
    a missed violation is worse than a false alarm. Flip if you optimize precision."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    verdict = next((ln for ln in reversed(lines) if "结论" in ln), lines[-1] if lines else "")
    if any(k in verdict for k in ("合规", "合法", "不违法", "未违法", "未违反", "不构成")):
        return False
    if any(k in verdict for k in ("违法", "违反", "不合规")):
        return True
    return True  # unclear -> flag for review (favor recall)


def rag_detector(clause):
    """clause -> harness.Prediction. Registered as DETECTORS['rag'] in harness.py."""
    from eval.harness import Prediction  # lazy: avoids an import cycle with harness
    chunks, matrix = _index()
    retrieved = rag.hybrid_retrieve(clause, chunks, matrix, k=5)
    context = "\n\n".join(f"[法条{i}] {c}" for i, c in retrieved)
    raw = rag.generate(PROMPT.format(context=context, clause=clause))
    return Prediction(violation=_parse_verdict(raw), reason=raw.strip()[:200])

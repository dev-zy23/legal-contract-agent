# -*- coding: utf-8 -*-
"""Local, private RAG over Chinese labor-law texts.

Everything runs on-device: embeddings via a local ONNX model, generation via a
local Ollama model. No corpus or query ever leaves the machine -- the core
requirement for a legal/contract assistant.

Pipeline:  corpus (data/laws) -> chunk -> embed -> hybrid retrieve -> grounded answer

Usage:
    python src/rag.py index               # build the index from data/laws
    python src/rag.py ask "试用期最长多久"   # answer a legal question, with citations
"""
import os
import sys
import glob
import json
import numpy as np
import onnxruntime as ort
import jieba
import httpx
from tokenizers import Tokenizer

# --- Config -----------------------------------------------------------------
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
CORPUS_DIR  = os.path.join(ROOT, "data", "laws")          # knowledge base: labor laws (.md)
INDEX_DIR   = os.path.join(ROOT, ".index")                # build artifacts (add to .gitignore)
INDEX_FILE  = os.path.join(INDEX_DIR, "index.npz")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")

# Local bge-small-zh ONNX model. Reused from the rag-demo cache for now.
# TODO: bundle the model or document the download step to make this repo self-contained.
MODEL_DIR    = r"Y:\rag-demo\.fastembed\fast-bge-small-zh-v1.5"

OLLAMA_URL   = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

MAX_TOKENS = 512    # embedding model input limit; every chunk must stay under this
RRF_C      = 60     # Reciprocal Rank Fusion constant
RRF_POOL   = 20     # candidates each retriever contributes to fusion
IDF_MIN    = 3.0    # drop query terms below this IDF (auto stop-word removal); TUNE per corpus


# --- Embedding (local ONNX) -------------------------------------------------
_session = _embed_tok = _input_names = None

def _embedder():
    """Lazily load the ONNX embedding model and tokenizer (once)."""
    global _session, _embed_tok, _input_names
    if _session is None:
        _embed_tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        _embed_tok.enable_truncation(max_length=MAX_TOKENS)
        _session = ort.InferenceSession(
            os.path.join(MODEL_DIR, "model_optimized.onnx"),
            providers=["CPUExecutionProvider"],
        )
        _input_names = {i.name for i in _session.get_inputs()}
    return _session, _embed_tok, _input_names


def embed(texts):
    """Embed a list of texts into L2-normalized vectors of shape (n, 512)."""
    session, tok, names = _embedder()
    texts, out = list(texts), []
    for start in range(0, len(texts), 32):                # batch to keep memory low
        encs = tok.encode_batch(texts[start:start + 32])
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), maxlen), np.int64)
        mask = np.zeros((len(encs), maxlen), np.int64)
        for j, e in enumerate(encs):
            n = len(e.ids)
            ids[j, :n] = e.ids
            mask[j, :n] = e.attention_mask
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        vec = session.run(None, feeds)[0][:, 0, :]        # CLS pooling (BGE)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10)
        out.append(vec.astype(np.float32))
    return np.vstack(out)


# --- Token counting (no truncation) -----------------------------------------
_count_tok = None

def count_tokens(text):
    """True token count. The embedder's tokenizer truncates at MAX_TOKENS, so it
    would under-report length; this one does not truncate."""
    global _count_tok
    if _count_tok is None:
        _count_tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    return len(_count_tok.encode(text).ids)


# --- Corpus loading + chunking ----------------------------------------------
def read_corpus():
    """Read every .md/.txt under CORPUS_DIR. Falls back to GB18030 for non-UTF-8 files."""
    docs = []
    for path in glob.glob(os.path.join(CORPUS_DIR, "**", "*.*"), recursive=True):
        if path.lower().endswith((".txt", ".md")):
            raw = open(path, "rb").read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("gb18030", errors="replace")
            docs.append((os.path.basename(path), text))
    return docs


def split_to_units(text, seps):
    """Recursively split text into units of <= MAX_TOKENS, trying coarse separators first."""
    if count_tokens(text) <= MAX_TOKENS:                  # base case: small enough
        return [text]
    if not seps:                                          # no separators left: hard cut
        return [text[:480]]
    units = []
    for piece in text.split(seps[0]):
        if piece:
            units += split_to_units(piece, seps[1:])
    return units


def pack_units(units, max_tokens=MAX_TOKENS):
    """Greedily merge small units up to max_tokens so chunks aren't tiny fragments."""
    chunks, buf, buf_tokens = [], [], 0
    for u in units:
        ut = count_tokens(u)
        if buf and buf_tokens + ut > max_tokens:
            chunks.append("".join(buf))
            buf, buf_tokens = [], 0
        buf.append(u)
        buf_tokens += ut
    if buf:
        chunks.append("".join(buf))
    return chunks


def chunk(text):
    """Boundary-aware chunking: respect paragraph/sentence breaks, keep chunks <= MAX_TOKENS.
    TODO(legal): add "第\\d+条" as a separator so each statute article stays intact."""
    return pack_units(split_to_units(text, ["\r\n\r\n", "\r\n", "。", "，"]))


# --- Retrieval --------------------------------------------------------------
def retrieve(query, chunks, matrix, k=5):
    """Dense (semantic) retrieval: cosine similarity between query and chunk vectors."""
    scores = matrix @ embed([query])[0]                   # vectors are normalized -> dot = cosine
    idx = np.argsort(scores)[::-1][:k]
    return [(i, chunks[i]) for i in idx]


def keyword_retrieve(query, chunks, k=5):
    """Sparse (keyword) retrieval via TF-IDF over jieba tokens. Catches rare exact terms
    that dense embeddings dilute. Low-IDF query terms (stop-words) are dropped first."""
    n_docs = len(chunks)
    chunk_words = [jieba.lcut(c) for c in chunks]
    query_words = jieba.lcut(query)
    idf = {q: np.log(n_docs / (sum(1 for w in chunk_words if q in w) + 1))
           for q in set(query_words)}
    query_words = [q for q in query_words if idf[q] >= IDF_MIN]   # keep discriminating terms only
    scores = [sum(w.count(q) * idf[q] for q in query_words) for w in chunk_words]
    idx = np.argsort(scores)[::-1][:k]
    return [(i, chunks[i]) for i in idx]


def hybrid_retrieve(query, chunks, matrix, k=5):
    """Fuse dense + keyword rankings with Reciprocal Rank Fusion (RRF).
    A chunk ranked high by either retriever surfaces; high by both ranks top."""
    rrf = {}
    for ranked in (retrieve(query, chunks, matrix, RRF_POOL),
                   keyword_retrieve(query, chunks, RRF_POOL)):
        for rank, (i, _) in enumerate(ranked, start=1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_C + rank)
    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [(i, chunks[i]) for i in top]


# --- Generation (local Ollama) ----------------------------------------------
def generate(prompt):
    """Send the prompt to the local Ollama model. trust_env=False keeps the request off
    any system proxy -- a localhost call must never be routed through one."""
    resp = httpx.post(
        OLLAMA_URL, timeout=300, trust_env=False,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "30m"},
    )
    return resp.json()["response"]


def answer(query, retrieved):
    """Grounded answer: use ONLY the retrieved law text, cite the source chunk, refuse if absent.
    NOTE: this prompt is a starting point -- refine it for your legal needs
    (e.g. cite statute name + 第X条 instead of chunk ids once chunks carry that metadata)."""
    context = "\n\n".join(f"[块{i}] {c}" for i, c in retrieved)
    prompt = (
        "你是劳动法阅读助手。只能依据下面提供的法律条文回答；"
        "每条结论后用方括号标出处块号（如 [块3]）；"
        "如果提供的内容里没有答案，就回答“我不知道”，不要编造。\n\n"
        f"法律条文：\n{context}\n\n问题：{query}"
    )
    print(generate(prompt))


# --- CLI --------------------------------------------------------------------
def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = read_corpus()
    if not docs:
        print(f"No corpus found under {CORPUS_DIR}")
        return
    chunks = [c for _, text in docs for c in chunk(text)]
    print(f"{len(chunks)} chunks; embedding locally ...")
    np.savez(INDEX_FILE, matrix=embed(chunks))
    json.dump(chunks, open(CHUNKS_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Index built: {len(chunks)} chunks (fully offline).")


def do_ask(query):
    chunks = json.load(open(CHUNKS_FILE, encoding="utf-8"))
    matrix = np.load(INDEX_FILE)["matrix"]
    answer(query, hybrid_retrieve(query, chunks, matrix, k=5))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "index":
        build_index()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        do_ask(sys.argv[2])
    else:
        print('Usage:  python src/rag.py index  |  python src/rag.py ask "<question>"')

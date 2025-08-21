# -*- coding: utf-8 -*-
"""
RAG (streaming ingest, memory-friendly)
- 流式切片：不把整本书读入内存
- 批量嵌入：小批量 encode，立刻写入索引
- 元数据落盘：SQLite 存文本与meta，查询时再按rowid取回
- 索引：优先 FAISS（内积=余弦），装不上自动回退NumPy

用法：
  # 下载示例并索引
  python rag.py download-sample --out docs/book.txt
  python rag.py ingest --input docs --size 800 --overlap 80 --batch-size 96

  # 提问
  python rag.py query --q "这本书的核心主题是什么？" --k 6

  # 查看/清空
  python rag.py stats
  python rag.py reset
"""
import os
import sys
import json
import argparse
import sqlite3
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable

import re
import numpy as np
from dotenv import load_dotenv

# ---- 可选：FAISS。装不上会自动回退到 NumPy 后端 ----
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

from pypdf import PdfReader
from openai import OpenAI

# ---------------- 基本常量 ----------------
SUPPORT_SUFFIXES = {".txt", ".md", ".pdf"}
INDEX_DIR = Path("./index_data")
INDEX_DIR.mkdir(exist_ok=True)
FAISS_PATH = INDEX_DIR / "vectors.faiss"
NUMPY_PATH = INDEX_DIR / "vectors.npz"
DB_PATH    = INDEX_DIR / "store.sqlite"
META_PATH  = INDEX_DIR / "meta.json"   # 保存维度/后端等小信息（可选）

# ---------------- 环境变量 ----------------
load_dotenv()  # 读取 .env（OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL / EMBEDDINGS / EMBEDDING_MODEL ...）


# ======================= I/O 与切片（流式） =======================
def gather_files(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() in SUPPORT_SUFFIXES:
            return [p]
        raise ValueError(f"不支持的文件类型：{p.suffix}，仅支持 {SUPPORT_SUFFIXES}")
    if p.is_dir():
        return [q for q in p.rglob("*") if q.suffix.lower() in SUPPORT_SUFFIXES]
    raise FileNotFoundError(f"找不到路径：{p}")


def iter_text_file(path: Path, block_chars: int = 64_000) -> Iterable[str]:
    """按块读取纯文本文件（每次返回最多 block_chars 个字符）"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            s = f.read(block_chars)
            if not s:
                break
            yield s


def iter_pdf_pages(path: Path) -> Iterable[str]:
    """逐页读取 PDF 文本"""
    reader = PdfReader(str(path))
    for p in reader.pages:
        yield (p.extract_text() or "")


def iter_chunks_from_stream(stream: Iterable[str], size: int, overlap: int) -> Iterable[str]:
    """把任意文本流切成 size/overlap 的块（不一次性加载全部）"""
    buf = ""
    for piece in stream:
        if not piece:
            continue
        # 简单清洗，避免多余空白
        piece = re.sub(r"\s+\n", "\n", piece)
        buf += piece
        while len(buf) >= size:
            yield buf[:size]
            buf = buf[size - overlap:]
    if buf:
        yield buf  # 收尾


def iter_file_chunks(path: Path, size: int, overlap: int) -> Iterable[str]:
    suf = path.suffix.lower()
    if suf in {".txt", ".md"}:
        return iter_chunks_from_stream(iter_text_file(path), size, overlap)
    elif suf == ".pdf":
        return iter_chunks_from_stream(iter_pdf_pages(path), size, overlap)
    else:
        raise ValueError(f"不支持的文件类型：{suf}")


# ======================= 嵌入器 =======================
@dataclass
class EmbeddingCfg:
    mode: str = os.getenv("EMBEDDINGS", "openai")  # openai | local
    name: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


class BaseEmbedder:
    def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OnlineEmbedder(BaseEmbedder):
    def __init__(self, name: str):
        base = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
        key  = os.getenv("EMBEDDING_API_KEY")  or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("在线嵌入需要 OPENAI_API_KEY（或 EMBEDDING_API_KEY）")
        self.client = OpenAI(api_key=key, base_url=base)
        self.name = name

    def encode(self, texts: List[str]) -> List[List[float]]:
        out = []
        B = 64  # 单批次条数；如需更省内存可调小
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            resp = self.client.embeddings.create(model=self.name, input=batch)
            out.extend([d.embedding for d in resp.data])
        arr = np.asarray(out, dtype=np.float32)
        # 归一化 -> 内积=余弦
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr.tolist()


class LocalEmbedder(BaseEmbedder):
    def __init__(self, name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("本地嵌入需要安装 sentence-transformers；请改用在线嵌入或安装依赖") from e
        self.model = SentenceTransformer(name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        arr = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return arr.tolist()


def build_embedder() -> BaseEmbedder:
    cfg = EmbeddingCfg()
    if cfg.mode.lower() == "local":
        return LocalEmbedder(os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))
    return OnlineEmbedder(cfg.name)


# ======================= 文本与元数据存储（SQLite） =======================
class DocStore:
    """SQLite 存放切片文本与meta，rowid 与 向量index 顺序一一对应"""
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                text TEXT NOT NULL,
                meta TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM docs")
        (n,) = cur.fetchone()
        return int(n)

    def add_batch(self, texts: List[str], metas: List[dict]):
        """插入一批文本，meta以JSON存储；顺序即rowid顺序"""
        rows = [(t, json.dumps(m, ensure_ascii=False)) for t, m in zip(texts, metas)]
        self.conn.executemany("INSERT INTO docs(text, meta) VALUES (?, ?)", rows)
        self.conn.commit()

    def fetch_by_rowids(self, rowids: List[int]) -> List[Tuple[str, dict]]:
        """按 rowid 批量取回（结果与传入顺序一致）"""
        if not rowids:
            return []
        # 构造占位符
        ph = ",".join("?" for _ in rowids)
        cur = self.conn.execute(f"SELECT rowid, text, meta FROM docs WHERE rowid IN ({ph})", rowids)
        got = {rid: (txt, json.loads(meta)) for rid, txt, meta in cur.fetchall()}
        return [got[rid] for rid in rowids]

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# ======================= 向量索引（FAISS/NumPy） =======================
class VectorIndex:
    """自动选择 FAISS/NumPy 的统一接口。索引内的向量顺序 == SQLite rowid-1"""
    def __init__(self):
        self.backend = None
        self.dim = None

    # ---------- 持久化 ----------
    def save(self):
        info = {"backend": self.backend, "dim": self.dim}
        META_PATH.write_text(json.dumps(info), encoding="utf-8")
        if self.backend == "faiss":
            faiss.write_index(self.faiss_index, str(FAISS_PATH))
        elif self.backend == "numpy":
            np.savez_compressed(NUMPY_PATH, matrix=self.numpy_matrix)

    def load(self):
        if META_PATH.exists():
            info = json.loads(META_PATH.read_text(encoding="utf-8"))
            self.backend = info.get("backend")
            self.dim = info.get("dim")
        if HAVE_FAISS and FAISS_PATH.exists():
            try:
                self.faiss_index = faiss.read_index(str(FAISS_PATH))
                self.backend = "faiss"
                if not self.dim:
                    self.dim = self.faiss_index.d
                return
            except Exception:
                pass
        if NUMPY_PATH.exists():
            npz = np.load(NUMPY_PATH, allow_pickle=False)
            self.numpy_matrix = npz["matrix"]
            self.backend = "numpy"
            if not self.dim and self.numpy_matrix.size > 0:
                self.dim = self.numpy_matrix.shape[1]

    # ---------- 构建/追加 ----------
    def _build_faiss(self, X: np.ndarray):
        self.faiss_index = faiss.IndexFlatIP(X.shape[1])  # 内积（归一化后=余弦）
        self.faiss_index.add(X)
        self.backend = "faiss"
        self.dim = X.shape[1]

    def _build_numpy(self, X: np.ndarray):
        self.numpy_matrix = X
        self.backend = "numpy"
        self.dim = X.shape[1]

    def add(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError("向量维度错误")
        if self.backend is None:
            if HAVE_FAISS:
                try:
                    self._build_faiss(X)
                except Exception:
                    self._build_numpy(X)
            else:
                self._build_numpy(X)
        else:
            if self.backend == "faiss":
                self.faiss_index.add(X)
            else:
                self.numpy_matrix = np.vstack([self.numpy_matrix, X])

    # ---------- 查询 ----------
    def search(self, q: np.ndarray, k: int = 6) -> List[Tuple[float, int]]:
        if self.backend == "faiss":
            D, I = self.faiss_index.search(q.reshape(1, -1), k)
            return [(float(d), int(i)) for d, i in zip(D[0], I[0]) if i != -1]
        elif self.backend == "numpy":
            if self.numpy_matrix.size == 0:
                return []
            sims = self.numpy_matrix @ q.reshape(-1, 1)  # 内积
            sims = sims.ravel()
            k = min(k, sims.shape[0])
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            return [(float(sims[i]), int(i)) for i in idx]
        else:
            return []

    def count(self) -> int:
        if self.backend == "faiss":
            return 0 if not hasattr(self, "faiss_index") else self.faiss_index.ntotal
        if self.backend == "numpy":
            return 0 if not hasattr(self, "numpy_matrix") else self.numpy_matrix.shape[0]
        return 0


# ======================= LLM 回答 =======================
def llm_answer(question: str, contexts: List[Tuple[str, str]], model: Optional[str] = None) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
    model = model or os.getenv("OPENAI_MODEL", "qwen-plus")
    sys_prompt = (
        "你是检索增强问答助手。仅基于提供的检索片段回答；"
        "若证据不足，请回答“不确定”。在答案末尾标注引用编号（如 [S1][S2]）。"
    )
    ctx_text = "\n\n".join(f"{tag}\n{txt}" for tag, txt in contexts)
    user_prompt = f"问题：{question}\n\n=== 检索片段 ===\n{ctx_text}\n\n=== 任务 ===\n用中文回答，并标注引用编号。"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()


# ======================= 索引/查询（内存友好） =======================
def ingest_path(input_path: str, size=900, overlap=120, batch_size=96):
    """
    逐文件 -> 逐块 -> 小批量嵌入 -> 立刻写入索引与SQLite
    默认 batch_size=96；如想更保守可调小（32/64）
    """
    files = gather_files(input_path)
    if not files:
        print("[WARN] 没有可导入的 .txt/.md/.pdf")
        return {"files": 0, "chunks": 0}

    embedder = build_embedder()
    store = VectorIndex()
    store.load()
    db = DocStore()

    total_files = 0
    total_chunks = 0

    for f in files:
        chunk_idx = 0
        batch_texts, batch_metas = [], []

        # —— 核心：流式产生切片，不把整书放进内存 ——
        for ck in iter_file_chunks(f, size=size, overlap=overlap):
            batch_texts.append(ck)
            batch_metas.append({"source": f.as_posix(), "chunk": chunk_idx})
            chunk_idx += 1

            if len(batch_texts) >= batch_size:
                # 先写文本/元数据到SQLite，保持与向量顺序一致
                db.add_batch(batch_texts, batch_metas)

                # 再做嵌入并写入向量索引
                vecs = embedder.encode(batch_texts)
                X = np.asarray(vecs, dtype=np.float32)
                store.add(X)
                # 及时释放
                batch_texts, batch_metas = [], []
                gc.collect()

        # 处理残留
        if batch_texts:
            db.add_batch(batch_texts, batch_metas)
            vecs = embedder.encode(batch_texts)
            X = np.asarray(vecs, dtype=np.float32)
            store.add(X)
            batch_texts, batch_metas = [], []
            gc.collect()

        total_files += 1
        total_chunks += chunk_idx
        print(f"[OK] {f.name} -> {chunk_idx} 块")

    store.save()
    db.close()
    print(f"[DONE] 导入完成：文件 {total_files} 个，切片 {total_chunks} 条。后端：{store.backend}")
    # 粗略内存估计：index_size ≈ count * dim * 4 bytes
    if store.dim and store.count():
        approx = store.count() * store.dim * 4 / (1024**3)
        print(f"[INFO] 估算索引体积 ~ {approx:.2f} GB （不含SQLite文本）")
    return {"files": total_files, "chunks": total_chunks}


def rag_query(question: str, k: int = 6):
    embedder = build_embedder()
    store = VectorIndex()
    store.load()
    if store.count() == 0:
        print("[WARN] 索引为空，请先 ingest")
        return

    q = np.asarray(embedder.encode([question])[0], dtype=np.float32)
    hits = store.search(q, k=k)
    if not hits:
        print("[WARN] 未检索到结果")
        return

    # 将 FAISS/NumPy 的 0-based 索引映射到 SQLite 的 rowid（= idx+1）
    rowids = [idx + 1 for _, idx in hits]
    db = DocStore()
    rows = db.fetch_by_rowids(rowids)
    db.close()

    contexts = []
    for rank, ((score, _idx), (txt, meta)) in enumerate(zip(hits, rows), start=1):
        tag = f"[S{rank}] {meta.get('source')}#{meta.get('chunk')} (sim={score:.3f})"
        contexts.append((tag, txt))

    answer = llm_answer(question, contexts)
    print("===== 回答 =====\n")
    print(answer)
    print("\n===== 证据片段 =====")
    for tag, _ in contexts:
        print(tag)


def stats():
    store = VectorIndex()
    store.load()
    db = DocStore()
    cnt = store.count()
    dim = store.dim or 0
    print(f"向量条数：{cnt} | 维度：{dim} | 后端：{store.backend or 'NONE'} | 文本条数(SQLite)：{db.count()}")
    if cnt and dim:
        approx = cnt * dim * 4 / (1024**3)
        print(f"估算索引体积 ~ {approx:.2f} GB （RAM）")
    db.close()


def reset():
    for p in [FAISS_PATH, NUMPY_PATH, META_PATH, DB_PATH]:
        if p.exists():
            p.unlink()
    print("[OK] 已清空索引与SQLite。")


def download_sample(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        url = "https://www.gutenberg.org/cache/epub/24022/pg24022.txt"
        urllib.request.urlretrieve(url, str(out_path))
        print("[OK] 下载示例文本：", out_path.resolve())
    except Exception as e:
        print("[WARN] 下载失败，写入一个内置小样文本。", e)
        out_path.write_text(
            "This is a tiny sample text about GraphRAG. "
            "It extracts entities and relations into a knowledge graph, "
            "builds community summaries, and uses subgraph retrieval for complex questions.",
            encoding="utf-8"
        )


# ======================= CLI =======================
def main():
    ap = argparse.ArgumentParser(description="Memory-friendly RAG (FAISS/NumPy + SQLite + OpenAI-compatible LLM)")
    sp = ap.add_subparsers(dest="cmd")

    p_ing = sp.add_parser("ingest", help="导入并建立索引（支持目录或单文件）")
    p_ing.add_argument("--input", type=str, default="docs", help="文档目录或文件（.txt/.md/.pdf）")
    p_ing.add_argument("--size", type=int, default=800, help="切片大小（字符）")
    p_ing.add_argument("--overlap", type=int, default=80, help="切片重叠（字符）")
    p_ing.add_argument("--batch-size", type=int, default=96, help="嵌入与写库的批量大小")

    p_q = sp.add_parser("query", help="问答")
    p_q.add_argument("--q", "--question", dest="question", required=True, type=str, help="问题内容")
    p_q.add_argument("--k", type=int, default=6, help="召回数量")

    sp.add_parser("stats", help="查看索引统计")
    sp.add_parser("reset", help="清空索引与SQLite")

    p_dl = sp.add_parser("download-sample", help="下载示例 book.txt 到 docs/")
    p_dl.add_argument("--out", type=str, default="docs/book.txt")

    args = ap.parse_args()
    if args.cmd == "ingest":
        ingest_path(args.input, size=args.size, overlap=args.overlap, batch_size=args.batch_size)
    elif args.cmd == "query":
        rag_query(args.question, k=args.k)
    elif args.cmd == "stats":
        stats()
    elif args.cmd == "reset":
        reset()
    elif args.cmd == "download-sample":
        download_sample(Path(args.out))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

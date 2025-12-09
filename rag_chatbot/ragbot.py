from __future__ import annotations

import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal, Optional

import numpy as np
from dotenv import load_dotenv
from google import genai

# =========================
# Config
# =========================
load_dotenv()

Grade = Literal["cap1", "cap2", "cap3", "all"]

@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    input_dir: str
    cache_dir: str

def get_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY. Hãy set trong .env hoặc env vars.")

    return Settings(
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip(),
        top_k=int(os.getenv("TOP_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        input_dir=os.getenv("INPUT_DIR", "./input").strip(),
        cache_dir=os.getenv("CACHE_DIR", "./.rag_cache").strip(),
    )

# =========================
# Utils
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def list_files_recursive(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("."):
                continue
            out.append(os.path.join(dirpath, fn))
    return out

# =========================
# Loader (txt/md/pdf/docx optional)
# =========================
def load_text(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()

    def read_txt(p: str) -> str:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return normalize_whitespace(f.read())

    def read_pdf(p: str) -> str:
        from pypdf import PdfReader  # optional
        reader = PdfReader(p)
        pages = [(pg.extract_text() or "") for pg in reader.pages]
        return normalize_whitespace("\n".join(pages))

    def read_docx(p: str) -> str:
        import docx  # python-docx optional
        d = docx.Document(p)
        parts = [para.text for para in d.paragraphs if para.text and para.text.strip()]
        return normalize_whitespace("\n".join(parts))

    try:
        if ext in [".txt", ".md"]:
            return read_txt(path)
        if ext == ".pdf":
            return read_pdf(path)
        if ext == ".docx":
            return read_docx(path)
        return None
    except Exception:
        return None

# =========================
# Chunk
# =========================
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str   # relative to grade dir
    grade: str         # cap1/cap2/cap3
    order: int

def chunk_text(text: str, source_file: str, grade: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    text = normalize_whitespace(text)
    if not text:
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks: List[Chunk] = []
    order = 0
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if len(piece) < 50:
            continue
        cid = f"{grade}:{source_file}:{order}"
        chunks.append(Chunk(chunk_id=cid, text=piece, source_file=source_file, grade=grade, order=order))
        order += 1
    return chunks

# =========================
# Gemini Client
# =========================
class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, model: str, prompt: str) -> str:
        res = self.client.models.generate_content(model=model, contents=prompt)
        return (getattr(res, "text", "") or "").strip()

    def generate_json(self, model: str, prompt: str) -> dict:
        raw = self.generate(model=model, prompt=prompt).strip()
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            raw = raw[s:e+1]
        return json.loads(raw)

    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for t in texts:
            res = self.client.models.embed_content(model=model, contents=t)
            emb = getattr(res, "embeddings", None)
            if emb is None:
                raise RuntimeError("Không lấy được embeddings.")
            if isinstance(emb, list) and emb and hasattr(emb[0], "values"):
                vectors.append(list(emb[0].values))
            elif isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
                vectors.append(list(emb[0]))
            else:
                vectors.append(list(emb))
        return vectors

# =========================
# Router (rule-first + fallback LLM)
# =========================
def route_grade(gemini: GeminiClient, model: str, question: str) -> Grade:
    q = question.lower().strip()

    # Rule-based nhanh
    if "cấp 1" in q or re.search(r"\blớp\s*(1|2|3|4|5)\b", q):
        return "cap1"
    if "cấp 2" in q or re.search(r"\blớp\s*(6|7|8|9)\b", q):
        return "cap2"
    if "cấp 3" in q or re.search(r"\blớp\s*(10|11|12)\b", q):
        return "cap3"

    # Fallback dùng Gemini phân loại
    prompt = f"""
Chỉ trả về JSON thuần.
Schema: {{"grade":"cap1|cap2|cap3|all","confidence":0.0-1.0}}

Quy ước:
- cap1: tiểu học (lớp 1-5)
- cap2: THCS (lớp 6-9)
- cap3: THPT (lớp 10-12)
- all: mơ hồ

Câu hỏi: {question}
"""
    try:
        obj = gemini.generate_json(model=model, prompt=prompt)
        grade = obj.get("grade", "all")
        if grade in ("cap1", "cap2", "cap3", "all"):
            return grade
    except Exception:
        pass
    return "all"

# =========================
# Vector Index + Store (cache per grade)
# =========================
@dataclass
class VectorIndex:
    chunks: List[Chunk]
    vectors: np.ndarray  # (n, d)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        if self.vectors.size == 0:
            return []
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        v = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-12)
        sims = v @ q
        idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in idx]

def build_or_load_index(
    gemini: GeminiClient,
    embedding_model: str,
    cache_dir: str,
    grade: str,
    grade_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> VectorIndex:
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"index_{grade}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        return VectorIndex(chunks=obj["chunks"], vectors=obj["vectors"])

    files = list_files_recursive(grade_dir)
    all_chunks: List[Chunk] = []
    for fp in files:
        txt = load_text(fp)
        if not txt:
            continue
        rel = os.path.relpath(fp, grade_dir)
        all_chunks.extend(chunk_text(txt, rel, grade, chunk_size, chunk_overlap))

    if not all_chunks:
        idx = VectorIndex(chunks=[], vectors=np.zeros((0, 1), dtype=np.float32))
    else:
        embs = gemini.embed_texts(model=embedding_model, texts=[c.text for c in all_chunks])
        vecs = np.array(embs, dtype=np.float32)
        idx = VectorIndex(chunks=all_chunks, vectors=vecs)

    with open(cache_path, "wb") as f:
        pickle.dump({"chunks": idx.chunks, "vectors": idx.vectors}, f)
    return idx

# =========================
# Prompt builder
# =========================
def build_rag_prompt(question: str, blocks: List[Dict]) -> str:
    ctx = []
    for i, b in enumerate(blocks, 1):
        ctx.append(f"[TÀI LIỆU {i}] (source: {b['source']}, score: {b['score']:.3f})\n{b['text']}")
    context = "\n\n".join(ctx) if ctx else "(Không có ngữ cảnh truy xuất được.)"

    return f"""
Bạn là trợ lý học tập môn Ngữ Văn.
Chỉ trả lời dựa trên NGỮ CẢNH. Nếu thiếu, nói rõ “không tìm thấy trong tài liệu” và gợi ý cần bổ sung gì.

Yêu cầu:
- Tiếng Việt, rõ ràng, đúng trọng tâm.
- Có thể gạch đầu dòng.
- Cuối trả lời có "Nguồn tham khảo:" liệt kê tên file nguồn đã dùng.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()

# =========================
# RAG Chatbot
# =========================
class RAGChatbot:
    def __init__(self, settings: Settings):
        self.s = settings
        self.gemini = GeminiClient(api_key=settings.gemini_api_key)
        self._indices: Dict[str, VectorIndex] = {}

    def _grade_dir(self, grade: str) -> str:
        return os.path.join(self.s.input_dir, grade)

    def _get_index(self, grade: str) -> VectorIndex:
        if grade in self._indices:
            return self._indices[grade]
        idx = build_or_load_index(
            gemini=self.gemini,
            embedding_model=self.s.embedding_model,
            cache_dir=self.s.cache_dir,
            grade=grade,
            grade_dir=self._grade_dir(grade),
            chunk_size=self.s.chunk_size,
            chunk_overlap=self.s.chunk_overlap,
        )
        self._indices[grade] = idx
        return idx

    def retrieve(self, grade: Grade, question: str) -> List[Dict]:
        q_vec = np.array(self.gemini.embed_texts(self.s.embedding_model, [question])[0], dtype=np.float32)
        grades = ["cap1", "cap2", "cap3"] if grade == "all" else [grade]

        pooled: List[Dict] = []
        for g in grades:
            idx = self._get_index(g)
            hits = idx.search(q_vec, self.s.top_k)
            for chunk, score in hits:
                pooled.append({
                    "source": f"{chunk.grade}/{chunk.source_file}",
                    "score": score,
                    "text": chunk.text,
                })

        pooled.sort(key=lambda x: x["score"], reverse=True)
        return pooled[: self.s.top_k]

    def answer(self, question: str) -> Dict:
        grade = route_grade(self.gemini, self.s.gemini_model, question)
        retrieved = self.retrieve(grade, question)
        prompt = build_rag_prompt(question, retrieved)
        reply = self.gemini.generate(self.s.gemini_model, prompt)
        return {"grade": grade, "retrieved": retrieved, "answer": reply}

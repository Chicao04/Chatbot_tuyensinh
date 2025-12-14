from __future__ import annotations

import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal, Optional

import numpy as np
from dotenv import load_dotenv

# =========================
# Config
# =========================
load_dotenv()

Category = Literal["chuyennganh", "diem_chuan", "thacsi_tiensi", "all"]


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


def list_txt_files_recursive(root: str) -> List[str]:
    """Chỉ lấy file .txt (đúng yêu cầu đầu vào)."""
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() != ".txt":
                continue
            out.append(os.path.join(dirpath, fn))
    return out


# =========================
# Loader (.txt only)
# =========================
def load_text(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext != ".txt":
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return normalize_whitespace(f.read())
    except Exception:
        return None


# =========================
# Chunk
# =========================
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str  # relative to category dir
    category: str     # chuyennganh/diem_chuan/thacsi_tiensi
    order: int


def chunk_text(
    text: str,
    source_file: str,
    category: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
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
        cid = f"{category}:{source_file}:{order}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=piece,
                source_file=source_file,
                category=category,
                order=order,
            )
        )
        order += 1
    return chunks


# =========================
# Gemini Client (supports google-genai AND google-generativeai)
# =========================
class GeminiClient:
    """
    Ưu tiên thư viện mới: google-genai (from google import genai)
    Fallback: google-generativeai (import google.generativeai as genai)
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.backend = None
        self.client = None  # for google-genai
        self._old = None    # for google-generativeai

        # Backend 1: google-genai
        try:
            from google import genai as genai_new  # type: ignore
            self.backend = "google-genai"
            self.client = genai_new.Client(api_key=api_key)
            return
        except Exception:
            pass

        # Backend 2: google-generativeai
        try:
            import google.generativeai as genai_old  # type: ignore
            self.backend = "google-generativeai"
            genai_old.configure(api_key=api_key)
            self._old = genai_old
            return
        except Exception:
            pass

        raise RuntimeError(
            "Không import được SDK Gemini.\n"
            "- Cách 1 (khuyến nghị): pip install -U google-genai\n"
            "- Hoặc (cũ): pip install -U google-generativeai\n"
            "Lưu ý: tránh cài nhầm package 'google' (dễ gây ImportError)."
        )

    def generate(self, model: str, prompt: str) -> str:
        if self.backend == "google-genai":
            res = self.client.models.generate_content(model=model, contents=prompt)  # type: ignore[union-attr]
            return (getattr(res, "text", "") or "").strip()

        # google-generativeai
        assert self._old is not None
        m = self._old.GenerativeModel(model)
        res = m.generate_content(prompt)
        return (getattr(res, "text", "") or "").strip()

    def generate_json(self, model: str, prompt: str) -> dict:
        raw = self.generate(model=model, prompt=prompt).strip()
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            raw = raw[s : e + 1]
        return json.loads(raw)

    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []

        if self.backend == "google-genai":
            for t in texts:
                res = self.client.models.embed_content(model=model, contents=t)  # type: ignore[union-attr]
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

        # google-generativeai
        assert self._old is not None
        for t in texts:
            res = self._old.embed_content(model=model, content=t)
            # res thường dạng dict: {"embedding":[...], ...}
            if isinstance(res, dict) and "embedding" in res:
                vectors.append(list(res["embedding"]))
            else:
                # best-effort
                vectors.append(list(getattr(res, "embedding", [])))
        return vectors


# =========================
# Router (rule-first + fallback LLM)
# =========================
def route_category(gemini: GeminiClient, model: str, question: str) -> Category:
    q = question.lower().strip()

    # Rule-based nhanh (ưu tiên độ chắc)
    if any(k in q for k in ["thạc sĩ", "cao học", "tiến sĩ", "nghiên cứu sinh", "sau đại học", "luận văn", "luận án", "master", "phd"]):
        return "thacsi_tiensi"

    if any(k in q for k in ["điểm chuẩn", "điểm trúng tuyển", "điểm sàn", "ngưỡng đảm bảo", "cut-off", "bao nhiêu điểm", "lấy bao nhiêu điểm"]):
        return "diem_chuan"
    # pattern năm, điểm, tổ hợp
    if re.search(r"\b(20\d{2})\b", q) and any(k in q for k in ["điểm", "trúng tuyển", "chuẩn"]):
        return "diem_chuan"

    if any(k in q for k in ["ngành", "chuyên ngành", "học gì", "chương trình", "môn học", "đầu ra", "cơ hội việc làm", "tổ hợp", "khối xét tuyển"]):
        return "chuyennganh"

    # Fallback dùng Gemini phân loại
    prompt = f"""
Chỉ trả về JSON thuần.
Schema: {{"category":"chuyennganh|diem_chuan|thacsi_tiensi|all","confidence":0.0-1.0}}

Định nghĩa:
- chuyennganh: thông tin ngành/chuyên ngành, chương trình đào tạo, tổ hợp xét tuyển, phương thức xét tuyển, học phí/học bổng cơ bản
- diem_chuan: điểm chuẩn/điểm trúng tuyển/điểm sàn theo năm hoặc theo ngành
- thacsi_tiensi: tuyển sinh sau đại học (thạc sĩ/tiến sĩ), hồ sơ, điều kiện, đề cương, quy trình
- all: mơ hồ/không đủ thông tin

Câu hỏi: {question}
"""
    try:
        obj = gemini.generate_json(model=model, prompt=prompt)
        cat = obj.get("category", "all")
        if cat in ("chuyennganh", "diem_chuan", "thacsi_tiensi", "all"):
            return cat  # type: ignore[return-value]
    except Exception:
        pass
    return "all"


# =========================
# Vector Index + Store (cache per category)
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
    category: str,
    category_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> VectorIndex:
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"index_{category}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        return VectorIndex(chunks=obj["chunks"], vectors=obj["vectors"])

    files = list_txt_files_recursive(category_dir)
    all_chunks: List[Chunk] = []
    for fp in files:
        txt = load_text(fp)
        if not txt:
            continue
        rel = os.path.relpath(fp, category_dir)
        all_chunks.extend(chunk_text(txt, rel, category, chunk_size, chunk_overlap))

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
# Prompt builder (Tuyển sinh)
# =========================
def build_rag_prompt(question: str, blocks: List[Dict]) -> str:
    ctx = []
    for i, b in enumerate(blocks, 1):
        ctx.append(f"[TÀI LIỆU {i}] (source: {b['source']}, score: {b['score']:.3f})\n{b['text']}")
    context = "\n\n".join(ctx) if ctx else "(Không có ngữ cảnh truy xuất được.)"

    return f"""
Bạn là trợ lý tư vấn tuyển sinh.
CHỈ trả lời dựa trên NGỮ CẢNH được cung cấp bên dưới. Nếu thiếu thông tin, phải nói rõ: "Không tìm thấy trong tài liệu hiện có" và gợi ý người dùng cần bổ sung file .txt nào (ví dụ: điểm chuẩn năm X, phương thức xét tuyển, học phí...).

Yêu cầu trả lời:
- Tiếng Việt, ngắn gọn nhưng đầy đủ ý.
- Nếu có bước/điều kiện/hồ sơ: trình bày dạng gạch đầu dòng.
- Không bịa số liệu. Nếu tài liệu không có số, hãy nói "chưa có".
- Cuối câu trả lời có mục "Nguồn tham khảo:" liệt kê các file đã dùng (đường dẫn source).

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

    def _category_dir(self, category: str) -> str:
        return os.path.join(self.s.input_dir, category)

    def _get_index(self, category: str) -> VectorIndex:
        if category in self._indices:
            return self._indices[category]
        idx = build_or_load_index(
            gemini=self.gemini,
            embedding_model=self.s.embedding_model,
            cache_dir=self.s.cache_dir,
            category=category,
            category_dir=self._category_dir(category),
            chunk_size=self.s.chunk_size,
            chunk_overlap=self.s.chunk_overlap,
        )
        self._indices[category] = idx
        return idx

    def retrieve(self, category: Category, question: str) -> List[Dict]:
        q_vec = np.array(self.gemini.embed_texts(self.s.embedding_model, [question])[0], dtype=np.float32)
        categories = ["chuyennganh", "diem_chuan", "thacsi_tiensi"] if category == "all" else [category]

        pooled: List[Dict] = []
        for c in categories:
            idx = self._get_index(c)
            hits = idx.search(q_vec, self.s.top_k)
            for chunk, score in hits:
                pooled.append(
                    {
                        "source": f"{chunk.category}/{chunk.source_file}",
                        "score": score,
                        "text": chunk.text,
                    }
                )

        pooled.sort(key=lambda x: x["score"], reverse=True)
        return pooled[: self.s.top_k]

    def answer(self, question: str) -> Dict:
        category = route_category(self.gemini, self.s.gemini_model, question)
        retrieved = self.retrieve(category, question)
        prompt = build_rag_prompt(question, retrieved)
        reply = self.gemini.generate(self.s.gemini_model, prompt)
        return {"grade": category, "retrieved": retrieved, "answer": reply}

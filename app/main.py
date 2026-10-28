"""FastAPI app: student chat, document upload/admin, feedback."""
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .chat import ChatBot
from .ingest import file_to_chunks
from .retrieval import Retriever, embed
from .store import Store

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD = 15 * 1024 * 1024


class AskIn(BaseModel):
    session_id: str = Field(min_length=4, max_length=64)
    question: str = Field(min_length=1, max_length=1000)


class FeedbackIn(BaseModel):
    question_id: int
    helpful: bool


def ingest_bytes(store: Store, name: str, data: bytes) -> dict:
    chunks = file_to_chunks(name, data)
    vecs = embed([f"{c.heading or ''}\n{c.text}" for c in chunks])
    blobs = [v.astype(np.float32).tobytes() for v in vecs] if vecs else None
    old = store.find_document(name)
    if old:  # re-uploading a file replaces it
        store.delete_document(old["id"])
    return store.add_document(name, chunks, blobs)


def load_samples(store: Store) -> None:
    if store.list_documents():
        return
    for f in sorted(config.SAMPLE_DIR.glob("*.md")):
        ingest_bytes(store, f.name, f.read_bytes())


def create_app(db_path: str | None = None, seed: bool = True) -> FastAPI:
    app = FastAPI(title=f"{config.INSTITUTE_NAME} - RAG Assistant", version="1.0.0")
    store = Store(db_path)
    if seed:
        load_samples(store)
    retriever = Retriever(store)
    bot = ChatBot(store, retriever)
    app.state.store, app.state.retriever = store, retriever

    def admin(x_admin_token: str = Header(default="")):
        if x_admin_token != config.ADMIN_TOKEN:
            raise HTTPException(401, "Invalid admin token")

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(STATIC / "index.html")

    @app.get("/admin", include_in_schema=False)
    def admin_page():
        return FileResponse(STATIC / "admin.html")

    @app.get("/health")
    def health():
        return {"ok": True, "llm": bool(config.LLM_API_KEY), "dense": retriever.dense is not None,
                "chunks": len(retriever.chunks)}

    @app.get("/api/info")
    def info():
        return {"name": config.INSTITUTE_NAME}

    @app.post("/api/ask")
    def ask(body: AskIn):
        with store.lock:
            return bot.ask(body.session_id, body.question)

    @app.post("/api/feedback")
    def feedback(body: FeedbackIn):
        if not store.set_feedback(body.question_id, 1 if body.helpful else -1):
            raise HTTPException(404, "Question not found")
        return {"ok": True}

    # ---------- admin
    @app.get("/api/admin/documents", dependencies=[Depends(admin)])
    def docs():
        return store.list_documents()

    @app.post("/api/admin/documents", status_code=201, dependencies=[Depends(admin)])
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "File too large (max 15 MB)")
        try:
            with store.lock:
                doc = ingest_bytes(store, Path(file.filename or "upload.txt").name, data)
                retriever.rebuild()
        except ValueError as e:
            raise HTTPException(400, str(e))
        return doc

    @app.delete("/api/admin/documents/{doc_id}", dependencies=[Depends(admin)])
    def delete(doc_id: int):
        with store.lock:
            if not store.delete_document(doc_id):
                raise HTTPException(404, "Document not found")
            retriever.rebuild()
        return {"ok": True}

    @app.get("/api/admin/search", dependencies=[Depends(admin)])
    def search(q: str):
        """Debug retrieval: see which chunks a question pulls up."""
        return retriever.search(q)

    @app.get("/api/admin/stats", dependencies=[Depends(admin)])
    def stats():
        return store.stats()

    @app.get("/api/admin/gaps", dependencies=[Depends(admin)])
    def gaps():
        return store.gaps()

    return app


app = create_app()

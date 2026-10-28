import pytest
from fastapi.testclient import TestClient

from app import chat, config
from app.ingest import chunk_text, file_to_chunks
from app.main import create_app
from app.retrieval import BM25, tokenize


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "EMBED_API_KEY", "")
    monkeypatch.setattr(config, "ADMIN_TOKEN", "secret")


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "t.db")))


H = {"X-Admin-Token": "secret"}


def ask(client, q, sid="sess1"):
    return client.post("/api/ask", json={"session_id": sid, "question": q}).json()


# ---------------- units

def test_chunker_respects_headings_and_size():
    text = "# Fees\n\nJEE costs Rs 1,20,000.\n\n# Hostel\n\n" + "Hostel food is good. " * 100
    chunks = chunk_text(text, size=300, overlap=50)
    assert chunks[0].heading == "Fees" and "Hostel" not in chunks[0].text
    assert all(c.heading == "Hostel" for c in chunks[1:])
    assert all(len(c.text) <= 300 + 60 for c in chunks)


def test_bm25_ranks_relevant_doc_first():
    docs = [tokenize("refund policy within seven days"), tokenize("hostel food and rooms"), tokenize("physics tips")]
    scores = BM25(docs).scores(tokenize("what is the refund policy"))
    assert scores[0] > 0 and scores[0] == max(scores)


def test_pdf_ingest():
    from pypdf import PdfWriter
    import io
    w = PdfWriter(); w.add_blank_page(200, 200); buf = io.BytesIO(); w.write(buf)
    with pytest.raises(ValueError):  # blank page has no text
        file_to_chunks("x.pdf", buf.getvalue())
    with pytest.raises(ValueError):
        file_to_chunks("x.docx", b"nope")


# ---------------- answering (offline mode)

@pytest.mark.parametrize("q, expected", [
    ("What is the NEET fee?", "1,10,000"),
    ("Refund policy?", "within 7 days"),
    ("Evening batch timings", "5:00 PM to 8:00 PM"),
    ("How do scholarships work?", "50% fee waiver"),
    ("Is there a hostel?", "9,000"),
    ("doubt clearing time", "4:00 PM"),
])
def test_answers_from_documents(client, q, expected):
    r = ask(client, q)
    assert r["answered"] and expected in r["answer"]
    assert r["sources"] and "[1]" in r["answer"]


def test_follow_up_uses_previous_question(client):
    ask(client, "What is the NEET fee?", sid="follow1")
    r = ask(client, "and can I pay it in instalments?", sid="follow1")
    assert "instalments" in r["answer"]


def test_out_of_scope_is_refused_and_logged(client):
    r = ask(client, "Who won the cricket world cup?")
    assert not r["answered"] and "couldn't find" in r["answer"]
    gaps = client.get("/api/admin/gaps", headers=H).json()
    assert gaps[0]["question"] == "Who won the cricket world cup?"


def test_greeting(client):
    assert "assistant" in ask(client, "hi")["answer"]


# ---------------- LLM path (mocked)

def test_llm_answer_with_citations(client, monkeypatch):
    seen = {}

    def fake_llm(messages, max_tokens=400):
        seen["prompt"] = messages[-1]["content"]
        return "The NEET program costs Rs 1,10,000 per year [1]."

    monkeypatch.setattr(config, "LLM_API_KEY", "x")
    monkeypatch.setattr(chat, "llm_chat", fake_llm)
    r = ask(client, "NEET fee?")
    assert "Context:" in seen["prompt"] and "[1]" in seen["prompt"]
    assert r["answer"].endswith("[1].") and len(r["sources"]) == 1


# ---------------- admin + feedback

def test_upload_search_delete(client):
    assert client.get("/api/admin/documents").status_code == 401
    files = {"file": ("transport.md", b"# Transport\n\nCollege buses run from Uppal and LB Nagar every morning at 6 AM.")}
    doc = client.post("/api/admin/documents", files=files, headers=H).json()
    assert doc["n_chunks"] == 1
    r = ask(client, "Is there a bus from Uppal?")
    assert "Uppal" in r["answer"]
    assert client.delete(f"/api/admin/documents/{doc['id']}", headers=H).status_code == 200
    assert not ask(client, "Is there a bus from Uppal?", sid="other")["answered"]


def test_reupload_replaces(client):
    f = lambda body: {"file": ("note.txt", body)}  # noqa: E731
    client.post("/api/admin/documents", files=f(b"Library opens at 8 AM on all days."), headers=H)
    client.post("/api/admin/documents", files=f(b"Library opens at 9 AM on all days."), headers=H)
    names = [d["name"] for d in client.get("/api/admin/documents", headers=H).json()]
    assert names.count("note.txt") == 1
    assert "9 AM" in ask(client, "When does the library open?")["answer"]


def test_feedback_and_stats(client):
    qid = ask(client, "Refund policy?")["question_id"]
    assert client.post("/api/feedback", json={"question_id": qid, "helpful": False}).status_code == 200
    s = client.get("/api/admin/stats", headers=H).json()
    assert s["documents"] == 3 and s["not_helpful"] == 1


def test_pages(client):
    assert client.get("/").status_code == 200 and client.get("/admin").status_code == 200

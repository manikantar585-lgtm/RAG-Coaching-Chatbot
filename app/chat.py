"""Answer pipeline: rewrite follow-ups -> retrieve -> answer with citations."""
import re

import httpx

from . import config
from .retrieval import tokenize

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|namaste|good (morning|afternoon|evening)|నమస్తే|హాయ్)\W*$", re.I)
THANKS_RE = re.compile(r"^\s*(thanks|thank you|thx|ok(ay)?|great|ధన్యవాదాలు)\W*$", re.I)
FOLLOWUP_RE = re.compile(r"^\s*(and|also|what about|how about|then)\b|\b(it|its|that|they|them|those|same one)\b", re.I)
TELUGU_RE = re.compile(r"[ఀ-౿]")

SYSTEM = (
    "You are the helpful student assistant of {name}, a coaching institute. Answer ONLY using the numbered "
    "context passages. Cite the passages you used like [1] or [2][3]. If the answer is not in the context, say "
    "you don't have that information and suggest contacting {contact}. Be concise and friendly (max ~120 words), "
    "use bullet points for lists, and answer in the same language as the question (English or Telugu)."
)


def llm_chat(messages: list[dict], max_tokens: int = 400) -> str | None:
    if not config.LLM_API_KEY:
        return None
    try:
        r = httpx.post(f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                       headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                       json={"model": config.LLM_MODEL, "temperature": 0.2, "max_tokens": max_tokens,
                             "messages": messages}, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def standalone_question(question: str, history: list[dict]) -> str:
    """Make follow-ups like 'and its fees?' searchable on their own, and
    translate Telugu questions to English for search (when an LLM is set)."""
    if TELUGU_RE.search(question):
        translated = llm_chat([
            {"role": "system", "content": "Translate the student's question to English. Reply with the translation only."},
            {"role": "user", "content": question}], max_tokens=80)
        if translated:
            question = translated
    if not history:
        return question
    if not FOLLOWUP_RE.search(question):
        return question
    prev = history[-1]["question"]
    rewritten = llm_chat([
        {"role": "system", "content": "Rewrite the follow-up as one standalone question. Reply with the question only."},
        {"role": "user", "content": f"Previous question: {prev}\nFollow-up: {question}"},
    ], max_tokens=60)
    return rewritten or f"{prev} {question}"


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s*\n\s*[-*•]\s*", ". ", "\n" + text)  # bullets -> sentences
    parts = re.split(r"(?<=[.!])\s+|\n+", text)  # keep "Question? Answer." together
    return [p.strip(" .") for p in parts if len(p.strip()) > 15]


def extractive_answer(query: str, hits: list[dict], idf: dict) -> str:
    """Offline answer: best-matching sentences from the top passages, with citations."""
    q = set(tokenize(query))
    top = hits[0]
    if top["heading"] and set(tokenize(top["heading"])) & q and len(top["text"]) <= 650:
        return f"{top['text']} [1]"  # a short section whose title matches: show it whole
    scored = []
    for n, h in enumerate(hits, 1):
        rank_w = 1 / (1 + 0.5 * (n - 1))                 # trust the top passage most
        head = set(tokenize(f"{h['heading'] or ''}")) & q  # "NEET" in the heading counts for its sentences
        head_bonus = 0.5 * sum(idf.get(t, 1.0) for t in head)
        for pos, s in enumerate(_sentences(h["text"])):
            toks = set(tokenize(s))
            overlap = sum(idf.get(t, 1.0) for t in q & toks)
            if overlap > 0 or (head and pos < 2):
                score = (overlap + head_bonus) * rank_w / (1 + 0.02 * len(toks))
                scored.append((score, n, pos, s))
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[0])
    best = [x for x in scored[:3] if x[0] >= 0.35 * scored[0][0]]  # drop weak filler sentences
    best.sort(key=lambda x: (x[1], x[2]))  # keep document order
    end = lambda s: s if s[-1:] in "?!" else s + "."  # noqa: E731
    return "\n".join(f"- {end(s)} [{n}]" for _, n, _, s in best)


class ChatBot:
    def __init__(self, store, retriever):
        self.store = store
        self.retriever = retriever

    def ask(self, session_id: str, question: str) -> dict:
        question = question.strip()
        if GREETING_RE.match(question):
            return {"answer": f"Hi! I'm the {config.INSTITUTE_NAME} assistant. Ask me about courses, fees, batches, "
                              f"exams or study tips.", "sources": [], "answered": True, "question_id": None}
        if THANKS_RE.match(question):
            return {"answer": "You're welcome! All the best with your preparation.",
                    "sources": [], "answered": True, "question_id": None}

        history = self.store.recent_history(session_id)
        query = standalone_question(question, history)
        hits = self.retriever.search(query)
        top = hits[0]["score"] if hits else 0.0
        best_cov = max((h["coverage"] for h in hits), default=0.0)
        q_tokens = set(tokenize(query))
        heading_hit = any(set(tokenize(h["heading"] or "")) & q_tokens for h in hits[:2])
        found = bool(hits) and (self.retriever.dense is not None
                                or (top >= config.MIN_SCORE and (best_cov >= config.MIN_COVERAGE or heading_hit)))

        answer = ""
        if found:
            context = "\n\n".join(
                f"[{n}] ({h['doc_name']}{' > ' + h['heading'] if h['heading'] else ''})\n{h['text']}"
                for n, h in enumerate(hits, 1))
            msgs = [{"role": "system", "content": SYSTEM.format(name=config.INSTITUTE_NAME, contact=config.CONTACT)}]
            for turn in history[-2:]:
                msgs += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
            msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
            answer = llm_chat(msgs) or extractive_answer(query, hits, self.retriever.bm25.idf)
            found = bool(answer)

        if not found:
            answer = (f"Sorry, I couldn't find that in {config.INSTITUTE_NAME}'s documents. "
                      f"Please contact {config.CONTACT} for help.")
            if TELUGU_RE.search(question):
                answer = f"క్షమించండి, ఈ సమాచారం మా డాక్యుమెంట్లలో లేదు. దయచేసి {config.CONTACT} ను సంప్రదించండి."
            hits = []

        used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [{"n": n, "doc": h["doc_name"], "page": h["page"], "heading": h["heading"],
                    "snippet": h["text"][:220] + ("…" if len(h["text"]) > 220 else "")}
                   for n, h in enumerate(hits, 1) if not used or n in used]
        qid = self.store.log_question(session_id, question, answer, found, top)
        return {"answer": answer, "sources": sources, "answered": found, "question_id": qid}

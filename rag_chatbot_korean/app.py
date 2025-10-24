import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict
from pypdf import PdfReader

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
st.set_page_config(page_title="나만의 RAG 챗봇 (초간단 안정판)", page_icon="🤖", layout="wide")

st.title("🤖 나만의 RAG 챗봇 (초간단 안정판)")
st.caption("파일 업로드 → 청크 분할 → 키워드 점수로 상위 문맥 선택 → LLM 답변 (근거 표시)")

# 세션 상태
if "splits" not in st.session_state:
    st.session_state.splits: List[Dict] = []
if "history" not in st.session_state:
    st.session_state.history = []

# 사이드바
st.sidebar.header("⚙️ 설정")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.sidebar.error("Secrets에 OPENAI_API_KEY를 등록하세요.")
else:
    st.sidebar.success("OPENAI_API_KEY 감지됨")

# 업로드
uploaded = st.file_uploader("📄 PDF/TXT 업로드 (여러 개 가능)", type=["pdf", "txt"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input("청크 크기", 200, 3000, 1000, 100)
with col2:
    chunk_overlap = st.number_input("청크 중첩", 0, 800, 150, 50)

def read_pdf(file) -> str:
    pdf = PdfReader(file)
    texts = []
    for p in pdf.pages:
        texts.append(p.extract_text() or "")
    return "\n".join(texts)

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def split_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

if st.button("🔨 인덱스 구축"):
    if not uploaded:
        st.warning("먼저 파일을 업로드하세요.")
    else:
        splits = []
        for f in uploaded:
            name = getattr(f, "name", "uploaded")
            if name.lower().endswith(".pdf"):
                content = read_pdf(f)
            else:
                content = read_txt(f)
            for ch in split_text(content, int(chunk_size), int(chunk_overlap)):
                if ch.strip():
                    splits.append({"text": ch, "source": name})
        st.session_state.splits = splits
        st.success(f"인덱스 구축 완료! 청크 수: {len(splits)}")

# 간단 키워드 점수 기반 검색
def rank_by_keywords(query: str, splits: List[Dict], k: int = 4) -> List[Dict]:
    q_tokens = [t for t in query.lower().split() if t.strip()]
    scores = []
    for s in splits:
        text = s["text"].lower()
        score = sum(text.count(t) for t in q_tokens) + 1e-6  # 최소값
        scores.append((score, s))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scores[:k]]

st.subheader("💬 질문하기")
q = st.text_input("예: 이 문서의 핵심 요약은?")
top_k = st.slider("검색 문서 수 (k)", 1, 10, 4)
ask = st.button("질문 보내기")

if ask:
    if not st.session_state.splits:
        st.warning("먼저 인덱스를 구축하세요.")
    elif not q.strip():
        st.warning("질문을 입력하세요.")
    else:
        try:
            top_docs = rank_by_keywords(q, st.session_state.splits, int(top_k))
            context = "\n\n---\n\n".join([d["text"] for d in top_docs])
            sources = ", ".join(sorted({d["source"] for d in top_docs}))

            template = """
당신은 업로드된 자료를 근거로 정확하고 간결하게 한국어로 답변합니다.
[자료]
{context}

# ---- 질문 ----
st.subheader("💬 질문하기")
q = st.text_input("예: 이 문서의 핵심 요약은?")
top_k = st.slider("검색 문서 수 (k)", 1, 10, 4)
ask = st.button("질문 보내기")

def split_text(text: str, size: int, overlap: int):
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        chunks.append(text[i:i+size])
        i += step
    return chunks

def ensure_index_from_uploads():
    # 이미 구축되어 있으면 그대로 사용
    if st.session_state.get("splits"):
        return
    if not uploaded:
        return
    tmp = []
    for f in uploaded:
        name = getattr(f, "name", "uploaded")
        if name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            pdf = PdfReader(f)
            text = "\n".join([(p.extract_text() or "") for p in pdf.pages])
        elif name.lower().endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
        else:
            continue  # 只支持 pdf/txt
        for ch in split_text(text, int(chunk_size), int(chunk_overlap)):
            if ch.strip():
                tmp.append({"text": ch, "source": name})
    st.session_state.splits = tmp

if ask:
    # ① 自动尝试建索引
    if not st.session_state.get("splits"):
        with st.spinner("인덱스를 자동으로 구축하는 중..."):
            ensure_index_from_uploads()

    # ② 仍然没有可用文本，给出提示
    if not st.session_state.get("splits"):
        st.warning("먼저 파일(.txt/.pdf)을 업로드하고 인덱스를 구축하세요.")
    elif not q.strip():
        st.warning("질문을 입력하세요.")
    else:
        # 下面是你的检索与回答逻辑（保持不变）
        top_docs = rank_by_keywords(q, st.session_state.splits, int(top_k))
        context = "\n\n---\n\n".join([d["text"] for d in top_docs])
        sources = ", ".join(sorted({d["source"] for d in top_docs}))
        template = """
당신은 업로드된 자료를 근거로 정확하고 간결하게 한국어로 답변합니다.
[자료]
{context}

# ---- 질문 ----
st.subheader("💬 질문하기")
q = st.text_input("예: 이 문서의 핵심 요약은?")
top_k = st.slider("검색 문서 수 (k)", 1, 10, 4)
ask = st.button("질문 보내기")

def split_text(text: str, size: int, overlap: int):
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        chunks.append(text[i:i+size])
        i += step
    return chunks

def ensure_index_from_uploads():
    # 이미 구축되어 있으면 그대로 사용
    if st.session_state.get("splits"):
        return
    if not uploaded:
        return
    tmp = []
    for f in uploaded:
        name = getattr(f, "name", "uploaded")
        if name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            pdf = PdfReader(f)
            text = "\n".join([(p.extract_text() or "") for p in pdf.pages])
        elif name.lower().endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
        else:
            continue  # 只支持 pdf/txt
        for ch in split_text(text, int(chunk_size), int(chunk_overlap)):
            if ch.strip():
                tmp.append({"text": ch, "source": name})
    st.session_state.splits = tmp

if ask:
    # ① 自动尝试建索引
    if not st.session_state.get("splits"):
        with st.spinner("인덱스를 자동으로 구축하는 중..."):
            ensure_index_from_uploads()

    # ② 仍然没有可用文本，给出提示
    if not st.session_state.get("splits"):
        st.warning("먼저 파일(.txt/.pdf)을 업로드하고 인덱스를 구축하세요.")
    elif not q.strip():
        st.warning("질문을 입력하세요.")
    else:
        # 下面是你的检索与回答逻辑（保持不变）
        top_docs = rank_by_keywords(q, st.session_state.splits, int(top_k))
        context = "\n\n---\n\n".join([d["text"] for d in top_docs])
        sources = ", ".join(sorted({d["source"] for d in top_docs}))
        template = """
당신은 업로드된 자료를 근거로 정확하고 간결하게 한국어로 답변합니다.
[자료]
{context}

[질문]
{question}

규칙:
- 반드시 한국어로 답변
- 마지막 줄에 '근거:' 뒤에 출처 파일명 나열

답변:
"""
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = PromptTemplate.from_template(template)
        resp = llm.invoke(prompt.format(context=context, question=q)).content
        if "근거:" not in resp:
            resp += f"\n\n근거: {sources if sources else '업로드 자료'}"
        st.session_state.history.append(("user", q))
        st.session_state.history.append(("bot", resp, list(sorted({d['source'] for d in top_docs}))))

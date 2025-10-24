import os
from typing import List, Dict
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ------------------------
# 기본 설정
# ------------------------
load_dotenv()
st.set_page_config(page_title="나만의 RAG 챗봇 (초간단 안정판)", page_icon="🤖", layout="wide")

# ---- 스타일 ----
st.markdown(
    """
    <style>
        .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .chat-bubble-user {background:#eef4ff; padding:14px 16px; border-radius:16px; margin:6px 0;}
        .chat-bubble-bot  {background:#f6f6f6; padding:14px 16px; border-radius:16px; margin:6px 0; border-left:4px solid #5b9bd5;}
        .tiny {font-size: 12px; color:#666;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 나만의 RAG 챗봇 (초간단 안정판)")
st.caption("파일 업로드 → 청크 분할 → 키워드 점수로 상위 문맥 선택 → LLM 답변 (근거 표시)")

# ------------------------
# API Key (Secrets 우선, 그 외 ENV)
# ------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key  # 하위 라이브러리에서 읽도록
    st.sidebar.success("OPENAI_API_KEY 감지됨")
else:
    st.sidebar.error("Secrets에 OPENAI_API_KEY를 등록하세요.")

# ------------------------
# 세션 상태
# ------------------------
if "splits" not in st.session_state:
    st.session_state.splits: List[Dict] = []
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------
# 업로드 & 파라미터
# ------------------------
st.sidebar.header("⚙️ 설정")
uploaded_files = st.file_uploader("📄 PDF/TXT 업로드 (여러 개 가능)", type=["pdf", "txt"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input("청크 크기", min_value=200, max_value=3000, step=100, value=1000)
with col2:
    chunk_overlap = st.number_input("청크 중첩", min_value=0, max_value=800, step=50, value=150)

# ------------------------
# 유틸 함수
# ------------------------
def read_pdf(file) -> str:
    pdf = PdfReader(file)
    texts = []
    for p in pdf.pages:
        texts.append(p.extract_text() or "")
    return "\n".join(texts)

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def split_text(text: str, size: int, overlap: int) -> List[str]:
    """아주 단순한 고정길이 청크 분할기 (의존성 최소화를 위해 직접 구현)."""
    chunks = []
    i = 0
    step = max(1, size - overlap)
    while i < len(text):
        chunks.append(text[i:i + size])
        i += step
    return chunks

def build_splits_from_uploads(files, size: int, overlap: int) -> List[Dict]:
    tmp = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        if name.lower().endswith(".pdf"):
            content = read_pdf(f)
        elif name.lower().endswith(".txt"):
            content = read_txt(f)
        else:
            continue  # 지원하지 않는 확장자
        for ch in split_text(content, int(size), int(overlap)):
            if ch.strip():
                tmp.append({"text": ch, "source": name})
    return tmp

def rank_by_keywords(query: str, splits: List[Dict], k: int = 4) -> List[Dict]:
    """아주 단순한 키워드 빈도 스코어(순수 파이썬)"""
    q_tokens = [t for t in query.lower().split() if t.strip()]
    if not q_tokens:
        return splits[:k]
    scored = []
    for s in splits:
        text = s["text"].lower()
        score = sum(text.count(t) for t in q_tokens) + 1e-6
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]]

# ------------------------
# 인덱스 구축 (수동 버튼)
# ------------------------
if st.button("🔨 인덱스 구축"):
    if not uploaded_files:
        st.warning("먼저 파일(.txt/.pdf)을 업로드하세요.")
    else:
        st.session_state.splits = build_splits_from_uploads(uploaded_files, chunk_size, chunk_overlap)
        st.success(f"인덱스 구축 완료! 청크 수: {len(st.session_state.splits)}")

# ------------------------
# 질문하기 (자동 인덱스 구축 지원)
# ------------------------
st.subheader("💬 질문하기")
q = st.text_input("예: 이 문서의 핵심 요약은?")
top_k = st.slider("검색 문서 수 (k)", 1, 10, 4)
ask = st.button("질문 보내기")

if ask:
    # 1) 인덱스가 없으면 업로드 파일로 자동 구축 시도
    if not st.session_state.splits and uploaded_files:
        with st.spinner("인덱스를 자동으로 구축하는 중..."):
            st.session_state.splits = build_splits_from_uploads(uploaded_files, chunk_size, chunk_overlap)

    # 2) 여전히 없으면 안내
    if not st.session_state.splits:
        st.warning("먼저 파일(.txt/.pdf)을 업로드하고 인덱스를 구축하세요.")
    elif not q.strip():
        st.warning("질문을 입력하세요.")
    elif not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. Secrets에 키를 등록하세요.")
    else:
        # 검색 → 컨텍스트 구성
        top_docs = rank_by_keywords(q, st.session_state.splits, int(top_k))
        context = "\n\n---\n\n".join([d["text"] for d in top_docs])
        sources = ", ".join(sorted({d["source"] for d in top_docs}))

        # 프롬프트 (삼중따옴표 + dedent 로 안전하게 문자열 처리)
        template = dedent("""\
        당신은 업로드된 자료를 근거로 정확하고 간결하게 한국어로 답변합니다.
        [자료]
        {context}

        [질문]
        {question}

        규칙:
        - 반드시 한국어로 답변
        - 마지막 줄에 '근거:' 뒤에 출처 파일명 나열

        답변:
        """)
        prompt = PromptTemplate.from_template(template)

        # LLM 호출
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        resp = llm.invoke(prompt.format(context=context, question=q)).content

        # '근거:' 보강
        if "근거:" not in resp:
            resp += f"\n\n근거: {sources if sources else '업로드 자료'}"

        # 기록
        st.session_state.history.append(("user", q))
        st.session_state.history.append(("bot", resp, list(sorted({d['source'] for d in top_docs}))))

# ------------------------
# 대화 기록
# ------------------------
st.subheader("🧾 대화 기록")
for item in st.session_state.history:
    if item[0] == "user":
        st.markdown(f'<div class="chat-bubble-user"><b>나</b><br>{item[1]}</div>', unsafe_allow_html=True)
    else:
        ans, srcs = item[1], item[2]
        st.markdown(f'<div class="chat-bubble-bot"><b>봇</b><br>{ans}</div>', unsafe_allow_html=True)
        if srcs:
            with st.expander("🔎 참고/근거 문서"):
                for s in srcs:
                    st.markdown(f"- `{s}`")

# ------------------------
# 푸터
# ------------------------
import time as _t
st.markdown(
    f'<p class="tiny">© {_t.strftime("%Y")} 중간고사용 RAG 초안정판 | Secrets에 OPENAI_API_KEY 등록 필수.</p>',
    unsafe_allow_html=True
)


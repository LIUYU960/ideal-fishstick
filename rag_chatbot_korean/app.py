import os
import streamlit as st
from typing import List
from dotenv import load_dotenv

# LangChain 관련
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# OpenAI (LangChain 전용 래퍼)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# PDF 처리
from pypdf import PdfReader

# 기본 설정
load_dotenv()

st.set_page_config(
    page_title="나만의 RAG 챗봇 (중간고사)",
    page_icon="🤖",
    layout="wide"
)

# ---- 사이드바 디자인 ----
st.sidebar.title("⚙️ 설정")
with st.sidebar.expander("📌 안내 (필독)", expanded=True):
    st.markdown(
        """
        **과제 체크리스트**
        - [x] LangChain + Python 라이브러리로 RAG 구현 (업로드 파일 기반)
        - [x] Streamlit UI 및 커스텀 디자인
        - [x] Streamlit Cloud 배포 가능 구조
        - [x] 리포트와 코드 제공
        
        **필수 준비**
        1) OpenAI API 키를 환경변수 `OPENAI_API_KEY`로 등록해 주세요.
        2) 좌측 상단에서 파일을 업로드하고, 임베딩 구축 후 질문을 입력하세요.
        """
    )

# ---- 간단한 스타일(Custom CSS) ----
st.markdown(
    """
    <style>
        .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .chat-bubble-user {background:#eef4ff; padding:14px 16px; border-radius:16px; margin:6px 0; }
        .chat-bubble-bot {background:#f6f6f6; padding:14px 16px; border-radius:16px; margin:6px 0; border-left:4px solid #5b9bd5;}
        .source-card {border:1px solid #e6e6e6; padding:10px; border-radius:12px; margin-top:8px;}
        .tiny {font-size: 12px; color:#666;}
        .pill {display:inline-block; padding:3px 8px; border-radius:999px; background:#edf2f7; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- 세션 상태 ----
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "history" not in st.session_state:
    st.session_state.history = []

# ---- 헤더 ----
st.title("🤖 나만의 RAG 챗봇")
st.caption("LangChain + Streamlit | 파일 업로드 기반 검색 증강 생성 (RAG)")

# ---- 파일 업로드 ----
uploaded_files = st.file_uploader(
    "📄 참고할 파일을 업로드하세요 (PDF/TXT, 여러 개 선택 가능)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    chunk_size = st.number_input("청크 크기", min_value=200, max_value=3000, step=100, value=1000)
with col2:
    chunk_overlap = st.number_input("청크 중첩", min_value=0, max_value=800, step=50, value=150)
with col3:
    top_k = st.slider("검색 문서 수 (k)", min_value=1, max_value=10, value=4)

build_clicked = st.button("🔨 임베딩/인덱스 구축")

def read_pdf(file) -> str:
    pdf = PdfReader(file)
    texts = []
    for page in pdf.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def load_documents(files) -> List[Document]:
    docs = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        if name.lower().endswith(".pdf"):
            text = read_pdf(f)
        elif name.lower().endswith(".txt"):
            text = read_txt(f)
        else:
            continue
        metadata = {"source": name}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def build_vectorstore(docs: List[Document], chunk_size:int, chunk_overlap:int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()  # OPENAI_API_KEY 필요
    vs = FAISS.from_documents(splits, embedding=embeddings)
    return vs

if build_clicked:
    if not uploaded_files:
        st.warning("먼저 파일을 업로드하세요.")
    else:
        with st.spinner("임베딩을 생성하고 인덱스를 구축하는 중..."):
            try:
                st.session_state.docs = load_documents(uploaded_files)
                st.session_state.vectorstore = build_vectorstore(st.session_state.docs, chunk_size, chunk_overlap)
                st.success("완료! 이제 질문을 해보세요.")
            except Exception as e:
                st.error(f"인덱스 구축 중 오류: {e}")

# ---- 질문 UI ----
st.subheader("💬 질문하기")
question = st.text_input("무엇을 도와드릴까요? (예: 이 문서의 목적은 무엇인가요?)")
ask = st.button("질문 보내기")

def build_chain(top_k:int):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": int(top_k)})
    template = """
당신은 업로드된 자료를 근거로 정확하고 간결하게 한국어로 답변하는 조교입니다.
반드시 근거(출처 파일명)를 요약 끝에 '근거:' 뒤에 나열하세요.
사용자 질문: {question}
자료를 바탕으로 한 한국어 답변:"""
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa

if ask:
    if st.session_state.vectorstore is None:
        st.warning("먼저 파일 업로드 후 '임베딩/인덱스 구축'을 눌러주세요.")
    elif not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("답변 생성 중..."):
            try:
                qa = build_chain(top_k)
                result = qa({"query": question})
                answer = result["result"]

                # 간단한 출처 표시용
                sources = []
                try:
                    docs = st.session_state.vectorstore.similarity_search(question, k=int(top_k))
                    for d in docs:
                        sources.append(d.metadata.get("source", "unknown"))
                except Exception:
                    pass

                st.session_state.history.append(("user", question))
                st.session_state.history.append(("bot", answer, sources))

            except Exception as e:
                st.error(f"질문 처리 중 오류: {e}")

# ---- 대화 표시 ----
st.subheader("🧾 대화 기록")
for item in st.session_state.history:
    if item[0] == "user":
        st.markdown(f'<div class="chat-bubble-user"><b>나</b><br>{item[1]}</div>', unsafe_allow_html=True)
    else:
        answer, srcs = item[1], item[2]
        st.markdown(f'<div class="chat-bubble-bot"><b>봇</b><br>{answer}</div>', unsafe_allow_html=True)
        if srcs:
            with st.expander("🔎 참고/근거 문서"):
                for s in srcs:
                    st.markdown(f"- `{s}`")

# ---- 푸터 ----
import time as _t
st.markdown(
    '<p class="tiny">© {year} 중간고사용 RAG 데모 | 배포 팁: 좌측 안내를 참고하여 Streamlit Cloud에 업로드하세요.</p>'.format(year=_t.strftime("%Y")),
    unsafe_allow_html=True
)
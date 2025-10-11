# LIUYU 중간고사 과제 — LangChain/LangGraph 기반 RAG 챗봇

## 제출 항목
1) **소스 코드 ZIP**: `LIUYU_중간고사_코드.zip` (본 레포 폴더 전체 포함, README 포함)
2) **에이전트 워크플로우 이미지**: `LIUYU_중간고사_워크플로우.png`
3) **배포 링크**: Streamlit Cloud에 업로드 후 생성된 URL
4) **추가 기능 구현 요소 텍스트**: 아래 예시 문구 복사/제출

### 추가 기능 구현 요소 (예시 문구)
- 웹 검색 노드 구현 (DuckDuckGo 기반)
- `add_conditional_edges` 총 3개 구현 (질문 유형 라우팅, 검증 실패 시 재질의, 최신성 분기)
- ReAct 기능 노드 구현 (도구: RAG 검색, 웹검색, 계산기)
- Hybrid search 노드 구현 (BM25 키워드 + FAISS 임베딩)
- Reranking 노드 구현 (코사인 유사도/휴리스틱 기반 재정렬)
- 각 노드의 결과 검증 노드 구현 (간단한 heuristic + LLM)
- 최종 답변 검증 노드 구현 (OpenAI `o4-mini` 연동 옵션)

**최대 3개까지만 가산점 인정**이므로, 제출 시 상위 3~4개만 선택해 적어도 됩니다.

---

## 로컬 실행
```bash
cd app
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # 키 입력
streamlit run app_streamlit.py
```

## Streamlit Cloud 배포
- 이 폴더를 GitHub로 푸시한 뒤, Streamlit Cloud에서 repo 선택 →
  **Main file path**: `app/app_streamlit.py` 로 설정
- Secrets에 `.env` 값(OPENAI_API_KEY 등) 입력

## 환경변수 (.env)
```
OPENAI_API_KEY=sk-...
# 선택: o4-mini를 이용한 최종답변 검증을 활성화하려면 아래 값 필요
OPENAI_MODEL_VALIDATOR=o4-mini
```

## 구성 개요
- `app_streamlit.py` — Streamlit UI, LangGraph 실행
- `graph/workflow.py` — 그래프/노드/엣지 정의 (필수요소 모두 포함)
- `tools/retrieval.py` — 문서 적재/벡터화/하이브리드 검색/재순위
- `tools/search.py` — DuckDuckGo 웹검색 Tool
- `validators.py` — 노드별/최종 답변 검증 로직
- `sample_data/` — 샘플 문서

## 필수 구현체 체크
- Agent 노드 3개 이상: `RAGAgent`, `WebSearchAgent`, `ReActAgent` (추가로 `AnswerComposer` 포함)
- `add_conditional_edges` 1개 이상: `needs_recent?`, `needs_math?`, `failed_validation?` 등 3개 구현
- 추가기능: ReAct, 웹검색, hybrid search, reranking, 각 노드 결과 검증, 최종 답변 검증(o4-mini)

## 주의
- OpenAI 키가 없으면 임베딩/검증 LLM이 동작하지 않을 수 있습니다. (임베딩은 대체 경로로 BM25만 사용 가능)
- Streamlit Cloud에서 `faiss-cpu` 설치 시간이 길 수 있습니다. 필요 시 `Chroma`로 교체 가능.
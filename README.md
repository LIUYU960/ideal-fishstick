# LIUYU — 중간고사 과제 (LangChain + LangGraph + RAG + Streamlit)

## 필수 구현요소
- 3개 이상 Agent 노드: retrieve, rerank, websearch, react, synthesize, validate
- add_conditional_edges 1개 이상: rerank 이후 분기, validate 이후 분기

## 추가기능 (가산점, 총 3개)
1) ReAct 기능 노드 구현 (react_node)
2) 웹 검색 노드 구현 (websearch_node)
3) reranking 노드 구현 (rerank_node)

## 실행
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"   # 선택
streamlit run app_streamlit.py
```

## 배포
- GitHub 업로드 → Streamlit Cloud에서 `app_streamlit.py` 지정
- Secrets에 `OPENAI_API_KEY` 필요 시 추가

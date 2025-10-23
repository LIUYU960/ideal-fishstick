# 🤖 중간고사용 RAG 챗봇 (초안정판: BM25)

**설치 실패 0 지향** 버전입니다. 벡터DB 대신 **BM25**를 사용해 Streamlit Cloud에서 의존성 설치가 매우 안정적입니다.

## ✅ 체크리스트
- [x] LangChain + 업로드 파일 RAG
- [x] Streamlit UI 커스텀
- [x] Streamlit Cloud 배포 구조
- [x] 국문 리포트 동봉

## 폴더
```
.
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ README_ko.md
└─ .streamlit/
   └─ config.toml
```

## 환경변수
- `OPENAI_API_KEY`

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 배포
- GitHub 업로드 → Streamlit Cloud → New app
- **Secrets**에 `OPENAI_API_KEY` 등록
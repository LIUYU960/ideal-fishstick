# 🤖 중간고사용 RAG 챗봇 (LangChain + Streamlit)

본 저장소는 **업로드한 파일을 기반으로 검색 증강 생성(RAG)** 을 수행하는 챗봇 데모입니다.  
**LangChain**과 **Streamlit**을 사용하였으며, **Streamlit Cloud 배포**를 전제로 구성되어 있습니다.

## ✅ 과제 체크리스트
- [x] LangChain 및 Python 라이브러리를 활용한 RAG (업로드 파일 형태)
- [x] Streamlit UI 커스텀 디자인
- [x] Streamlit Cloud 배포 구조
- [x] 리포트(Word) 동봉

## 📦 폴더 구조
```
.
├─ app.py                # Streamlit 앱
├─ requirements.txt
├─ README_ko.md
├─ utils.py              # 추후 확장용
└─ .streamlit/
   └─ config.toml        # 테마 설정
```

## 🔑 환경변수
- `OPENAI_API_KEY` : OpenAI API 키 (필수)

로컬에서는 `.env` 파일을 만들어 다음처럼 작성하세요:
```
OPENAI_API_KEY=sk-...yourkey...
```

## ▶️ 실행 방법 (로컬)
```bash
pip install -r requirements.txt
streamlit run app.py
```
브라우저에서 `http://localhost:8501` 로 접속합니다.

## ☁️ 배포 (Streamlit Cloud)
1. 이 레포를 GitHub에 업로드
2. Streamlit Cloud에서 **New app** → 해당 레포 선택
3. Python 버전은 기본(3.12 권장), `requirements.txt` 자동 설치
4. **Advanced settings**에서 `OPENAI_API_KEY`를 Secrets에 등록
5. 배포 후 앱 링크를 과제에 제출

## 📄 사용 방법
1. 좌측에서 PDF/TXT 파일 업로드 (여러 개 가능)  
2. `임베딩/인덱스 구축` 클릭  
3. 질문 입력 후 `질문 보내기`  
4. 답변과 함께 근거(출처 파일명)가 표시됩니다.

## 🎨 디자인 포인트
- 테마 색상/버튼/말풍선 형태 적용
- 파일 수, 청크 크기/중첩, k 설정 슬라이더 제공

## 🧩 확장 아이디어(선택)
- 대화 메모리/요약/평가 노드 추가
- 리랭킹 및 하이브리드 검색(키워드+벡터)
- 감정/톤 조절 파라미터
- LangGraph로 멀티노드 워크플로우 구성

## ⚠️ 주의
- OpenAI 과금/쿼터 상태를 확인하세요.
- 인코딩 문제 시 TXT는 UTF-8로 저장 권장.
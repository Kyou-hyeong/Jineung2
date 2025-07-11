# 🧠 Jineung2: 실용형 한국어 AI 비서

Jineung2는 한국어 기반의 질문에 똑똑하게 대답하는 **LLM 기반 AI 비서**입니다.  
SentencePiece, PyTorch, KoNLPy 기반의 인코더-디코더 구조로 질문 응답, 일정 추천, 간단한 계산 등 실용적인 응답을 생성할 수 있습니다.

---

## 📁 프로젝트 구조

```
llm_project/   (== Jineung2)
├── data/
│   ├── raw/              # source.txt, target.txt, QnA JSON
│   └── processed/        # SentencePiece 학습 결과, 전처리본
├── models/
│   ├── model.py          # Encoder-Decoder 모델 정의
│   └── checkpoints/      # 학습된 .pth 파일 저장
├── tokenizer/
│   ├── train_spm.py      # SentencePiece 학습 스크립트
│   └── tokenizer.py      # SPM 래퍼: encode/decode 함수
├── dataset/
│   └── qna_dataset.py    # 학습용 PyTorch Dataset 클래스
├── train/
│   └── train.py          # 학습 루프 실행 스크립트
├── inference/
│   └── generate.py       # 질문 → 응답 생성기
├── utils/
│   └── config.py         # 경로, 토큰 ID, 하이퍼파라미터 등 설정
├── Dockerfile            # 전체 환경 구축용
├── requirements.txt      # Python 의존성 목록
├── README.md             # 문서 파일
└── notebooks/            # 분석 및 실험용 Jupyter 노트북
```

---

## 🚀 실행 방법

### 🔧 1. Docker로 실행

```bash
docker build -t jineung2 .
docker run --gpus all -it -v $(pwd):/workspace jineung2
```

### 🔧 2. SentencePiece 토크나이저 학습

```bash
cd tokenizer
python train_spm.py --input ../data/raw/qna_virtual_assistant_1000.json
```

### 🔧 3. 데이터셋 생성

```bash
cd dataset
python build_dataset.py  # source.txt, target.txt 기반
```

### 🔧 4. 모델 학습

```bash
cd train
python train.py --epochs 100 --batch_size 32
```

### 🔧 5. 질문 → 응답 생성

```bash
cd inference
python generate.py
```

---

## ✨ 주요 기능

- ✅ 질문-응답 학습 기반 인코더-디코더 구조
- ✅ 실용형 비서 스타일 질문 (날씨, 일정, 알람, 계산, 감정 위로 등)
- ✅ SentencePiece 기반 토크나이저
- ✅ KoNLPy 기반 형태소 분석기 (Kkma)
- ✅ PyTorch 기반 학습 및 추론 지원
- ✅ Docker로 한 번에 환경 세팅

---

## 🔍 예시

```txt
질문 ▶ 오늘 날씨 어때?
🤖 오늘은 맑고 기온은 24도예요.

질문 ▶ 5시에 알람 맞춰줘.
🤖 오후 5시에 알람을 설정했어요.
```

---

## 🛠 주요 라이브러리

- [PyTorch](https://pytorch.org/)
- [KoNLPy](https://konlpy.org/)
- [SentencePiece](https://github.com/google/sentencepiece)
- [pykospacing](https://github.com/haven-jeon/PyKoSpacing)
- [pyspellchecker](https://github.com/barrust/pyspellchecker)

---

## 🧩 향후 확장 계획

- [ ] Web UI (Gradio or FastAPI)
- [ ] 대화 이력 기반 상시 메모리 추가
- [ ] 다국어 대응 (영어 + 일본어)
- [ ] GPT 스타일 디코딩 방식 (Top-k, Top-p)
- [ ] Instruction-tuning + 미세조정 지원

---

## 📜 라이선스

MIT License

---

## 🙋‍♂️ 기여

이 프로젝트는 개인 연구 및 학습을 위한 목적이며, 누구나 포크하고 개선할 수 있습니다.  
기여하고 싶으시다면 PR 또는 이슈를 자유롭게 등록해주세요!
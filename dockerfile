# ------------------------------------------------------------
# 1) Base Image  (GPU CUDA 런타임)  ────────────────────────────
#    - GPU가 없다면 아래 라인을 주석처리하고
#      'python:3.10-slim' 정도로 교체하세요.
# ------------------------------------------------------------
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ------------------------------------------------------------
# 2) 시스템 패키지 설치
#    - KoNLPy(Kkma)  → Java 필요
#    - build-essential / wget 등 빌드·다운로드 도구
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        openjdk-11-jdk \
        wget \
        git \
        unzip \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/

# ------------------------------------------------------------
# 3) Python 라이브러리 설치
# ------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
        sentencepiece \
        konlpy \
        pykospacing \
        pyspellchecker \
        tqdm pandas numpy \
        torchtext==0.17.0 \
        fastapi uvicorn[standard] gradio \
        # ↓ 필요 시 추가
        transformers accelerate bitsandbytes

# ------------------------------------------------------------
# 4) KoNLPy Kkma 초기화(데이터 다운로드) ─ optional
#    - 최초 실행 시에도 자동 다운로드되지만,
#      Docker build 단계에서 다운로드해두면 런타임이 빠름
# ------------------------------------------------------------
RUN python - <<'PY'
from konlpy.tag import Kkma
print("📥  KoNLPy Kkma 데이터 다운로드 중 …")
_ = Kkma()
PY

# ------------------------------------------------------------
# 5) 작업 디렉터리 및 소스 복사
# ------------------------------------------------------------
WORKDIR /workspace/llm_project
COPY . /workspace/llm_project

# ------------------------------------------------------------
# 6) 기본 실행 명령
# ------------------------------------------------------------
CMD ["/bin/bash"]

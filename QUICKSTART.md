# Quick Start Guide

빠르게 시작하는 3단계 가이드입니다.

## 가장 빠른 방법 (Ollama + uv)

**[QUICKSTART_OLLAMA.md](QUICKSTART_OLLAMA.md)를 확인하세요!**

Ollama와 uv를 사용하면 5분 안에 실행할 수 있습니다.

## 1. 간단 설치 (Ollama + uv)

### macOS / Linux

```bash
# 1. Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 2. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Ollama 모델 다운로드
ollama serve  # 별도 터미널
ollama pull midm-2.0-q8_0:base

# 4. 프로젝트 설정
uv sync
cp .env.example .env

# 5. 서버 실행
./run.sh
```

### 기존 방식 (pip)

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 설정
cp .env.example .env

# 4. 서버 실행
./run.sh
```

### Windows

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 설정
copy .env.example .env

# 4. 서버 실행
python -m server.main
```

## 2. 최소 설정 (.env)

`.env` 파일은 기본 설정으로 동작합니다:

```bash
# LLM (Ollama)
LLM_BACKEND=ollama
LLM_MODEL_PATH=midm-2.0-q8_0:base
OLLAMA_HOST=http://localhost:11434

# STT (CPU)
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# TTS
TTS_ENGINE=piper
```

## 3. 브라우저 접속

```
http://localhost:8000/client
```

1. "Connect" 버튼 클릭
2. 마이크 권한 허용
3. 말하기 시작!

## 빠른 테스트

처음 실행 시 텍스트 입력으로 테스트:

1. 하단 텍스트 박스에 "안녕하세요" 입력
2. "Send Text" 버튼 클릭
3. AI 응답이 음성으로 재생됨

## 다음 단계

- GPU 가속: [README.md#GPU-활용](README.md#gpu-활용)
- 성능 최적화: [README.md#최적화-가이드](README.md#최적화-가이드)
- 트러블슈팅: [README.md#트러블슈팅](README.md#트러블슈팅)

## Docker로 실행

더 간단한 방법:

```bash
# Docker Compose로 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

## 주의사항

### 필수 요구사항

1. **마이크**: 웹 브라우저에서 마이크 접근 권한 필요
2. **HTTPS**: 로컬호스트 외에는 HTTPS 필요 (브라우저 보안 정책)
3. **메모리**: 최소 8GB RAM

### 첫 실행 시

- Whisper 모델 자동 다운로드: ~500MB
- Silero VAD 모델 다운로드: ~2MB
- 첫 실행은 다소 느릴 수 있음 (모델 로딩)

### LLM 설정

**Option 1: Ollama (기본, 권장)**

```bash
# Ollama 설치 및 실행
ollama serve
ollama pull midm-2.0-q8_0:base

# .env는 기본 설정 사용
LLM_BACKEND=ollama
LLM_MODEL_PATH=midm-2.0-q8_0:base
```

**Option 2: 다른 Ollama 모델**

```bash
# 한국어 특화 모델
ollama pull beomi/llama3-ko-8b

# .env 수정
LLM_MODEL_PATH=beomi/llama3-ko-8b
```

**Option 3: vLLM (GPU 필요)**

```bash
# GPU와 충분한 메모리 필요 (최소 10GB)
uv pip install vllm

# .env 설정
LLM_BACKEND=vllm
LLM_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
```

## 성능 기대치

### CPU 모드 (기본)
- 응답 시간: 2-4초
- CPU 사용률: 50-80%
- 메모리: 4-6GB

### GPU 모드
- 응답 시간: 1-2초
- GPU 사용률: 30-50%
- 메모리: 8-12GB

## 문제 해결

### "ModuleNotFoundError"

```bash
# 가상환경이 활성화되어 있는지 확인
which python  # venv/bin/python이어야 함

# 재설치
pip install --upgrade -r requirements.txt
```

### "포트 8000이 이미 사용 중"

```bash
# .env에서 포트 변경
PORT=8001
```

### "마이크를 찾을 수 없음"

- 시스템 설정에서 마이크 권한 확인
- 브라우저를 새로고침하고 다시 권한 요청
- HTTPS 사용 또는 localhost에서 실행

## 다음 읽을거리

- 전체 문서: [README.md](README.md)
- 아키텍처: [README.md#시스템-아키텍처](README.md#시스템-아키텍처)
- 최적화: [README.md#최적화-가이드](README.md#최적화-가이드)

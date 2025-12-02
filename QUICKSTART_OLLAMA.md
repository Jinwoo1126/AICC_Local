# Quick Start Guide - Ollama + uv

**Ollama**와 **uv**를 사용한 가장 빠른 시작 가이드입니다.

## 사전 준비 (5분)

### 1. Ollama 설치

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 또는 https://ollama.com에서 다운로드
```

### 2. Ollama 모델 다운로드

```bash
# Ollama 서버 시작 (별도 터미널)
ollama serve

# midm-2.0 모델 다운로드 (약 4.7GB)
ollama pull midm-2.0-q8_0:base
```

### 3. uv 설치 (Python 패키지 매니저)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 터미널 재시작 후 확인
uv --version
```

## 빠른 실행 (3단계)

### 1. 프로젝트 설정

```bash
# 의존성 설치 (uv가 자동으로 가상환경 생성)
uv sync

# 환경 설정
cp .env.example .env
```

### 2. .env 파일 확인

기본 설정으로 바로 동작합니다:

```bash
# 주요 설정
LLM_BACKEND=ollama
LLM_MODEL_PATH=midm-2.0-q8_0:base
OLLAMA_HOST=http://localhost:11434

WHISPER_MODEL=base
WHISPER_DEVICE=cpu
TTS_ENGINE=piper
```

### 3. 서버 실행

```bash
# 자동 실행 스크립트 (모든 것을 체크)
./run.sh

# 또는 수동 실행
uv run python -m server.main
```

## 브라우저 접속

```
http://localhost:8000/client
```

1. **Connect** 버튼 클릭
2. 마이크 권한 허용
3. 말하기 시작!

## Ollama 장점

### 빠른 응답 속도
- **Time To First Token**: 100-300ms
- **전체 응답**: 1-2초
- CPU로도 실시간 대화 가능

### 쉬운 설치
- 복잡한 GPU 설정 불필요
- 단일 명령어로 모델 관리
- 자동 메모리 관리

### 모델 전환 간편

```bash
# 다른 모델 시도
ollama pull llama3.1:8b
ollama pull qwen2.5:7b

# .env에서 변경
LLM_MODEL_PATH=llama3.1:8b
```

## uv 장점

### 매우 빠른 설치
- pip보다 10-100배 빠름
- 병렬 다운로드 및 설치

### 자동 가상환경 관리
```bash
uv sync          # 의존성 설치 + 가상환경 생성
uv add <package> # 패키지 추가
uv run <cmd>     # 자동으로 가상환경에서 실행
```

### 의존성 잠금
- `uv.lock` 파일로 정확한 버전 관리
- 재현 가능한 환경

## 트러블슈팅

### Ollama 연결 실패

```bash
# Ollama 상태 확인
curl http://localhost:11434/api/tags

# Ollama 재시작
pkill ollama
ollama serve
```

### 모델이 없음

```bash
# 사용 가능한 모델 확인
ollama list

# 모델 다운로드
ollama pull midm-2.0-q8_0:base
```

### uv 명령어 없음

```bash
# uv 재설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 환경 변수 확인
source ~/.bashrc  # 또는 ~/.zshrc
```

### Python 버전 문제

```bash
# uv로 특정 Python 버전 사용
uv python install 3.11
uv python pin 3.11
uv sync
```

## 성능 튜닝

### CPU 최적화

```bash
# .env 설정
WHISPER_MODEL=tiny           # 더 빠른 STT
WHISPER_COMPUTE_TYPE=int8    # 양자화
LLM_MAX_TOKENS=256           # 짧은 응답
```

**예상 성능**: 1.5-2.5초 전체 응답

### GPU 사용 (선택)

```bash
# .env 설정
WHISPER_DEVICE=cuda

# CUDA PyTorch 설치
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**예상 성능**: 0.8-1.5초 전체 응답

## 한국어 모델 추천

```bash
# 한국어 특화 모델들
ollama pull beomi/llama3-ko-8b
ollama pull LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct

# .env 업데이트
LLM_MODEL_PATH=beomi/llama3-ko-8b
```

## 다음 단계

- 전체 문서: [README.md](README.md)
- 최적화 가이드: [README.md#최적화-가이드](README.md#최적화-가이드)
- Ollama 모델 탐색: https://ollama.com/library

## 주요 명령어 요약

```bash
# 환경 설정
uv sync                              # 의존성 설치
cp .env.example .env                 # 환경 파일 생성

# Ollama
ollama serve                         # 서버 시작
ollama pull midm-2.0-q8_0:base      # 모델 다운로드
ollama list                          # 설치된 모델 확인

# 서버 실행
./run.sh                             # 자동 실행
uv run python -m server.main         # 수동 실행

# 개발
uv add <package>                     # 패키지 추가
uv pip install <package>             # pip 대체
uv run pytest                        # 테스트 실행
```

## 왜 Ollama + uv?

| 기존 방식 | Ollama + uv |
|----------|-------------|
| vLLM 설치 복잡 | Ollama 단순 설치 |
| GPU 필수 | CPU로도 빠름 |
| pip 느린 설치 | uv 초고속 설치 |
| 수동 가상환경 | 자동 관리 |
| 모델 관리 복잡 | `ollama pull` 한 줄 |

**결론**: 가장 쉽고 빠른 온프레미스 실시간 음성 AI!

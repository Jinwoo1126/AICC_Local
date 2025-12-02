# Real-time Voice Assistant

OpenAI Realtime API에 근접한 저지연 음성 대화 시스템을 온프레미스 환경에서 구현한 프로젝트입니다.

## 주요 특징

- **저지연 스트리밍**: 모든 단계(VAD → STT → LLM → TTS)가 파이프라인으로 처리되어 2-4초 응답
- **Multimodal 파이프라인**: 음성 입력부터 음성 출력까지 끊김 없는 처리
- **Barge-in 지원**: 사용자가 AI의 대화를 중단하고 다시 말할 수 있음
- **온프레미스**: 모든 처리가 로컬에서 이루어져 프라이버시 보장
- **Mac 최적화**: Apple Silicon (M1/M2/M3)에서 MLX 프레임워크로 가속

## ⚠️ 현재 알려진 제한사항

- **MLX Whisper 한국어 인식 불가**: Apple Silicon에서 MLX Whisper의 한국어 모드가 작동하지 않습니다
  - 현재 영어 모드로만 동작
  - 한국어 지원이 필요한 경우 OpenAI Whisper 사용 권장 (느리지만 안정적)
- **STT 인식 불안정**: MLX Whisper가 일부 음성을 인식하지 못하는 경우 발생
  - 명확한 발음과 적절한 볼륨 필요
  - 배경 소음 최소화 필요

## 시스템 아키텍처

```
User Speech → VAD (Silero) → STT (MLX Whisper) → LLM (Ollama) → TTS (Piper) → Audio Output
                  ↓                  ↓                  ↓              ↓
              음성 감지         텍스트 변환        텍스트 생성      음성 합성
              (~10ms)          (~200ms)         (1-3초)         (~100ms)
```

### 핵심 기술 스택

| 컴포넌트 | 기술 | 특징 |
|---------|-----|-----|
| **VAD** | Silero VAD | 매우 빠르고 정확한 음성 구간 감지, 16kHz 최적화 |
| **STT** | MLX Whisper (Mac) | Apple Silicon GPU 가속, 영어 모드만 지원 |
| **LLM** | Ollama (midm-2.0-q8_0) | 로컬 실행, 토큰 스트리밍 |
| **TTS** | Piper TTS (en_US-lessac-medium) | 빠른 음성 합성, ONNX 최적화 |
| **Server** | FastAPI + WebSocket | 비동기 처리, 실시간 통신 |
| **환경 관리** | uv | 초고속 Python 패키지 관리자 |

### Mac (Apple Silicon) 특화 설정

- **MLX Whisper**: Apple의 MLX 프레임워크로 M1/M2/M3 최적화
- **Ollama**: Apple Silicon 네이티브 지원
- **주의**: CUDA 설정은 무시됨 (자동으로 Apple Silicon GPU 사용)

## 설치 가이드

### 1. 시스템 요구사항

- **OS**: Linux, macOS, Windows (WSL2)
- **Python**: 3.9 이상
- **Ollama**: 로컬 LLM 실행 (https://ollama.com)
- **uv**: Python 패키지 매니저 (https://docs.astral.sh/uv/)
- **GPU**: CUDA 11.8+ (선택사항, CPU로도 동작)
- **RAM**: 최소 8GB (16GB 권장)
- **Disk**: 10GB 여유 공간 (모델 다운로드용)

### 2. 빠른 설치 (Ollama + uv 사용)

**가장 추천하는 방법입니다!**

```bash
# 1. Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 2. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Ollama 모델 다운로드
ollama serve  # 별도 터미널에서 실행
ollama pull midm-2.0-q8_0:base

# 4. 프로젝트 설정
uv sync  # 의존성 자동 설치 + 가상환경 생성

# 5. 환경 설정
cp .env.example .env

# 6. 서버 실행
./run.sh
```

**자세한 가이드**: [QUICKSTART_OLLAMA.md](QUICKSTART_OLLAMA.md)

### 2-1. 기존 방식 설치 (pip 사용)

```bash
# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 모델 다운로드

#### Whisper 모델 (자동 다운로드)
첫 실행 시 자동으로 다운로드됩니다. 수동 다운로드 시:

```bash
# Faster-Whisper는 자동으로 캐시됨 (~/.cache/huggingface)
# 또는 수동 설치:
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

#### Piper TTS 모델

```bash
# Piper 모델 다운로드 (예: 영어)
mkdir -p models/piper
cd models/piper

# 영어 모델
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json

# 한국어 모델 (사용 가능한 경우)
# 한국어 Piper 모델은 현재 제한적이므로, Coqui TTS 사용 권장
```

#### LLM 모델

**Option 1: Ollama (권장)**

```bash
# Ollama 서버 시작
ollama serve

# 추천 모델
ollama pull midm-2.0-q8_0:base        # 기본 (빠름)
ollama pull llama3.1:8b               # 영어 특화
ollama pull beomi/llama3-ko-8b        # 한국어 특화
ollama pull qwen2.5:7b                # 다국어
```

**Option 2: vLLM (GPU 필요)**

```bash
uv pip install vllm

# Hugging Face에서 모델 다운로드
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

### 4. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 수정 (필수 설정)
nano .env
```

**주요 설정 항목**:

```bash
# LLM 설정 (Ollama - 기본)
LLM_BACKEND=ollama                     # ollama, vllm, or openai
LLM_MODEL_PATH=midm-2.0-q8_0:base     # Ollama 모델명
OLLAMA_HOST=http://localhost:11434    # Ollama 서버 URL

# STT 설정
WHISPER_MODEL=base           # tiny, base, small, medium, large-v3
WHISPER_DEVICE=cpu           # cpu 또는 cuda
WHISPER_COMPUTE_TYPE=int8    # int8 (빠름) 또는 float16 (정확)

# TTS 설정
TTS_ENGINE=piper            # piper 또는 coqui
PIPER_MODEL_PATH=models/piper/en_US-lessac-medium.onnx
```

## 사용 방법

### 서버 시작

```bash
# 방법 1: 자동 실행 스크립트 (권장)
./run.sh

# 방법 2: uv 사용
uv run python -m server.main

# 방법 3: 수동 실행
source .venv/bin/activate  # uv를 사용한 경우
# source venv/bin/activate  # pip를 사용한 경우
python -m server.main

# 방법 4: uvicorn 직접 실행
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

서버가 시작되면:
```
INFO:     Started server on http://0.0.0.0:8000
INFO:     Pipeline initialized successfully
```

### 클라이언트 접속

브라우저에서 `http://localhost:8000/client` 접속

#### 웹 인터페이스 사용법

1. **Connect 버튼 클릭**: 마이크 권한 요청 후 WebSocket 연결
2. **음성으로 대화**: 말하면 자동으로 음성 인식 및 응답 생성
3. **Stop Talking 버튼**: AI가 말하는 중간에 끊기 (Barge-in)
4. **Reset 버튼**: 대화 내역 초기화
5. **텍스트 입력**: 음성 대신 텍스트로 입력 가능

## 최적화 가이드

### 지연 시간 최소화

#### 1. STT 최적화

```bash
# .env 설정
WHISPER_MODEL=base          # large 모델은 느림
WHISPER_COMPUTE_TYPE=int8   # float16보다 빠름
WHISPER_DEVICE=cuda         # GPU 사용 시 훨씬 빠름
```

**예상 지연시간**:
- CPU (int8, base): ~200-500ms
- GPU (int8, base): ~50-150ms

#### 2. LLM 최적화

**vLLM 사용 (권장)**:

```bash
# GPU 필수
pip install vllm

# .env 설정
USE_VLLM=true
LLM_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
LLM_GPU_MEMORY_UTILIZATION=0.9
```

**Time To First Token (TTFT) 최적화**:
- 작은 모델 사용: 7B-8B 모델
- 양자화: AWQ 4-bit 모델 사용
- Prompt 최소화

#### 3. TTS 최적화

**Piper TTS (가장 빠름)**:
- 지연시간: ~50-100ms per chunk
- 품질: 중상급 (약간 로봇 같을 수 있음)

**Coqui TTS (고품질)**:
- 지연시간: ~200-500ms per chunk
- 품질: 상급 (자연스러움)

#### 4. 네트워크 최적화

```python
# server/pipeline.py에서 청크 크기 조정
chunk_size = 5  # 작을수록 빠르지만 품질 저하 가능
```

### GPU 활용

```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# .env 설정
WHISPER_DEVICE=cuda
```

## 트러블슈팅

### 1. 마이크가 인식되지 않음

**브라우저 권한 확인**:
- Chrome: 설정 → 개인정보 및 보안 → 사이트 설정 → 마이크
- HTTPS 필요: 로컬에서는 localhost만 허용됨

### 2. WebSocket 연결 실패

```bash
# 서버 로그 확인
tail -f server.log

# 방화벽 확인
sudo ufw allow 8000/tcp

# 포트 사용 확인
lsof -i :8000
```

### 3. STT가 느리거나 인식이 안됨 (Apple Silicon)

**증상**:
- MLX Whisper가 일부 음성을 인식하지 못하고 빈 문자열 반환
- 한국어 모드에서 인식 실패 (영어로만 인식됨)
- 일부 발화는 인식되고 일부는 안됨

**원인**:
- MLX Whisper의 한국어 지원 불완전
- 음성 품질/볼륨에 민감
- 배경 소음에 취약

**해결책**:

1. **영어 모드 사용** (현재 설정):
   ```python
   # server/main.py
   stt_config = {
       "language": "en",  # 한국어는 "ko"로 변경 (불안정)
   }
   ```

2. **음성 입력 품질 개선**:
   - 마이크에 가까이 대고 명확하게 발음
   - 배경 소음 최소화
   - 적절한 볼륨 유지

3. **더 작은 Whisper 모델 사용**:
   ```bash
   # .env 설정
   WHISPER_MODEL=base  # tiny는 너무 부정확, small은 느림
   ```

4. **대안: OpenAI Whisper 사용** (느리지만 안정적):
   ```bash
   pip install openai-whisper
   # server/pipeline.py에서 MLXWhisperSTT 대신 WhisperSTT 사용
   ```

### 4. LLM 메모리 부족

```bash
# 더 작은 모델 사용
LLM_MODEL_PATH=meta-llama/Llama-3.1-7B-Instruct

# GPU 메모리 제한
LLM_GPU_MEMORY_UTILIZATION=0.7

# 또는 CPU 사용
USE_VLLM=false
```

### 5. LLM 응답이 느림 (주요 병목)

**증상**:
- 전체 응답 시간의 50-70%를 LLM이 차지
- TTFT (Time To First Token)가 1-3초

**해결책**:

1. **더 작은 모델 사용**:
   ```bash
   # Ollama에서 더 작은 모델 다운로드
   ollama pull llama3.2:3b       # 3B 모델 (가장 빠름)
   ollama pull qwen2.5:3b        # 3B 다국어 모델

   # .env 설정
   LLM_MODEL_PATH=llama3.2:3b
   ```

2. **Ollama 설정 최적화**:
   ```bash
   # 더 많은 GPU 레이어 사용 (Apple Silicon)
   export OLLAMA_NUM_GPU=999

   # 컨텍스트 크기 줄이기
   LLM_MAX_TOKENS=256  # .env에서 512 → 256
   ```

3. **vLLM 백엔드 사용** (Linux/NVIDIA GPU만):
   ```bash
   pip install vllm
   # .env 설정
   LLM_BACKEND=vllm
   ```

### 6. TTS 품질 문제

**Piper 모델 변경**:
```bash
# 더 높은 품질 모델 다운로드
# high quality > medium > low quality
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-high.onnx
```

**Coqui TTS 사용**:
```bash
pip install TTS

# .env 설정
TTS_ENGINE=coqui
```

## 성능 벤치마크

### 테스트 환경

**Apple Silicon (현재 구성)**:
- CPU: Apple M1/M2/M3
- RAM: 16GB+
- Framework: MLX (Apple Silicon GPU 가속)

**Intel/NVIDIA (참고)**:
- CPU: Intel i7-12700K
- GPU: NVIDIA RTX 3080 (10GB)
- RAM: 32GB

### End-to-End 지연시간 (실제 측정치)

| 구성 | TTFT | 전체 응답 시간 |
|-----|------|--------------|
| **Apple Silicon (MLX + Ollama)** | **~1-2s** | **2-4s** |
| CPU (base 모델) | ~800ms | 2.5-3.5s |
| GPU (base 모델) | ~400ms | 1.2-1.8s |
| GPU (small 모델, vLLM) | ~250ms | 0.8-1.5s |

### 컴포넌트별 지연시간 (Apple Silicon 기준)

| 컴포넌트 | 지연시간 | 병목 여부 |
|---------|---------|----------|
| VAD (Silero) | ~10ms | ✓ 빠름 |
| STT (MLX Whisper base) | ~200ms | ✓ 빠름 |
| **LLM (Ollama midm-2.0)** | **1-3s** | ⚠️ **주요 병목** |
| TTS (Piper) | ~100-500ms | ✓ 빠름 |

**참고**: LLM이 전체 지연시간의 50-70%를 차지하는 주요 병목입니다.

## API 문서

### WebSocket Protocol

**연결**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

**메시지 타입**:

1. **Audio Chunks (Binary)**:
   - Client → Server: Int16 PCM audio data (16kHz, mono)
   - Server → Client: WAV audio chunks

2. **Control Messages (JSON)**:

```javascript
// Barge-in (중단)
ws.send(JSON.stringify({ type: 'barge_in' }));

// Reset conversation
ws.send(JSON.stringify({ type: 'reset' }));

// Text input
ws.send(JSON.stringify({
    type: 'text',
    content: 'Hello, how are you?'
}));

// Ping (heartbeat)
ws.send(JSON.stringify({ type: 'ping' }));
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Server info
curl http://localhost:8000/
```

## 개발 로드맵

**완료된 기능**:
- [x] 기본 스트리밍 파이프라인
- [x] VAD 통합 (Silero VAD)
- [x] STT 통합 (MLX Whisper for Apple Silicon)
- [x] LLM 스트리밍 (Ollama + vLLM + OpenAI API)
- [x] TTS 스트리밍 (Piper + Coqui)
- [x] Barge-in 기능
- [x] 웹 클라이언트
- [x] Apple Silicon (M1/M2/M3) 최적화

**현재 작업 중**:
- [~] LLM-TTS 토큰 스트리밍 개선 (현재 불안정)
- [~] MLX Whisper 안정성 개선 (한국어 인식 문제)

**향후 계획**:
- [ ] 안정적인 한국어 STT 지원 (OpenAI Whisper 대체)
- [ ] LLM 응답 속도 최적화 (더 작은 모델, 양자화)
- [ ] StyleTTS2 통합 (고품질 TTS)
- [ ] 한국어 TTS 모델 추가
- [ ] Multi-turn 대화 개선 (컨텍스트 관리)
- [ ] Emotion detection
- [ ] Docker 이미지
- [ ] 성능 모니터링 대시보드

## 라이선스

MIT License

## 기여하기

Pull Request와 Issue는 언제나 환영합니다!

## 참고 자료

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [vLLM](https://github.com/vllm-project/vllm)
- [Piper TTS](https://github.com/rhasspy/piper)

## 문의

이슈나 질문은 GitHub Issues를 이용해주세요.

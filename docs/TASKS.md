# 구현 작업 목록

## ✅ Phase 1: 기본 구조 생성 (완료)
- [x] Task 1.1: 프로젝트 폴더 구조 생성
- [x] Task 1.2: requirements.txt 작성
- [x] Task 1.3: README.md 기본 내용

## ✅ Phase 2: 인터페이스 정의 (완료)
- [x] Task 2.1: BaseTTS 추상 클래스 구현 (models/base.py)
- [x] Task 2.2: 타입 힌트 및 docstring 추가

## 🚧 Phase 3: 모델 래퍼 구현 (진행 중)

### ✅ Task 3.1: gTTS 래퍼 (완료)
- 파일: `models/gtts_wrapper.py`
- 클래스: `GTTSWrapper(BaseTTS)`
- 라이브러리: `gtts`
- 특징: Google TTS API, 한국어 완벽 지원

### 🔲 Task 3.2: Zonos 래퍼
**Cursor 프롬프트:**
@models/base.py @config/models_config.yaml 참고
models/zonos.py 구현:

ZonosTTS(BaseTTS)

Zyphra/Zonos-v0.1-transformer

transformers 사용

sample_rate = 44100

text

### 🔲 Task 3.3: CosyVoice2 래퍼
**Cursor 프롬프트:**
@models/base.py 참고
models/cosyvoice.py 구현:

CosyVoiceTTS(BaseTTS)

FunAudioLLM/CosyVoice2-0.5B

sample_rate = 24000

text

### 🔲 Task 3.4: Kokoro 래퍼
**Cursor 프롬프트:**
@models/base.py 참고
models/kokoro.py 구현:

KokoroTTS(BaseTTS)

hexgrad/Kokoro-82M

sample_rate = 22050

text

### 🔲 Task 3.5: MeloTTS 래퍼
**Cursor 프롬프트:**
@models/base.py 참고
models/melotts.py 구현:

MeloTTSKorean(BaseTTS)

melo-tts 라이브러리

language='KR'

sample_rate = 44100

text

## ✅ Phase 4: 성능 측정 (완료)
- [x] Task 4.1: PerformanceMonitor 클래스 (utils/performance_monitor.py)

## ✅ Phase 5: 테스트 러너 (완료)
- [x] Task 5.1: BenchmarkRunner 클래스 (tests/test_runner.py)
- [x] Task 5.2: test_sentences.txt 샘플 문장
- [x] Task 5.3: run_benchmark.py CLI 스크립트

## 🔲 Phase 6: test_runner.py 통합

### Task 6.1: initialize_models() 업데이트
**Cursor 프롬프트:**
@tests/test_runner.py 수정
initialize_models() 함수를 업데이트해서 5개 모델 모두 지원:

gtts, zonos, cosyvoice, kokoro, melotts

각 모델 import 및 초기화

에러 핸들링 포함

text

## 🎯 Phase 7: 실행 및 분석

### Task 7.1: 벤치마크 실행
python scripts/run_benchmark.py --iterations 3

text

### Task 7.2: 결과 분석
cat results/metrics/summary_statistics_*.csv

text

---

## 📋 빠른 실행 체크리스트

현재 남은 작업:
- [ ] models/zonos.py 생성 (Cursor)
- [ ] models/cosyvoice.py 생성 (Cursor)
- [ ] models/kokoro.py 생성 (Cursor)
- [ ] models/melotts.py 생성 (Cursor)
- [ ] tests/test_runner.py 업데이트 (Cursor)
- [ ] python scripts/run_benchmark.py 실행

## 🚀 한 번에 실행하는 Cursor 프롬프트

@docs/SPEC.md @docs/PLAN.md @docs/TASKS.md @models/base.py @config/models_config.yaml @tests/test_runner.py

Phase 3~6 전체 구현:

models/zonos.py (ZonosTTS)

Zyphra/Zonos-v0.1-transformer

models/cosyvoice.py (CosyVoiceTTS)

FunAudioLLM/CosyVoice2-0.5B

models/kokoro.py (KokoroTTS)

hexgrad/Kokoro-82M

models/melotts.py (MeloTTSKorean)

melo-tts 라이브러리

tests/test_runner.py 업데이트

initialize_models()에 5개 모델 추가

@models/gtts_wrapper.py 패턴 참고.
.cursorrules 규칙 준수.
모든 파일 동시 생성.

text

완료 후 실행:
python scripts/run_benchmark.py --iterations 2

text

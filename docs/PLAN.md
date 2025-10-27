text
# 기술 계획

## 기술 스택
### 핵심 라이브러리
- Python 3.10+
- torch, torchaudio
- transformers, accelerate

### 성능 측정
- psutil (CPU/메모리)
- GPUtil (GPU 모니터링)

### 데이터 처리
- pandas, numpy
- scipy, librosa
- soundfile

## 아키텍처 설계

### 핵심 컴포넌트
BaseTTS (추상 클래스)
├── load_model()
├── synthesize(text) -> audio
├── save_audio(audio, path)
└── get_model_info() -> dict

PerformanceMonitor (컨텍스트 매니저)
├── measure() -> metrics
└── calculate_rtf()

BenchmarkRunner (오케스트레이터)
├── initialize_models()
├── run_single_test()
└── save_results()

text

### 데이터 흐름
1. 테스트 문장 로드 (test_sentences.txt)
2. 각 모델에 대해:
   - PerformanceMonitor 시작
   - TTS 모델로 오디오 생성
   - 메트릭 수집 (시간, 메모리)
   - 오디오 저장
3. 결과 집계 및 CSV 저장
4. 시각화 그래프 생성

## 제약 조건
- GPU 없이도 CPU에서 실행 가능
- 각 모델은 독립적으로 테스트 가능
- 메모리 누수 방지 (테스트 후 모델 언로드)

## 디렉토리 규칙
- 모델 래퍼: `models/`
- 유틸리티: `utils/`
- 설정 파일: `config/`
- 결과물: `results/`
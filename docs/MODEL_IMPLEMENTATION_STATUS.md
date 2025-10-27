# TTS 모델 구현 상태

## 개요

이 프로젝트는 5개의 한국어 TTS 모델을 벤치마크하기 위한 시스템입니다.

## 모델 구현 상태

### ✅ 완전히 구현됨

#### 1. Google TTS (gTTS)
- **클래스**: `GTTSWrapper`
- **파일**: `models/gtts_wrapper.py`
- **타입**: API 기반
- **샘플레이트**: 24000 Hz
- **상태**: ✅ 완전히 동작
- **성능**: RTF 평균 0.12 (목표 달성 🎉)
- **특징**:
  - 인터넷 연결 필요
  - 별도 모델 다운로드 불필요
  - 한국어 완벽 지원
  - 속도 조절 가능 (slow 옵션)

#### 2. MeloTTS (myshell-ai/MeloTTS-Korean)
- **클래스**: `MeloTTSKorean`
- **파일**: `models/melotts.py`
- **타입**: pip 패키지
- **샘플레이트**: 44100 Hz
- **상태**: ✅ 완전히 동작
- **성능**: RTF 평균 0.18 (목표 달성 🎉)
- **설치**: `pip install melo-tts`
- **특징**:
  - MIT 라이선스
  - 한국어 전용
  - CPU/CUDA 지원
  - 고품질 음성 합성

#### 3. Zonos (Zyphra/Zonos-v0.1-transformer)
- **클래스**: `ZonosTTS`
- **파일**: `models/zonos.py`
- **타입**: GitHub 저장소 (pip installable)
- **샘플레이트**: 44100 Hz
- **상태**: ✅ 완전히 동작
- **성능**: RTF 평균 0.52 (첫 실행 제외, 실시간보다 2배 빠름 🎉)
- **설치**: 
  ```bash
  git clone https://github.com/Zyphra/Zonos.git /tmp/Zonos
  cp -r /tmp/Zonos/zonos ~/.local/lib/python3.10/site-packages/
  pip install torch torchaudio transformers Pillow
  ```
- **특징**:
  - 200k시간 학습 데이터
  - 고품질 음성 합성 (44.1kHz)
  - bfloat16 지원 (CUDA)
  - cfg_scale, max_new_tokens 파라미터
  - 첫 실행 시 torch.compile 최적화 (느림)
  - 이후 실행은 매우 빠름
- **주의**: 
  - 한국어 지원이 제한적 (영어 중심 모델)
  - GPU 메모리 약 3.6GB 필요
  - zonos 라이브러리가 설치되지 않으면 건너뜀

### ⚠️ 부분 구현 (구조만 완성)

#### 4. CosyVoice (FunAudioLLM/CosyVoice-300M)
- **클래스**: `CosyVoiceTTS`
- **파일**: `models/cosyvoice.py`
- **타입**: 공식 라이브러리 (GitHub)
- **샘플레이트**: 24000 Hz
- **상태**: ✅ 구현 완료 (CosyVoice 라이브러리 필요)
- **설치**:
  ```bash
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
  cd CosyVoice
  conda create -n cosyvoice python=3.8
  conda activate cosyvoice
  pip install -r requirements.txt
  ```
- **구현 내용**:
  - ✅ `__init__()`: 완전 구현
  - ✅ `load_model()`: cosyvoice.cli.cosyvoice.CosyVoice 사용
  - ✅ `synthesize()`: 
    - SFT 모드 (inference_sft): 화자 선택
    - Zero-shot 모드 (inference_zero_shot): 프롬프트 음성 사용
  - ✅ `_load_wav()`: 프롬프트 음성 로드 및 리샘플링
  - ✅ 한국어 언어 태그 자동 추가 (`<|ko|>`)
  - ✅ 타입 힌트 및 Google 스타일 docstring
  - ✅ 에러 핸들링 (ImportError, RuntimeError, ValueError)
- **특징**:
  - **다국어 지원**: 한국어, 중국어, 영어, 일본어, 광동어 (`<|ko|>`, `<|zh|>`, `<|en|>`, `<|jp|>`, `<|yue|>`)
  - **SFT 모드**: 사전 훈련된 화자 선택
  - **Zero-shot 음성 복제**: 3초 프롬프트로 화자 복제
  - **Cross-lingual**: 언어 간 변환
  - **Instruct 모드**: 감정 제어 (`<strong>`, `<laughter>`, `[breath]`)
  - **fp16 지원**: CUDA 사용 시 자동 최적화
  - **Streaming**: 실시간 스트리밍 지원
- **주의**: 
  - 공식 라이브러리 설치 필수
  - GitHub: https://github.com/FunAudioLLM/CosyVoice
  - Python 3.8 권장
  - 모델 경로: `pretrained_models/CosyVoice-300M` 또는 `iic/CosyVoice-300M`

#### 5. Kokoro (hexgrad/Kokoro-82M)
- **클래스**: `KokoroTTS`
- **파일**: `models/kokoro.py`
- **타입**: Hugging Face
- **샘플레이트**: 22050 Hz
- **상태**: 구조 완성, 실제 구현 대기
- **필요 작업**:
  - Kokoro 라이브러리 설치
  - GitHub: https://github.com/hexgrad/kokoro
  - `load_model()` 실제 구현
  - `synthesize()` 실제 구현
- **특징**: 초경량 82M 파라미터 모델


## 구현 패턴

모든 모델은 동일한 구조를 따릅니다:

```python
class ModelTTS(BaseTTS):
    def __init__(self, ...):
        # config 로드
        # 설정 초기화
        # 출력 디렉토리 생성
        pass
    
    def load_model(self):
        # 모델 로드 (Hugging Face, pip, 또는 API)
        pass
    
    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        # 텍스트 → 오디오 변환
        # numpy array 반환
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        # 모델 정보 반환
        pass
```

## 모델 추가하는 방법

### 1. 새로운 모델 래퍼 생성

`models/your_model.py`:
```python
from models.base import BaseTTS

class YourModelTTS(BaseTTS):
    # gtts_wrapper.py를 참고하여 구현
    pass
```

### 2. models/__init__.py 업데이트

```python
from models.your_model import YourModelTTS

__all__ = [..., "YourModelTTS"]
```

### 3. config/models_config.yaml에 설정 추가

```yaml
your_model:
  model_path: "..."
  sample_rate: 22050
  output_dir: "data/output/your_model"
  config:
    # 모델별 설정
```

### 4. tests/test_runner.py의 initialize_models() 업데이트

```python
elif model_name == 'your_model':
    from models.your_model import YourModelTTS
    model = YourModelTTS()
    model.load_model()
    self.models[model_name] = model
```

## 현재 벤치마크 실행

현재 gtts와 melotts가 완전히 구현되어 있습니다:

```bash
# gtts만 테스트
python scripts/run_benchmark.py --models gtts

# melotts만 테스트
python scripts/run_benchmark.py --models melotts

# gtts와 melotts 비교
python scripts/run_benchmark.py --models gtts melotts

# 전체 모델 시도 (gtts, melotts 성공, 나머지는 건너뜀)
python scripts/run_benchmark.py
```

### 최근 벤치마크 결과 (2024-10-27)

| 모델 | 평균 RTF | 평균 추론시간 | 최대 메모리 | 평가 |
|-----|---------|-------------|----------|------|
| **gtts** | 0.12 | 0.42초 | 1.67 GB | 🎉 매우 빠름 |
| **melotts** | 0.18 | 0.42초 | 2.52 GB | 🎉 매우 빠름 |

두 모델 모두 RTF < 0.5 목표를 달성했습니다!

## 다음 단계

1. **✅ MeloTTS 구현 완료!**
   - melo-tts 라이브러리 통합 완료
   - 한국어 전용 TTS 활용
   - 벤치마크 테스트 성공

2. **Zonos 모델 구현**
   - Zonos 공식 API 확인
   - transformers 또는 전용 라이브러리 사용
   
3. **CosyVoice2 모델 구현**
   - CosyVoice 라이브러리 설치
   - 공식 문서 참고하여 구현

4. **Kokoro 모델 구현**
   - Kokoro 라이브러리 확인
   - 경량 모델의 장점 활용

5. **전체 벤치마크 및 시각화**
   - 현재까지 구현된 모델 비교 분석 (gtts vs melotts)
   - 성능 그래프 생성
   - 최종 리포트 작성

## 참고 자료

- BaseTTS 추상 클래스: `models/base.py`
- 완전 구현 예시: `models/gtts_wrapper.py`
- 벤치마크 러너: `tests/test_runner.py`
- 성능 모니터: `utils/performance_monitor.py`


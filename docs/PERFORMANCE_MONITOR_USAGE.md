# PerformanceMonitor 사용 가이드

## 개요

`PerformanceMonitor`는 TTS 모델의 성능을 측정하기 위한 컨텍스트 매니저 클래스입니다.

## 기본 사용법

### 1. 간단한 성능 측정

```python
from utils import PerformanceMonitor

# 모니터 생성
monitor = PerformanceMonitor(device='auto')

# 성능 측정
with monitor.measure() as metrics:
    # 측정하고 싶은 코드
    result = my_computation()

# 결과 확인
print(f"추론 시간: {metrics['inference_time']}초")
print(f"메모리 사용량: {metrics['peak_memory_mb']} MB")
print(f"CPU 사용률: {metrics['cpu_percent']}%")
```

### 2. TTS 모델과 함께 사용

```python
from models import GTTSWrapper
from utils import PerformanceMonitor

# TTS 모델 준비
tts = GTTSWrapper()
tts.load_model()

# 성능 모니터
monitor = PerformanceMonitor(device='cpu')

# 음성 합성 + 성능 측정
with monitor.measure() as metrics:
    audio = tts.synthesize("안녕하세요")

# 오디오 길이 계산
audio_duration = len(audio) / tts.sample_rate

# RTF 계산
rtf = PerformanceMonitor.calculate_rtf(
    metrics['inference_time'],
    audio_duration
)

print(f"RTF: {rtf:.4f}")
if rtf < 0.5:
    print("🎉 목표 달성!")
```

## 측정 메트릭

### 기본 메트릭 (항상 제공)

- `inference_time` (float): 추론 시간 (초)
- `peak_memory_mb` (float): 최대 메모리 사용량 (MB)
- `cpu_percent` (float): CPU 사용률 (%)
- `memory_percent` (float): 메모리 사용률 (%)
- `device` (str): 사용된 디바이스 ('cuda' 또는 'cpu')

### GPU 메트릭 (CUDA 사용 가능 시)

- `gpu_memory_mb` (float): GPU 메모리 사용량 (MB)
- `gpu_max_memory_mb` (float): 최대 GPU 메모리 (MB)
- `gpu_memory_allocated` (float): 할당된 GPU 메모리 (MB)
- `gpu_utilization` (float): GPU 사용률 (%) - GPUtil 필요
- `gpu_temperature` (float): GPU 온도 (°C) - GPUtil 필요

## RTF (Real-Time Factor) 계산

RTF는 실시간 성능을 평가하는 지표입니다.

```python
rtf = PerformanceMonitor.calculate_rtf(
    inference_time=1.5,  # 추론 시간 (초)
    audio_duration=3.0   # 오디오 길이 (초)
)
# RTF = 1.5 / 3.0 = 0.5
```

### RTF 해석

- **RTF < 0.5**: 🎉 매우 빠름 (목표 달성!)
- **RTF < 1.0**: ✅ 실시간보다 빠름
- **RTF = 1.0**: 실시간과 동일
- **RTF > 1.0**: ⚠️ 실시간보다 느림

## 시스템 정보 조회

```python
monitor = PerformanceMonitor(device='auto')
info = monitor.get_system_info()

print(f"CPU 코어: {info['cpu_count']}")
print(f"전체 메모리: {info['total_memory_gb']} GB")
print(f"GPU: {info.get('gpu_name', 'N/A')}")
```

## 고급 사용법

### 다중 측정 및 통계

```python
monitor = PerformanceMonitor(device='auto')
results = []

for text in test_sentences:
    with monitor.measure() as metrics:
        audio = tts.synthesize(text)
    results.append(metrics)

# 평균 계산
avg_time = sum(m['inference_time'] for m in results) / len(results)
print(f"평균 추론 시간: {avg_time:.3f}초")
```

### GPU 메모리 최적화

```python
import torch

monitor = PerformanceMonitor(device='cuda')

with monitor.measure() as metrics:
    # 계산
    result = model.forward(input)
    torch.cuda.synchronize()  # GPU 동기화

# GPU 메모리 정리
torch.cuda.empty_cache()
```

## 주의사항

1. **짧은 연산**: 매우 짧은 연산(< 0.1초)은 측정 오차가 클 수 있습니다.
2. **CPU 사용률**: 멀티코어 시스템에서 100%를 초과할 수 있습니다.
3. **GPU 메트릭**: GPUtil이 설치되지 않으면 일부 메트릭이 제공되지 않습니다.
4. **인터넷 API**: gTTS 같은 API 기반 모델은 네트워크 지연이 포함됩니다.

## 예제 출력

```
추론 시간: 0.44초
오디오 길이: 3.89초
RTF: 0.1137
Peak Memory: 808.43 MB
CPU 사용률: 0.00%
평가: 🎉 매우 빠름 (목표 달성!)
```

## 문제 해결

### GPU 메모리 측정이 안 됨

```bash
# PyTorch CUDA 설치 확인
python3 -c "import torch; print(torch.cuda.is_available())"

# GPUtil 설치 (선택사항)
pip install gputil
```

### CPU 사용률이 0%로 나옴

짧은 연산의 경우 정상입니다. 더 긴 시간 동안 측정하세요.


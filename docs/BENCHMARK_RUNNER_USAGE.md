# BenchmarkRunner 사용 가이드

## 개요

`BenchmarkRunner`는 여러 TTS 모델의 성능을 자동으로 측정하고 비교하는 클래스입니다.

## 기본 사용법

### 1. 빠른 시작

```bash
# 벤치마크 실행 (기본 설정: 3회 반복)
python3 scripts/run_benchmark.py

# 반복 횟수 지정
python3 scripts/run_benchmark.py --iterations 5

# 커스텀 설정 파일 사용
python3 scripts/run_benchmark.py --config config/custom_config.yaml
```

### 2. 프로그래밍 방식 사용

```python
from tests import BenchmarkRunner

# 1. BenchmarkRunner 생성
runner = BenchmarkRunner()

# 2. 모델 초기화
models = runner.initialize_models()

# 3. 테스트 문장 로드
sentences = runner.load_test_sentences()

# 4. 벤치마크 실행
runner.run_benchmark(num_iterations=3)

# 5. 결과 저장
runner.save_results()
```

## 상세 기능

### 모델 초기화

```python
runner = BenchmarkRunner()
models = runner.initialize_models()

# 초기화된 모델 확인
print(f"사용 가능한 모델: {list(models.keys())}")

# 특정 모델만 사용
runner.models = {k: v for k, v in models.items() if k in ['gtts', 'kokoro']}
```

**특징:**
- 모델 로드 실패 시 자동으로 건너뜀
- 다른 모델은 정상적으로 계속 진행
- 최소 한 개 이상의 모델이 필요

### 테스트 문장 관리

```python
# 테스트 문장 로드
sentences = runner.load_test_sentences()

# 문장 커스터마이징
runner.test_sentences = [
    "안녕하세요.",
    "테스트 문장입니다.",
]

# 또는 일부만 사용
runner.test_sentences = runner.test_sentences[:5]
```

**테스트 문장 파일:** `tests/test_sentences.txt`
- 한 줄에 한 문장
- `#`으로 시작하는 줄은 주석으로 무시
- 빈 줄도 무시

### 단일 테스트 실행

```python
# 특정 모델로 단일 테스트
result = runner.run_single_test(
    model_name='gtts',
    text='안녕하세요',
    iteration=1
)

print(f"RTF: {result['rtf']:.4f}")
print(f"추론 시간: {result['inference_time']:.3f}초")
print(f"메모리: {result['peak_memory_mb']:.2f} MB")
```

**반환값:**
```python
{
    'model': 'gtts',
    'text': '안녕하세요',
    'text_length': 5,
    'iteration': 1,
    'inference_time': 0.47,        # 추론 시간 (초)
    'rtf': 0.34,                   # Real-Time Factor
    'peak_memory_mb': 803.97,      # 최대 메모리 (MB)
    'cpu_percent': 0.0,            # CPU 사용률 (%)
    'audio_duration': 1.368,       # 오디오 길이 (초)
    'sample_rate': 24000,          # 샘플링 레이트
    'output_path': '...',          # 출력 파일 경로
    'timestamp': '20251027_112834' # 타임스탬프
}
```

### 벤치마크 실행

```python
# 전체 벤치마크 실행
runner.run_benchmark(num_iterations=5)

# 진행 상황은 tqdm으로 자동 표시됨
# 각 테스트 간 0.5초 대기 (API 제한 고려)
```

**실행 순서:**
1. 모든 모델에 대해
2. 모든 테스트 문장에 대해
3. 지정된 반복 횟수만큼
4. PerformanceMonitor로 성능 측정
5. 결과를 내부 리스트에 저장

### 결과 저장

```python
runner.save_results()
```

**생성 파일:**
1. `benchmark_results_YYYYMMDD_HHMMSS.csv` - 전체 결과
2. `summary_statistics_YYYYMMDD_HHMMSS.csv` - 통계 요약

**통계 요약 내용:**
- 평균 추론 시간 (mean, std, min, max)
- 평균 RTF (mean, std, min, max)
- 메모리 사용량 (mean, std, max)
- CPU 사용률 (mean, std)
- GPU 메트릭 (사용 시)

## 결과 분석

### CSV 파일 읽기

```python
import pandas as pd

# 전체 결과
df = pd.read_csv('results/metrics/benchmark_results_*.csv')

# 특정 모델 필터링
gtts_df = df[df['model'] == 'gtts']

# RTF 분석
print(f"평균 RTF: {df['rtf'].mean():.4f}")
print(f"최소 RTF: {df['rtf'].min():.4f}")
print(f"최대 RTF: {df['rtf'].max():.4f}")

# 모델별 비교
model_comparison = df.groupby('model')['rtf'].agg(['mean', 'std', 'min', 'max'])
print(model_comparison)
```

### 결과 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

# RTF 분포 박스플롯
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='model', y='rtf')
plt.axhline(y=0.5, color='r', linestyle='--', label='목표 (RTF < 0.5)')
plt.axhline(y=1.0, color='orange', linestyle='--', label='실시간')
plt.legend()
plt.title('모델별 RTF 분포')
plt.ylabel('RTF (Real-Time Factor)')
plt.xlabel('모델')
plt.tight_layout()
plt.savefig('results/plots/rtf_comparison.png')
```

## 고급 사용법

### 커스텀 진행 상황 모니터링

```python
from tqdm import tqdm

# 자체 진행 표시
for model_name in runner.models.keys():
    for text in tqdm(runner.test_sentences, desc=f"{model_name}"):
        result = runner.run_single_test(model_name, text, iteration=1)
        # 결과 처리
```

### 부분 벤치마크

```python
# 특정 모델만 테스트
runner.models = {'gtts': runner.models['gtts']}

# 짧은 문장만 테스트
runner.test_sentences = [s for s in runner.test_sentences if len(s) < 20]

# 벤치마크 실행
runner.run_benchmark(num_iterations=10)
```

### 에러 처리

```python
try:
    runner.run_benchmark(num_iterations=5)
except RuntimeError as e:
    print(f"벤치마크 실패: {e}")
    # 부분 결과라도 저장
    if runner.results:
        runner.save_results()
```

## 성능 최적화 팁

### 1. API 제한 대응

Google TTS 같은 API 기반 모델은 요청 제한이 있을 수 있습니다.

```python
import time

# 테스트 간 대기 시간 증가
for model_name in runner.models.keys():
    for text in runner.test_sentences:
        result = runner.run_single_test(model_name, text, iteration=1)
        time.sleep(1.0)  # 1초 대기
```

### 2. 메모리 관리

```python
import gc
import torch

# 각 모델 테스트 후 메모리 정리
for model_name in runner.models.keys():
    # 테스트 실행
    # ...
    
    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 3. 병렬 처리 (주의)

성능 측정의 정확도를 위해 병렬 처리는 권장하지 않지만, 
대량 테스트 시 고려할 수 있습니다.

```python
from concurrent.futures import ThreadPoolExecutor

# 단, 성능 메트릭이 부정확할 수 있음
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    for text in runner.test_sentences:
        future = executor.submit(
            runner.run_single_test,
            'gtts',
            text,
            1
        )
        futures.append(future)
    
    results = [f.result() for f in futures]
```

## 문제 해결

### 모델 초기화 실패

```
✗ model_name 초기화 실패: ...
```

**해결 방법:**
1. 모델 래퍼 클래스가 구현되었는지 확인
2. 필요한 라이브러리가 설치되었는지 확인
3. 인터넷 연결 확인 (API 모델의 경우)

### 테스트 문장 파일 없음

```
FileNotFoundError: 테스트 문장 파일을 찾을 수 없습니다
```

**해결 방법:**
```bash
# 파일 확인
ls tests/test_sentences.txt

# 없으면 생성
echo "안녕하세요." > tests/test_sentences.txt
echo "테스트 문장입니다." >> tests/test_sentences.txt
```

### 사용 가능한 모델이 없음

```
RuntimeError: 사용 가능한 모델이 없습니다
```

**해결 방법:**
1. 최소 한 개의 모델 래퍼가 구현되어야 함
2. `initialize_models()` 로그 확인
3. 모델 초기화 오류 해결

## 예제 시나리오

### 시나리오 1: 빠른 테스트

```python
# 한 개 모델, 한 개 문장, 1회 반복
runner = BenchmarkRunner()
runner.initialize_models()
runner.test_sentences = ["안녕하세요."]
runner.run_benchmark(num_iterations=1)
runner.save_results()
```

### 시나리오 2: 전체 벤치마크

```python
# 모든 모델, 모든 문장, 5회 반복
runner = BenchmarkRunner()
runner.initialize_models()
runner.load_test_sentences()
runner.run_benchmark(num_iterations=5)
runner.save_results()
```

### 시나리오 3: 문장 길이별 성능

```python
runner = BenchmarkRunner()
runner.initialize_models()
runner.load_test_sentences()

# 짧은 문장 (< 10자)
short_sentences = [s for s in runner.test_sentences if len(s) < 10]
runner.test_sentences = short_sentences
runner.run_benchmark(num_iterations=5)
runner.save_results()  # benchmark_results_short_*.csv

# 긴 문장 (>= 20자)
long_sentences = [s for s in runner.test_sentences if len(s) >= 20]
runner.test_sentences = long_sentences
runner.results = []  # 결과 초기화
runner.run_benchmark(num_iterations=5)
runner.save_results()  # benchmark_results_long_*.csv
```

## 참고

- 성능 측정의 정확도를 위해 벤치마크 실행 중 다른 작업 최소화
- GPU 사용 모델은 첫 실행 시 워밍업 시간 필요
- API 기반 모델은 네트워크 상태에 따라 결과 변동 가능


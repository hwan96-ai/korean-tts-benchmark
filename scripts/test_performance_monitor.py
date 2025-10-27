"""PerformanceMonitor 클래스 테스트 스크립트.

이 스크립트는 PerformanceMonitor의 기본 사용법과
성능 측정 기능을 테스트합니다.
"""

import sys
import time
from pathlib import Path
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.performance_monitor import PerformanceMonitor
from models.gtts_wrapper import GTTSWrapper


def dummy_computation(duration: float = 1.0) -> np.ndarray:
    """더미 연산을 수행합니다.
    
    Args:
        duration: 연산 시간 (초)
    
    Returns:
        더미 numpy array
    """
    start = time.time()
    result = np.random.rand(1000, 1000)
    
    # 지정된 시간만큼 연산
    while time.time() - start < duration:
        result = np.dot(result, result.T) / 1000
    
    return result


def test_basic_functionality() -> None:
    """기본 기능 테스트."""
    print("=" * 60)
    print("1. PerformanceMonitor 기본 기능 테스트")
    print("=" * 60)
    
    # 인스턴스 생성
    monitor = PerformanceMonitor(device='auto')
    
    # 시스템 정보 출력
    print("\n[시스템 정보]")
    sys_info = monitor.get_system_info()
    for key, value in sys_info.items():
        print(f"  - {key}: {value}")
    
    # 간단한 연산 측정
    print("\n[성능 측정 테스트]")
    print("더미 연산 수행 중 (약 1초)...")
    
    with monitor.measure() as metrics:
        result = dummy_computation(duration=1.0)
    
    print("\n[측정 결과]")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")


def test_rtf_calculation() -> None:
    """RTF 계산 테스트."""
    print("\n" + "=" * 60)
    print("2. RTF 계산 테스트")
    print("=" * 60)
    
    test_cases = [
        (1.0, 2.0, "실시간의 절반 속도 (빠름)"),
        (2.0, 2.0, "실시간과 동일"),
        (3.0, 2.0, "실시간의 1.5배 속도 (느림)"),
        (0.5, 5.0, "실시간의 0.1배 속도 (매우 빠름)"),
    ]
    
    for inference_time, audio_duration, description in test_cases:
        rtf = PerformanceMonitor.calculate_rtf(inference_time, audio_duration)
        print(f"\n  추론시간: {inference_time}초, 오디오: {audio_duration}초")
        print(f"  RTF: {rtf:.4f} - {description}")


def test_with_tts_model() -> None:
    """실제 TTS 모델과 통합 테스트."""
    print("\n" + "=" * 60)
    print("3. TTS 모델 통합 테스트")
    print("=" * 60)
    
    try:
        # TTS 모델 준비
        print("\n[Google TTS 모델 로드]")
        tts = GTTSWrapper()
        tts.load_model()
        
        # 성능 모니터 생성
        monitor = PerformanceMonitor(device='cpu')  # gTTS는 CPU 기반
        
        # 테스트 문장
        test_text = "안녕하세요, 성능 측정 테스트입니다."
        print(f"\n[음성 합성 테스트]")
        print(f"텍스트: \"{test_text}\"")
        
        # 성능 측정하며 음성 합성
        with monitor.measure() as metrics:
            audio = tts.synthesize(test_text)
        
        # 오디오 길이 계산
        audio_duration = len(audio) / tts.sample_rate
        
        # RTF 계산
        rtf = PerformanceMonitor.calculate_rtf(
            metrics['inference_time'],
            audio_duration
        )
        
        print("\n[측정 결과]")
        print(f"  추론 시간: {metrics['inference_time']:.2f}초")
        print(f"  오디오 길이: {audio_duration:.2f}초")
        print(f"  RTF: {rtf:.4f}")
        print(f"  Peak Memory: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU 사용률: {metrics['cpu_percent']:.2f}%")
        
        # RTF 평가
        if rtf < 0.5:
            print(f"  평가: 🎉 매우 빠름 (목표 달성!)")
        elif rtf < 1.0:
            print(f"  평가: ✅ 빠름 (실시간보다 빠름)")
        else:
            print(f"  평가: ⚠️  느림 (실시간보다 느림)")
        
    except Exception as e:
        print(f"\n⚠️  TTS 모델 테스트 실패: {e}")
        print("(인터넷 연결이 필요합니다)")


def test_error_handling() -> None:
    """에러 핸들링 테스트."""
    print("\n" + "=" * 60)
    print("4. 에러 핸들링 테스트")
    print("=" * 60)
    
    # 잘못된 device
    print("\n[1] 잘못된 device 테스트")
    try:
        monitor = PerformanceMonitor(device='invalid')
        print("  ✗ 실패: 예외가 발생하지 않음")
    except ValueError as e:
        print(f"  ✓ 성공: {str(e)[:50]}...")
    
    # RTF 계산 - 0으로 나누기
    print("\n[2] RTF 계산 - 잘못된 오디오 길이")
    try:
        rtf = PerformanceMonitor.calculate_rtf(1.0, 0.0)
        print("  ✗ 실패: 예외가 발생하지 않음")
    except ValueError as e:
        print(f"  ✓ 성공: {str(e)[:50]}...")
    
    # RTF 계산 - 음수
    print("\n[3] RTF 계산 - 음수 추론 시간")
    try:
        rtf = PerformanceMonitor.calculate_rtf(-1.0, 2.0)
        print("  ✗ 실패: 예외가 발생하지 않음")
    except ValueError as e:
        print(f"  ✓ 성공: {str(e)[:50]}...")


def main() -> None:
    """메인 함수."""
    try:
        print("\n" + "=" * 60)
        print("PerformanceMonitor 종합 테스트")
        print("=" * 60)
        
        test_basic_functionality()
        test_rtf_calculation()
        test_error_handling()
        test_with_tts_model()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


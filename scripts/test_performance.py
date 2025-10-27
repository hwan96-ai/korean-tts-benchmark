"""PerformanceMonitor와 BenchmarkRunner 테스트"""

from utils.performance_monitor import PerformanceMonitor
from tests.test_runner import BenchmarkRunner

def test_performance_monitor():
    print("=" * 60)
    print("PerformanceMonitor 테스트")
    print("=" * 60)
    
    monitor = PerformanceMonitor(device='auto')
    
    # 시스템 정보
    sys_info = monitor.get_system_info()
    print("\n시스템 정보:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # RTF 계산 테스트
    rtf = PerformanceMonitor.calculate_rtf(1.5, 3.0)
    print(f"\nRTF 테스트: {rtf:.4f} (1.5초 추론 / 3.0초 오디오)")
    
    print("\n✅ PerformanceMonitor 테스트 완료!\n")

def test_benchmark_runner():
    print("=" * 60)
    print("BenchmarkRunner 테스트")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    
    # 테스트 문장 로드
    sentences = runner.load_test_sentences()
    print(f"\n테스트 문장: {len(sentences)}개")
    print(f"  예시: {sentences[0]}")
    
    # 모델 초기화 (gtts만)
    models = runner.initialize_models()
    print(f"\n초기화된 모델: {list(models.keys())}")
    
    print("\n✅ BenchmarkRunner 테스트 완료!\n")

if __name__ == "__main__":
    test_performance_monitor()
    test_benchmark_runner()
    
    print("=" * 60)
    print("✅ 모든 테스트 통과!")
    print("=" * 60)

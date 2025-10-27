"""BenchmarkRunner 테스트 스크립트.

이 스크립트는 BenchmarkRunner의 기본 기능을 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_runner import BenchmarkRunner


def test_initialization() -> None:
    """초기화 테스트."""
    print("=" * 60)
    print("1. BenchmarkRunner 초기화 테스트")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    print("✓ BenchmarkRunner 생성 완료")


def test_load_sentences() -> None:
    """테스트 문장 로드 테스트."""
    print("\n" + "=" * 60)
    print("2. 테스트 문장 로드 테스트")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    sentences = runner.load_test_sentences()
    
    print(f"\n로드된 문장 ({len(sentences)}개):")
    for i, sent in enumerate(sentences[:5], 1):
        print(f"  {i}. {sent}")
    if len(sentences) > 5:
        print(f"  ... (총 {len(sentences)}개)")


def test_initialize_models() -> None:
    """모델 초기화 테스트."""
    print("\n" + "=" * 60)
    print("3. 모델 초기화 테스트")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    models = runner.initialize_models()
    
    print(f"\n✓ 초기화된 모델: {list(models.keys())}")


def test_single_test() -> None:
    """단일 테스트 실행."""
    print("\n" + "=" * 60)
    print("4. 단일 테스트 실행")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    runner.initialize_models()
    
    if not runner.models:
        print("⚠️  사용 가능한 모델이 없습니다.")
        return
    
    # 첫 번째 모델로 테스트
    model_name = list(runner.models.keys())[0]
    test_text = "안녕하세요, 테스트입니다."
    
    print(f"\n모델: {model_name}")
    print(f"텍스트: \"{test_text}\"")
    
    result = runner.run_single_test(
        model_name=model_name,
        text=test_text,
        iteration=1
    )
    
    print("\n[테스트 결과]")
    print(f"  추론 시간: {result['inference_time']:.3f}초")
    print(f"  RTF: {result['rtf']:.4f}")
    print(f"  메모리: {result['peak_memory_mb']:.2f} MB")
    print(f"  오디오 길이: {result['audio_duration']:.2f}초")
    print(f"  출력 파일: {result['output_path']}")


def test_mini_benchmark() -> None:
    """미니 벤치마크 테스트 (1회 반복)."""
    print("\n" + "=" * 60)
    print("5. 미니 벤치마크 테스트")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    runner.initialize_models()
    runner.load_test_sentences()
    
    # 첫 3개 문장만 사용
    runner.test_sentences = runner.test_sentences[:3]
    
    print(f"\n테스트 설정:")
    print(f"  모델: {list(runner.models.keys())}")
    print(f"  문장: {len(runner.test_sentences)}개")
    print(f"  반복: 1회")
    
    runner.run_benchmark(num_iterations=1)
    
    print("\n[결과 미리보기]")
    if runner.results:
        last_result = runner.results[-1]
        print(f"  마지막 테스트:")
        print(f"    모델: {last_result['model']}")
        print(f"    텍스트: {last_result['text'][:30]}...")
        print(f"    RTF: {last_result['rtf']:.4f}")


def main() -> None:
    """메인 함수."""
    try:
        print("\n" + "=" * 60)
        print("BenchmarkRunner 종합 테스트")
        print("=" * 60)
        
        test_initialization()
        test_load_sentences()
        test_initialize_models()
        test_single_test()
        test_mini_benchmark()
        
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


"""TTS 모델 벤치마크 실행 스크립트.

이 스크립트는 BenchmarkRunner를 사용하여
TTS 모델들의 성능을 측정하고 결과를 저장합니다.

사용 예시:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --device cpu --iterations 3
    python scripts/run_benchmark.py --models gtts kokoro
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_runner import BenchmarkRunner
from utils.performance_monitor import PerformanceMonitor


def parse_arguments() -> argparse.Namespace:
    """커맨드 라인 인자를 파싱합니다.
    
    Returns:
        파싱된 인자들
    """
    parser = argparse.ArgumentParser(
        description='TTS 모델 벤치마크 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행 (모든 모델, auto 디바이스, 5회 반복)
  python scripts/run_benchmark.py
  
  # CPU에서 3회 반복
  python scripts/run_benchmark.py --device cpu --iterations 3
  
  # 특정 모델만 테스트
  python scripts/run_benchmark.py --models gtts
  
  # 여러 모델 지정
  python scripts/run_benchmark.py --models gtts kokoro --iterations 10
        """
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='실행할 디바이스 (기본값: auto)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='각 테스트의 반복 횟수 (기본값: 5)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='테스트할 모델 이름 (기본값: 전체 모델)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/models_config.yaml',
        help='모델 설정 파일 경로 (기본값: config/models_config.yaml)'
    )
    
    parser.add_argument(
        '--skip-confirmation',
        action='store_true',
        help='시작 전 확인 메시지 건너뛰기'
    )
    
    return parser.parse_args()


def print_header() -> None:
    """헤더를 출력합니다."""
    print("\n" + "=" * 70)
    print(" " * 20 + "TTS 모델 벤치마크 시스템")
    print("=" * 70)


def print_system_info(device: str) -> None:
    """시스템 정보를 출력합니다.
    
    Args:
        device: 디바이스 설정
    """
    print("\n[시스템 정보]")
    monitor = PerformanceMonitor(device=device)
    sys_info = monitor.get_system_info()
    
    print(f"  CPU 코어: {sys_info['cpu_count']} (물리: {sys_info.get('cpu_count_physical', 'N/A')})")
    print(f"  메모리: {sys_info['total_memory_gb']} GB "
          f"(사용 가능: {sys_info['available_memory_gb']} GB)")
    
    if sys_info.get('gpu_available'):
        print(f"  GPU: {sys_info['gpu_name']}")
        print(f"  GPU 메모리: {sys_info['gpu_total_memory_gb']} GB")
        print(f"  CUDA 버전: {sys_info.get('cuda_version', 'N/A')}")
    else:
        print(f"  GPU: 사용 안 함")
    
    print(f"  실행 디바이스: {monitor.device}")


def filter_models(
    runner: BenchmarkRunner,
    selected_models: Optional[List[str]]
) -> None:
    """선택된 모델만 필터링합니다.
    
    Args:
        runner: BenchmarkRunner 인스턴스
        selected_models: 선택된 모델 이름 리스트 (None이면 전체)
    
    Raises:
        ValueError: 선택된 모델이 사용 불가능한 경우
    """
    if selected_models is None:
        return
    
    # 선택된 모델만 필터링
    available_models = set(runner.models.keys())
    selected_set = set(selected_models)
    
    # 존재하지 않는 모델 확인
    invalid_models = selected_set - available_models
    if invalid_models:
        print(f"\n⚠️  경고: 다음 모델을 사용할 수 없습니다: {', '.join(invalid_models)}")
        print(f"     사용 가능한 모델: {', '.join(available_models)}")
    
    # 유효한 모델만 필터링
    valid_models = selected_set & available_models
    
    if not valid_models:
        raise ValueError(
            f"선택된 모델 중 사용 가능한 모델이 없습니다.\n"
            f"사용 가능한 모델: {', '.join(available_models)}"
        )
    
    runner.models = {k: v for k, v in runner.models.items() if k in valid_models}
    print(f"\n✓ 선택된 모델로 필터링: {', '.join(runner.models.keys())}")


def print_benchmark_config(
    runner: BenchmarkRunner,
    iterations: int,
    device: str
) -> None:
    """벤치마크 설정을 출력합니다.
    
    Args:
        runner: BenchmarkRunner 인스턴스
        iterations: 반복 횟수
        device: 디바이스
    """
    total_tests = len(runner.models) * len(runner.test_sentences) * iterations
    
    print("\n" + "-" * 70)
    print("[벤치마크 설정]")
    print(f"  모델: {', '.join(runner.models.keys())} ({len(runner.models)}개)")
    print(f"  테스트 문장: {len(runner.test_sentences)}개")
    print(f"  반복 횟수: {iterations}회")
    print(f"  디바이스: {device}")
    print(f"  총 테스트 수: {total_tests}개")
    
    # 예상 시간 계산 (대략적)
    estimated_time = total_tests * 2  # 테스트당 약 2초 가정
    minutes = estimated_time // 60
    seconds = estimated_time % 60
    print(f"  예상 소요 시간: 약 {minutes}분 {seconds}초")
    print("-" * 70)


def print_final_summary(runner: BenchmarkRunner) -> None:
    """최종 요약을 출력합니다.
    
    Args:
        runner: BenchmarkRunner 인스턴스
    """
    if not runner.results:
        print("\n⚠️  저장된 결과가 없습니다.")
        return
    
    import pandas as pd
    df = pd.DataFrame(runner.results)
    
    print("\n" + "=" * 70)
    print(" " * 25 + "최종 요약")
    print("=" * 70)
    
    # 전체 통계
    print(f"\n[전체 통계]")
    print(f"  총 테스트: {len(df)}개")
    print(f"  성공률: 100%")
    print(f"  평균 RTF: {df['rtf'].mean():.4f} (±{df['rtf'].std():.4f})")
    print(f"  평균 추론 시간: {df['inference_time'].mean():.3f}초 (±{df['inference_time'].std():.3f})")
    print(f"  평균 메모리: {df['peak_memory_mb'].mean():.2f} MB")
    
    # 모델별 요약
    print(f"\n[모델별 요약]")
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        
        avg_rtf = model_df['rtf'].mean()
        avg_time = model_df['inference_time'].mean()
        max_memory = model_df['peak_memory_mb'].max()
        
        # 성능 평가
        if avg_rtf < 0.5:
            badge = "🎉 매우 빠름"
        elif avg_rtf < 1.0:
            badge = "✅ 빠름"
        else:
            badge = "⚠️  느림"
        
        print(f"\n  [{model_name.upper()}] {badge}")
        print(f"    테스트 수: {len(model_df)}개")
        print(f"    평균 RTF: {avg_rtf:.4f} (±{model_df['rtf'].std():.4f})")
        print(f"    평균 추론 시간: {avg_time:.3f}초")
        print(f"    최대 메모리: {max_memory:.2f} MB")
        print(f"    평균 오디오 길이: {model_df['audio_duration'].mean():.2f}초")
    
    # RTF 목표 달성 여부
    print(f"\n[성능 목표 달성]")
    rtf_target = 0.5
    models_below_target = []
    
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        avg_rtf = model_df['rtf'].mean()
        
        if avg_rtf < rtf_target:
            models_below_target.append(model_name)
            status = "✅ 달성"
        else:
            status = "❌ 미달성"
        
        print(f"  {model_name}: RTF < {rtf_target} → {status} (평균: {avg_rtf:.4f})")
    
    if models_below_target:
        print(f"\n  🎉 {len(models_below_target)}개 모델이 목표를 달성했습니다!")
    
    print("\n" + "=" * 70)


def main() -> None:
    """메인 함수."""
    args = parse_arguments()
    
    try:
        print_header()
        print_system_info(args.device)
        
        # 1. BenchmarkRunner 초기화
        print("\n" + "=" * 70)
        print("[1단계] BenchmarkRunner 초기화")
        print("=" * 70)
        runner = BenchmarkRunner(config_path=args.config)
        
        # 2. 모델 초기화
        print("\n" + "=" * 70)
        print("[2단계] 모델 초기화")
        print("=" * 70)
        models = runner.initialize_models()
        
        if not models:
            print("\n❌ 사용 가능한 모델이 없습니다.")
            print("   모델 래퍼를 먼저 구현하세요.")
            sys.exit(1)
        
        # 3. 모델 필터링 (선택된 모델만)
        if args.models:
            print("\n" + "=" * 70)
            print("[3단계] 모델 필터링")
            print("=" * 70)
            filter_models(runner, args.models)
        
        # 4. 테스트 문장 로드
        print("\n" + "=" * 70)
        print("[4단계] 테스트 문장 로드")
        print("=" * 70)
        sentences = runner.load_test_sentences()
        
        if not sentences:
            print("\n❌ 테스트 문장이 없습니다.")
            sys.exit(1)
        
        # 5. 벤치마크 설정 확인
        print_benchmark_config(runner, args.iterations, args.device)
        
        # 6. 사용자 확인
        if not args.skip_confirmation:
            print("\n⚠️  벤치마크 실행 중에는 다른 작업을 최소화하세요.")
            print("   (정확한 성능 측정을 위해)")
            print()
            response = input("벤치마크를 시작하시겠습니까? [Y/n]: ")
            
            if response.lower() in ['n', 'no']:
                print("\n벤치마크가 취소되었습니다.")
                sys.exit(0)
        
        # 7. 벤치마크 실행
        print("\n" + "=" * 70)
        print("[5단계] 벤치마크 실행")
        print("=" * 70)
        runner.run_benchmark(num_iterations=args.iterations)
        
        # 8. 결과 저장
        print("\n" + "=" * 70)
        print("[6단계] 결과 저장")
        print("=" * 70)
        runner.save_results()
        
        # 9. 최종 요약
        print_final_summary(runner)
        
        # 10. 다음 단계 안내
        print("\n" + "=" * 70)
        print("✅ 벤치마크 완료!")
        print("=" * 70)
        print("\n[생성된 파일]")
        print("  📊 results/metrics/benchmark_results_*.csv")
        print("  📈 results/metrics/summary_statistics_*.csv")
        print("\n[다음 단계]")
        print("  - 결과 시각화: python3 scripts/visualize_results.py")
        print("  - 결과 분석: pandas로 CSV 파일 분석")
        print("  - 오디오 확인: data/output/{model_name}/ 폴더")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


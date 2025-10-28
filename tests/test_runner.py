"""TTS 모델 벤치마크 러너 모듈."""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

from utils.performance_monitor import PerformanceMonitor


class BenchmarkRunner:
    """TTS 모델들의 성능을 벤치마크하는 클래스.
    
    여러 TTS 모델에 대해 동일한 테스트 문장들을 실행하고
    추론 속도, 메모리 사용량 등의 성능 지표를 측정합니다.
    
    Attributes:
        config_path: 설정 파일 경로
        models_config: 모델 설정 정보
        models: 초기화된 모델 딕셔너리
        test_sentences: 테스트 문장 리스트
        results: 벤치마크 결과 리스트
        performance_monitor: 성능 측정 모니터
    """
    
    def __init__(self, config_path: str = 'config/models_config.yaml') -> None:
        """BenchmarkRunner 초기화.
        
        Args:
            config_path: 모델 설정 파일 경로 (프로젝트 루트 기준)
        
        Raises:
            FileNotFoundError: 설정 파일을 찾을 수 없는 경우
        """
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / config_path
        
        # 설정 파일 로드
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.models_config = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {self.config_path}"
            ) from e
        
        # 초기화
        self.models: Dict[str, Any] = {}
        self.test_sentences: List[str] = []
        self.results: List[Dict[str, Any]] = []
        self.performance_monitor = PerformanceMonitor(device='auto')
        
        print(f"✓ BenchmarkRunner 초기화 완료")
        print(f"  설정 파일: {self.config_path}")
    
    def initialize_models(self) -> Dict[str, Any]:
        """모델들을 초기화합니다.
        
        config에 정의된 모든 모델을 로드를 시도합니다.
        로드 실패한 모델은 건너뛰고 다른 모델은 계속 진행합니다.
        
        Returns:
            초기화에 성공한 모델들의 딕셔너리
            {model_name: model_instance}
        
        Raises:
            RuntimeError: 사용 가능한 모델이 하나도 없는 경우
        """
        print("\n" + "=" * 60)
        print("모델 초기화 시작")
        print("=" * 60)
        
        # config에 정의된 모든 모델 (벤치마크에 사용할 모델만)
        available_models = ['gtts', 'melotts', 'cosyvoice', 'coqui']
        
        for model_name in available_models:
            if model_name not in self.models_config:
                print(f"\n⚠️  {model_name}: 설정 없음, 건너뜀")
                continue
            
            try:
                print(f"\n[{model_name.upper()}] 초기화 중...")
                
                if model_name == 'gtts':
                    from models.gtts_wrapper import GTTSWrapper
                    model = GTTSWrapper()
                    model.load_model()
                    self.models[model_name] = model
                    print(f"✓ {model_name} 초기화 성공")
                
                elif model_name == 'zonos':
                    try:
                        from models.zonos import ZonosTTS
                        model = ZonosTTS(device='auto')
                        model.load_model()
                        self.models[model_name] = model
                        print(f"✓ {model_name} 초기화 성공")
                    except (NotImplementedError, ImportError) as e:
                        print(f"⚠️  {model_name}: {str(e)[:100]}...")
                        print(f"  → 이 모델은 건너뜁니다.")
                        continue
                
                elif model_name == 'cosyvoice':
                    try:
                        from models.cosyvoice import CosyVoiceTTS
                        model = CosyVoiceTTS(device='auto')
                        model.load_model()
                        self.models[model_name] = model
                        print(f"✓ {model_name} 초기화 성공")
                    except (NotImplementedError, ImportError) as e:
                        print(f"⚠️  {model_name}: {str(e)[:100]}...")
                        print(f"  → 이 모델은 건너뜁니다.")
                        continue
                
                elif model_name == 'kokoro':
                    try:
                        from models.kokoro import KokoroTTS
                        model = KokoroTTS(device='auto')
                        model.load_model()
                        self.models[model_name] = model
                        print(f"✓ {model_name} 초기화 성공")
                    except (NotImplementedError, ImportError) as e:
                        print(f"⚠️  {model_name}: {str(e)[:100]}...")
                        print(f"  → 이 모델은 건너뜁니다.")
                        continue
                
                elif model_name == 'melotts':
                    try:
                        from models.melotts import MeloTTSKorean
                        model = MeloTTSKorean(device='auto')
                        model.load_model()
                        self.models[model_name] = model
                        print(f"✓ {model_name} 초기화 성공")
                    except (NotImplementedError, ImportError) as e:
                        print(f"⚠️  {model_name}: {str(e)[:100]}...")
                        print(f"  → 이 모델은 건너뜁니다.")
                        continue
                
                elif model_name == 'coqui':
                    try:
                        from models.coqui_tts import CoquiTTS
                        model = CoquiTTS(device='auto')
                        model.load_model()
                        self.models[model_name] = model
                        print(f"✓ {model_name} 초기화 성공")
                    except (NotImplementedError, ImportError) as e:
                        print(f"⚠️  {model_name}: {str(e)[:100]}...")
                        print(f"  → 이 모델은 건너뜁니다.")
                        continue
                
            except Exception as e:
                print(f"✗ {model_name} 초기화 실패: {e}")
                print(f"  → 이 모델은 건너뜁니다.")
                continue
        
        # 사용 가능한 모델 확인
        if not self.models:
            raise RuntimeError(
                "사용 가능한 모델이 없습니다. "
                "최소 한 개 이상의 모델이 초기화되어야 합니다."
            )
        
        print("\n" + "=" * 60)
        print(f"✓ 총 {len(self.models)}개 모델 초기화 완료")
        print(f"  활성 모델: {', '.join(self.models.keys())}")
        print("=" * 60)
        
        return self.models
    
    def load_test_sentences(self) -> List[str]:
        """테스트 문장들을 로드합니다.
        
        Returns:
            테스트 문장 리스트
        
        Raises:
            FileNotFoundError: 테스트 문장 파일을 찾을 수 없는 경우
        """
        sentences_path = self.project_root / "tests" / "test_sentences.txt"
        
        try:
            with open(sentences_path, 'r', encoding='utf-8') as f:
                sentences = [
                    line.strip() 
                    for line in f 
                    if line.strip() and not line.startswith('#')
                ]
            
            self.test_sentences = sentences
            print(f"\n✓ 테스트 문장 로드 완료: {len(sentences)}개")
            
            return sentences
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"테스트 문장 파일을 찾을 수 없습니다: {sentences_path}"
            ) from e
    
    def run_single_test(
        self,
        model_name: str,
        text: str,
        iteration: int
    ) -> Dict[str, Any]:
        """단일 테스트를 실행합니다.
        
        지정된 모델로 텍스트를 음성으로 합성하고
        성능 지표를 측정합니다.
        
        Args:
            model_name: 모델 이름
            text: 합성할 텍스트
            iteration: 반복 횟수 (파일명에 사용)
        
        Returns:
            테스트 결과 딕셔너리
                - model: 모델 이름
                - text: 입력 텍스트
                - iteration: 반복 번호
                - inference_time: 추론 시간 (초)
                - rtf: Real-Time Factor
                - peak_memory_mb: 최대 메모리 사용량 (MB)
                - cpu_percent: CPU 사용률 (%)
                - gpu_memory_mb: GPU 메모리 (MB, 사용 시)
                - audio_duration: 오디오 길이 (초)
                - sample_rate: 샘플링 레이트
                - output_path: 출력 파일 경로
        
        Raises:
            ValueError: 모델이 초기화되지 않은 경우
            RuntimeError: 테스트 실행 실패
        """
        if model_name not in self.models:
            raise ValueError(
                f"모델 '{model_name}'이 초기화되지 않았습니다. "
                f"사용 가능한 모델: {list(self.models.keys())}"
            )
        
        model = self.models[model_name]
        
        try:
            # 출력 디렉토리 생성
            output_dir = self.project_root / "data" / "output" / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 성능 측정하며 음성 합성
            with self.performance_monitor.measure() as metrics:
                audio = model.synthesize(text)
            
            # 오디오 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"iter_{iteration}_{timestamp}.wav"
            output_path = output_dir / output_filename
            
            model.save_audio(
                audio,
                str(output_path),
                sample_rate=model.sample_rate
            )
            
            # librosa로 오디오 길이 측정 (검증용)
            try:
                y, sr = librosa.load(str(output_path), sr=None)
                audio_duration = librosa.get_duration(y=y, sr=sr)
            except Exception as e:
                # librosa 실패 시 numpy로 계산
                print(f"  경고: librosa 측정 실패, numpy로 계산: {e}")
                audio_duration = len(audio) / model.sample_rate
            
            # RTF 계산
            rtf = PerformanceMonitor.calculate_rtf(
                metrics['inference_time'],
                audio_duration
            )
            
            # 결과 딕셔너리 생성
            result = {
                'model': model_name,
                'text': text,
                'text_length': len(text),
                'iteration': iteration,
                'inference_time': metrics['inference_time'],
                'rtf': rtf,
                'peak_memory_mb': metrics['peak_memory_mb'],
                'cpu_percent': metrics['cpu_percent'],
                'memory_percent': metrics.get('memory_percent', 0.0),
                'audio_duration': round(audio_duration, 3),
                'sample_rate': model.sample_rate,
                'output_path': str(output_path),
                'timestamp': timestamp,
            }
            
            # GPU 메트릭 추가 (있는 경우)
            if 'gpu_memory_mb' in metrics:
                result['gpu_memory_mb'] = metrics['gpu_memory_mb']
            if 'gpu_max_memory_mb' in metrics:
                result['gpu_max_memory_mb'] = metrics['gpu_max_memory_mb']
            if 'gpu_utilization' in metrics:
                result['gpu_utilization'] = metrics['gpu_utilization']
            
            return result
            
        except Exception as e:
            raise RuntimeError(
                f"테스트 실행 실패 (model={model_name}, text='{text[:30]}...'): {e}"
            ) from e
    
    def run_benchmark(self, num_iterations: int = 5) -> None:
        """전체 벤치마크를 실행합니다.
        
        모든 모델에 대해 모든 테스트 문장을 지정된 횟수만큼 반복 실행합니다.
        
        Args:
            num_iterations: 각 테스트의 반복 횟수 (기본값: 5)
        
        Raises:
            RuntimeError: 모델이나 테스트 문장이 로드되지 않은 경우
        """
        if not self.models:
            raise RuntimeError(
                "모델이 초기화되지 않았습니다. "
                "initialize_models()를 먼저 호출하세요."
            )
        
        if not self.test_sentences:
            raise RuntimeError(
                "테스트 문장이 로드되지 않았습니다. "
                "load_test_sentences()를 먼저 호출하세요."
            )
        
        print("\n" + "=" * 60)
        print("벤치마크 실행 시작")
        print("=" * 60)
        print(f"  모델 수: {len(self.models)}")
        print(f"  테스트 문장: {len(self.test_sentences)}개")
        print(f"  반복 횟수: {num_iterations}")
        print(f"  총 테스트: {len(self.models) * len(self.test_sentences) * num_iterations}개")
        print("=" * 60)
        
        # 전체 진행 상황 추적
        total_tests = len(self.models) * len(self.test_sentences) * num_iterations
        
        with tqdm(total=total_tests, desc="벤치마크 진행") as pbar:
            for model_name in self.models.keys():
                print(f"\n\n[모델: {model_name.upper()}]")
                
                for sentence_idx, text in enumerate(self.test_sentences, 1):
                    print(f"\n  문장 {sentence_idx}/{len(self.test_sentences)}: \"{text[:40]}{'...' if len(text) > 40 else ''}\"")
                    
                    for iteration in range(1, num_iterations + 1):
                        try:
                            # 단일 테스트 실행
                            result = self.run_single_test(
                                model_name=model_name,
                                text=text,
                                iteration=iteration
                            )
                            
                            # 결과 저장
                            self.results.append(result)
                            
                            # 진행 상황 업데이트
                            pbar.set_postfix({
                                'model': model_name,
                                'rtf': f"{result['rtf']:.3f}",
                                'time': f"{result['inference_time']:.2f}s"
                            })
                            pbar.update(1)
                            
                        except Exception as e:
                            print(f"\n    ✗ 테스트 실패 (반복 {iteration}): {e}")
                            pbar.update(1)
                            continue
                        
                        # 모델 간 간격 (API 제한 고려)
                        time.sleep(0.5)
        
        print("\n" + "=" * 60)
        print(f"✓ 벤치마크 완료!")
        print(f"  성공한 테스트: {len(self.results)}개")
        print("=" * 60)
    
    def save_results(self) -> None:
        """벤치마크 결과를 저장합니다.
        
        결과를 CSV 파일과 통계 요약 파일로 저장합니다.
        
        Raises:
            RuntimeError: 저장할 결과가 없는 경우
        """
        if not self.results:
            raise RuntimeError(
                "저장할 결과가 없습니다. run_benchmark()를 먼저 실행하세요."
            )
        
        print("\n" + "=" * 60)
        print("결과 저장 중...")
        print("=" * 60)
        
        # DataFrame 생성
        df = pd.DataFrame(self.results)
        
        # 결과 디렉토리 생성
        metrics_dir = self.project_root / "results" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 전체 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = metrics_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 전체 결과 저장: {results_path}")
        
        # 2. 통계 요약 생성
        print("\n통계 요약 생성 중...")
        
        # 모델별 통계
        summary_data = []
        
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            
            summary = {
                'model': model_name,
                'test_count': len(model_df),
                
                # 추론 시간
                'inference_time_mean': model_df['inference_time'].mean(),
                'inference_time_std': model_df['inference_time'].std(),
                'inference_time_min': model_df['inference_time'].min(),
                'inference_time_max': model_df['inference_time'].max(),
                
                # RTF
                'rtf_mean': model_df['rtf'].mean(),
                'rtf_std': model_df['rtf'].std(),
                'rtf_min': model_df['rtf'].min(),
                'rtf_max': model_df['rtf'].max(),
                
                # 메모리
                'peak_memory_mean': model_df['peak_memory_mb'].mean(),
                'peak_memory_std': model_df['peak_memory_mb'].std(),
                'peak_memory_max': model_df['peak_memory_mb'].max(),
                
                # CPU
                'cpu_percent_mean': model_df['cpu_percent'].mean(),
                'cpu_percent_std': model_df['cpu_percent'].std(),
                
                # 오디오
                'audio_duration_mean': model_df['audio_duration'].mean(),
                'sample_rate': model_df['sample_rate'].iloc[0],
            }
            
            # GPU 메트릭 (있는 경우)
            if 'gpu_memory_mb' in model_df.columns:
                gpu_data = model_df['gpu_memory_mb'].dropna()
                if len(gpu_data) > 0:
                    summary['gpu_memory_mean'] = gpu_data.mean()
                    summary['gpu_memory_max'] = gpu_data.max()
            
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = metrics_dir / f"summary_statistics_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✓ 통계 요약 저장: {summary_path}")
        
        # 3. 콘솔에 요약 출력
        print("\n" + "=" * 60)
        print("벤치마크 요약")
        print("=" * 60)
        
        for _, row in summary_df.iterrows():
            print(f"\n[{row['model'].upper()}]")
            print(f"  테스트 수: {row['test_count']}")
            print(f"  평균 추론 시간: {row['inference_time_mean']:.3f}s (±{row['inference_time_std']:.3f})")
            print(f"  평균 RTF: {row['rtf_mean']:.4f} (±{row['rtf_std']:.4f})")
            print(f"  최대 메모리: {row['peak_memory_max']:.2f} MB")
            
            if row['rtf_mean'] < 0.5:
                print(f"  평가: 🎉 매우 빠름 (목표 달성!)")
            elif row['rtf_mean'] < 1.0:
                print(f"  평가: ✅ 빠름 (실시간보다 빠름)")
            else:
                print(f"  평가: ⚠️  느림 (실시간보다 느림)")
        
        print("\n" + "=" * 60)
        print("✅ 모든 결과 저장 완료!")
        print("=" * 60)


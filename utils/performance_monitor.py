"""TTS 모델 성능 측정을 위한 PerformanceMonitor 클래스."""

import time
import gc
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
import psutil
import torch


class PerformanceMonitor:
    """TTS 모델의 성능을 측정하는 컨텍스트 매니저 클래스.
    
    CPU/GPU 사용률, 메모리 사용량, 추론 시간 등을 측정하여
    TTS 모델의 성능을 정량적으로 평가합니다.
    
    Attributes:
        device: 모델이 실행되는 디바이스 ('cuda' 또는 'cpu')
        process: 현재 프로세스의 psutil Process 객체
        gpu_available: GPU 사용 가능 여부
    
    Example:
        >>> monitor = PerformanceMonitor(device='auto')
        >>> with monitor.measure() as metrics:
        ...     audio = tts_model.synthesize("테스트 문장")
        >>> print(metrics)
        {'inference_time': 1.23, 'peak_memory_mb': 512.5, ...}
    """
    
    def __init__(self, device: str = 'auto') -> None:
        """PerformanceMonitor 초기화.
        
        Args:
            device: 모니터링할 디바이스
                   'auto': CUDA 사용 가능시 자동으로 선택
                   'cuda': GPU 모니터링
                   'cpu': CPU 모니터링
        
        Raises:
            ValueError: 잘못된 device 값
        """
        # device 설정
        if device not in ['auto', 'cuda', 'cpu']:
            raise ValueError(
                f"잘못된 device 값: {device}. "
                "'auto', 'cuda', 'cpu' 중 하나를 선택하세요."
            )
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # GPU 사용 가능 여부 확인
        self.gpu_available = torch.cuda.is_available() and self.device == 'cuda'
        
        # 현재 프로세스
        self.process = psutil.Process()
        
        print(f"✓ PerformanceMonitor 초기화 (device: {self.device})")
    
    @contextmanager
    def measure(self) -> Generator[Dict[str, Any], None, None]:
        """성능 측정 컨텍스트 매니저.
        
        TTS 추론 전후의 성능 지표를 측정하여 딕셔너리로 반환합니다.
        
        Yields:
            성능 메트릭을 담을 딕셔너리 (컨텍스트 종료 시 값이 채워짐)
        
        Returns:
            Dict[str, Any]: 성능 측정 결과
                - inference_time (float): 추론 시간 (초)
                - peak_memory_mb (float): 최대 메모리 사용량 (MB)
                - cpu_percent (float): 평균 CPU 사용률 (%)
                - memory_percent (float): 평균 메모리 사용률 (%)
                - gpu_memory_mb (float, optional): GPU 메모리 사용량 (MB)
                - gpu_utilization (float, optional): GPU 사용률 (%)
                - device (str): 사용된 디바이스
        
        Example:
            >>> with monitor.measure() as metrics:
            ...     result = heavy_computation()
            >>> print(f"Inference time: {metrics['inference_time']:.2f}s")
        """
        # 결과를 저장할 딕셔너리
        metrics: Dict[str, Any] = {
            'device': self.device,
            'inference_time': 0.0,
            'peak_memory_mb': 0.0,
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
        }
        
        try:
            # 가비지 컬렉션 실행 (깨끗한 측정을 위해)
            gc.collect()
            if self.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 초기 상태 기록
            start_time = time.time()
            start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            start_cpu_times = self.process.cpu_times()
            
            # GPU 초기 상태
            gpu_start_memory = 0.0
            if self.gpu_available:
                try:
                    gpu_start_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                except Exception as e:
                    print(f"경고: GPU 메모리 측정 실패: {e}")
            
            # CPU 사용률 측정 시작 (non-blocking)
            cpu_percent_start = self.process.cpu_percent()
            
            # yield하여 사용자 코드 실행
            yield metrics
            
            # GPU 동기화 (정확한 시간 측정을 위해)
            if self.gpu_available:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            
            # 종료 시간 기록
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 종료 시점 메모리 측정
            end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            peak_memory = max(start_memory, end_memory)
            
            # CPU 사용률 계산
            try:
                cpu_percent = self.process.cpu_percent(interval=0.1)
                # 0이면 짧은 간격으로 재측정
                if cpu_percent == 0.0:
                    cpu_percent = self.process.cpu_percent(interval=0.05)
            except Exception as e:
                print(f"경고: CPU 사용률 측정 실패: {e}")
                cpu_percent = 0.0
            
            # 메모리 사용률
            try:
                memory_percent = self.process.memory_percent()
            except Exception as e:
                print(f"경고: 메모리 사용률 측정 실패: {e}")
                memory_percent = 0.0
            
            # 기본 메트릭 업데이트
            metrics.update({
                'inference_time': round(inference_time, 4),
                'peak_memory_mb': round(peak_memory, 2),
                'cpu_percent': round(cpu_percent, 2),
                'memory_percent': round(memory_percent, 2),
            })
            
            # GPU 메트릭 (사용 가능한 경우)
            if self.gpu_available:
                try:
                    gpu_end_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    gpu_max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                    
                    metrics.update({
                        'gpu_memory_mb': round(gpu_end_memory, 2),
                        'gpu_max_memory_mb': round(gpu_max_memory, 2),
                        'gpu_memory_allocated': round(gpu_end_memory - gpu_start_memory, 2),
                    })
                    
                    # GPU 사용률 (torch로는 직접 측정 불가, GPUtil 사용)
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # 첫 번째 GPU
                            metrics.update({
                                'gpu_utilization': round(gpu.load * 100, 2),
                                'gpu_temperature': round(gpu.temperature, 1),
                            })
                    except ImportError:
                        # GPUtil이 설치되지 않은 경우
                        pass
                    except Exception as e:
                        print(f"경고: GPUtil 측정 실패: {e}")
                
                except Exception as e:
                    print(f"경고: GPU 메트릭 수집 실패: {e}")
            
        except Exception as e:
            # 측정 중 오류 발생 시에도 기본 메트릭 반환
            print(f"경고: 성능 측정 중 오류 발생: {e}")
            metrics['error'] = str(e)
        
        finally:
            # 정리 작업
            if self.gpu_available:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    
    @staticmethod
    def calculate_rtf(inference_time: float, audio_duration: float) -> float:
        """Real-Time Factor (RTF)를 계산합니다.
        
        RTF는 추론 시간과 오디오 길이의 비율로,
        실시간 성능을 평가하는 지표입니다.
        
        RTF < 1.0: 실시간보다 빠름 (좋음)
        RTF = 1.0: 실시간과 동일
        RTF > 1.0: 실시간보다 느림 (나쁨)
        
        Args:
            inference_time: 추론에 걸린 시간 (초)
            audio_duration: 생성된 오디오의 길이 (초)
        
        Returns:
            RTF 값 (추론시간 / 오디오길이)
        
        Raises:
            ValueError: audio_duration이 0 이하인 경우
            TypeError: 입력값이 숫자가 아닌 경우
        
        Example:
            >>> rtf = PerformanceMonitor.calculate_rtf(1.5, 3.0)
            >>> print(f"RTF: {rtf:.2f}")  # 0.50 (실시간의 절반)
        """
        try:
            # 입력 검증
            inference_time = float(inference_time)
            audio_duration = float(audio_duration)
            
            if audio_duration <= 0:
                raise ValueError(
                    f"오디오 길이는 0보다 커야 합니다. 입력값: {audio_duration}"
                )
            
            if inference_time < 0:
                raise ValueError(
                    f"추론 시간은 0 이상이어야 합니다. 입력값: {inference_time}"
                )
            
            rtf = inference_time / audio_duration
            return round(rtf, 4)
            
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"RTF 계산 실패: {e}. "
                f"inference_time={inference_time}, audio_duration={audio_duration}"
            ) from e
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보를 반환합니다.
        
        Returns:
            시스템 정보를 담은 딕셔너리
                - cpu_count: CPU 코어 수
                - total_memory_gb: 전체 메모리 (GB)
                - available_memory_gb: 사용 가능한 메모리 (GB)
                - gpu_available: GPU 사용 가능 여부
                - gpu_name: GPU 이름 (사용 가능한 경우)
                - gpu_total_memory_gb: GPU 전체 메모리 (GB, 사용 가능한 경우)
        """
        info: Dict[str, Any] = {
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
        }
        
        # 메모리 정보
        try:
            mem = psutil.virtual_memory()
            info.update({
                'total_memory_gb': round(mem.total / (1024**3), 2),
                'available_memory_gb': round(mem.available / (1024**3), 2),
            })
        except Exception as e:
            print(f"경고: 메모리 정보 수집 실패: {e}")
        
        # GPU 정보
        info['gpu_available'] = self.gpu_available
        if self.gpu_available:
            try:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                info['gpu_total_memory_gb'] = round(total_memory / (1024**3), 2)
                info['cuda_version'] = torch.version.cuda
            except Exception as e:
                print(f"경고: GPU 정보 수집 실패: {e}")
        
        return info


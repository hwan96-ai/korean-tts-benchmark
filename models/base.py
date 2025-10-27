"""TTS 모델을 위한 기본 추상 클래스 모듈."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import soundfile as sf


class BaseTTS(ABC):
    """모든 TTS 모델이 상속해야 하는 추상 기본 클래스.
    
    이 클래스는 TTS 모델의 공통 인터페이스를 정의합니다.
    모든 TTS 모델 구현체는 이 클래스를 상속하고 추상 메서드를 구현해야 합니다.
    
    Attributes:
        model_name: 모델의 이름
        model: 로드된 모델 인스턴스
    """
    
    def __init__(self, model_name: str) -> None:
        """BaseTTS 초기화.
        
        Args:
            model_name: 모델의 이름
        """
        self.model_name = model_name
        self.model: Any = None
    
    @abstractmethod
    def load_model(self) -> None:
        """모델을 메모리에 로드합니다.
        
        각 TTS 모델 구현체에서 이 메서드를 구현하여
        모델을 로드하고 초기화해야 합니다.
        
        Raises:
            Exception: 모델 로드 실패 시
        """
        pass
    
    @abstractmethod
    def synthesize(self, text: str, **kwargs: Any) -> np.ndarray:
        """텍스트를 음성으로 변환합니다.
        
        Args:
            text: 합성할 텍스트
            **kwargs: 모델별 추가 파라미터 (예: speaker_id, speed 등)
        
        Returns:
            합성된 오디오 데이터 (numpy array 형태)
        
        Raises:
            Exception: 음성 합성 실패 시
        """
        pass
    
    def save_audio(self, audio: np.ndarray, path: str, sample_rate: int = 22050) -> None:
        """오디오 데이터를 파일로 저장합니다.
        
        Args:
            audio: 저장할 오디오 데이터 (numpy array)
            path: 저장할 파일 경로 (예: 'output.wav')
            sample_rate: 오디오 샘플링 레이트 (기본값: 22050 Hz)
        
        Raises:
            IOError: 파일 저장 실패 시
            ValueError: 잘못된 오디오 데이터 형식
        """
        try:
            if audio is None or len(audio) == 0:
                raise ValueError("오디오 데이터가 비어있습니다.")
            
            # 오디오 데이터 정규화 (-1.0 ~ 1.0 범위로)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 값 범위 확인 및 클리핑
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            sf.write(path, audio, sample_rate)
            
        except Exception as e:
            raise IOError(f"오디오 파일 저장 실패: {path}, 에러: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델의 정보를 반환합니다.
        
        Returns:
            모델 정보를 담은 딕셔너리
            - name: 모델 이름
            - loaded: 모델 로드 여부
            - type: 모델 타입 (클래스명)
        """
        return {
            "name": self.model_name,
            "loaded": self.model is not None,
            "type": self.__class__.__name__
        }


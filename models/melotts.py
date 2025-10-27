"""MeloTTS Korean 모델 래퍼 모듈."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
import yaml
import numpy as np
import librosa
import torch

from models.base import BaseTTS


class MeloTTSKorean(BaseTTS):
    """MeloTTS-Korean TTS 모델 래퍼 클래스.
    
    MyShell의 MeloTTS-Korean 모델을 사용하여
    한국어 텍스트를 음성으로 변환합니다.
    
    MIT 라이선스의 한국어 전용 TTS 모델입니다.
    
    Attributes:
        config: 모델 설정 정보
        sample_rate: 오디오 샘플링 레이트
        output_dir: 생성된 오디오 저장 디렉토리
    """
    
    def __init__(self, device: str = 'auto') -> None:
        """MeloTTSKorean 초기화.
        
        Args:
            device: 모델을 실행할 디바이스 ('auto', 'cuda', 'cpu')
        
        Raises:
            FileNotFoundError: config 파일을 찾을 수 없는 경우
            ValueError: 잘못된 device 값 또는 config 파싱 실패
        """
        super().__init__(model_name="MeloTTS-Korean")
        
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
        
        # config 파일 로드
        try:
            config_path = Path(__file__).parent.parent / "config" / "models_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                all_config = yaml.safe_load(f)
                self.config: Dict[str, Any] = all_config.get('melotts', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 오류: {e}") from e
        
        # config에서 설정 읽기
        self.model_path: str = self.config.get('model_path', 'myshell-ai/MeloTTS-Korean')
        self.sample_rate: int = self.config.get('sample_rate', 44100)
        self.language: str = self.config.get('language', 'KR')
        
        # 고급 설정
        melo_config = self.config.get('config', {})
        self.speed: float = melo_config.get('speed', 1.0)
        self.speaker_id: int = melo_config.get('speaker_id', 0)
        
        # 출력 디렉토리 설정
        output_dir_path = self.config.get('output_dir', 'data/output/melotts')
        self.output_dir = Path(__file__).parent.parent / output_dir_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ MeloTTSKorean 초기화 완료 (language: {self.language}, device: {self.device})")
    
    def load_model(self) -> None:
        """MeloTTS 모델을 로드합니다.
        
        melo-tts 라이브러리를 사용하여 한국어 TTS 모델을 로드합니다.
        
        Raises:
            ImportError: melo-tts 라이브러리가 설치되지 않은 경우
            RuntimeError: 모델 로드 실패
        """
        try:
            print(f"MeloTTS 모델 로드 중... (language: {self.language}, device: {self.device})")
            
            # melo-tts 라이브러리 import
            try:
                from melo.api import TTS
            except ImportError as e:
                raise ImportError(
                    "melo-tts 라이브러리가 설치되지 않았습니다.\n"
                    "설치: pip install melo-tts\n"
                    "자세한 정보: https://github.com/myshell-ai/MeloTTS"
                ) from e
            
            # MeloTTS 모델 로드
            self.model = TTS(language=self.language, device=self.device)
            
            print(f"✓ MeloTTS 모델 로드 완료 (device: {self.device})")
            
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"MeloTTS 모델 로드 실패: {e}"
            ) from e
    
    def synthesize(
        self,
        text: str,
        **kwargs: Any
    ) -> np.ndarray:
        """텍스트를 음성으로 변환합니다.
        
        Args:
            text: 합성할 한국어 텍스트
            **kwargs: 추가 모델 파라미터 (speaker_id, speed 등)
        
        Returns:
            합성된 오디오 데이터 (numpy array, shape: [samples])
        
        Raises:
            RuntimeError: 모델이 로드되지 않았거나 합성 실패
            ValueError: 잘못된 입력 파라미터
        """
        # 모델 로드 확인
        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요."
            )
        
        # 입력 검증
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 합성할 수 없습니다.")
        
        try:
            print(f"텍스트 합성 중: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            # speaker_id와 speed 설정
            speaker_id = kwargs.get('speaker_id', self.speaker_id)
            speed = kwargs.get('speed', self.speed)
            
            # 임시 파일 생성 (MeloTTS는 파일로 저장)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # MeloTTS로 음성 합성 (파일로 저장)
                self.model.tts_to_file(
                    text=text,
                    speaker_id=speaker_id,
                    output_path=tmp_path,
                    speed=speed
                )
                
                # librosa로 오디오 파일 로드
                audio, sr = librosa.load(tmp_path, sr=self.sample_rate)
                
                # float32로 변환 (librosa는 이미 float32로 반환하지만 명시적으로)
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                
                print(f"✓ 음성 합성 완료 (길이: {len(audio)} samples, "
                      f"시간: {len(audio)/self.sample_rate:.2f}초)")
                
                return audio
                
            finally:
                # 임시 파일 삭제
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception as e:
                    print(f"경고: 임시 파일 삭제 실패: {e}")
            
        except ValueError as e:
            raise ValueError(f"입력 파라미터 오류: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"음성 합성 중 오류 발생: {e}. "
                f"입력 텍스트: \"{text[:100]}\""
            ) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """MeloTTS 모델의 상세 정보를 반환합니다.
        
        Returns:
            모델 정보를 담은 딕셔너리
        """
        base_info = super().get_model_info()
        base_info.update({
            "device": self.device,
            "sample_rate": self.sample_rate,
            "model_path": self.model_path,
            "language": self.language,
            "output_dir": str(self.output_dir),
            "model_type": "pip",
            "speed": self.speed,
            "speaker_id": self.speaker_id,
            "license": "MIT"
        })
        return base_info


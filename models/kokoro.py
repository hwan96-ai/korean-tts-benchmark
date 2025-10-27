"""Kokoro TTS 모델 래퍼 모듈."""

from pathlib import Path
from typing import Any, Dict
import yaml
import numpy as np
import torch

from models.base import BaseTTS


class KokoroTTS(BaseTTS):
    """Kokoro-82M TTS 모델 래퍼 클래스.
    
    hexgrad/Kokoro-82M 초경량 모델을 사용하여
    텍스트를 음성으로 변환합니다.
    
    TTS Arena #1 모델로, 82M 파라미터의 경량화된 고품질 TTS입니다.
    
    Attributes:
        device: 모델을 실행할 디바이스
        config: 모델 설정 정보
        sample_rate: 오디오 샘플링 레이트
        output_dir: 생성된 오디오 저장 디렉토리
    """
    
    def __init__(self, device: str = 'auto') -> None:
        """KokoroTTS 초기화.
        
        Args:
            device: 모델을 실행할 디바이스 ('auto', 'cuda', 'cpu')
        
        Raises:
            FileNotFoundError: config 파일을 찾을 수 없는 경우
            ValueError: 잘못된 device 값 또는 config 파싱 실패
        """
        super().__init__(model_name="Kokoro-82M")
        
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
                self.config: Dict[str, Any] = all_config.get('kokoro', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 오류: {e}") from e
        
        # config에서 설정 읽기
        self.model_path: str = self.config.get('model_path', 'hexgrad/Kokoro-82M')
        self.sample_rate: int = self.config.get('sample_rate', 22050)
        self.language: str = self.config.get('language', 'ko')
        
        # 고급 설정
        kokoro_config = self.config.get('config', {})
        self.voice: str = kokoro_config.get('voice', 'af_bella')
        self.speed: float = kokoro_config.get('speed', 1.0)
        
        # 출력 디렉토리 설정
        output_dir_path = self.config.get('output_dir', 'data/output/kokoro')
        self.output_dir = Path(__file__).parent.parent / output_dir_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ KokoroTTS 초기화 완료 (device: {self.device})")
    
    def load_model(self) -> None:
        """Kokoro 모델을 로드합니다.
        
        kokoro 라이브러리의 KPipeline을 사용하여
        Kokoro-82M 모델을 로드합니다.
        
        Raises:
            ImportError: kokoro 라이브러리가 설치되지 않은 경우
            RuntimeError: 모델 로드 실패
        """
        try:
            print(f"Kokoro 모델 로드 중...")
            print("⚠️  참고: Kokoro는 한국어를 공식 지원하지 않습니다.")
            print("  지원 언어: 영어, 일본어, 중국어, 스페인어, 프랑스어 등")
            print("  한국어 텍스트는 영어 음소로 변환되어 부자연스러울 수 있습니다.")
            
            from kokoro import KPipeline
            
            # 영어 파이프라인 사용 ('a' = American English)
            # 한국어는 공식 지원되지 않으므로 영어 폴백 사용
            self.model = KPipeline(lang_code='a')
            
            print(f"✓ Kokoro 모델 로드 완료")
            print(f"  - 샘플레이트: 24000 Hz")
            print(f"  - 언어: 영어 (한국어 폴백)")
            
        except ImportError as e:
            raise ImportError(
                "kokoro 라이브러리가 설치되지 않았습니다.\n"
                "설치: pip install kokoro soundfile\n"
                "espeak-ng도 필요합니다: sudo apt-get install espeak-ng\n"
                "자세한 정보: https://github.com/hexgrad/kokoro"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Kokoro 모델 로드 실패: {e}\n"
                "espeak-ng가 설치되지 않았을 수 있습니다.\n"
                "설치: sudo apt-get install espeak-ng"
            ) from e
    
    def synthesize(
        self,
        text: str,
        **kwargs: Any
    ) -> np.ndarray:
        """텍스트를 음성으로 변환합니다.
        
        Args:
            text: 합성할 텍스트 (한국어는 제한적 지원)
            **kwargs: 추가 모델 파라미터
                - voice (str): 음성 스타일 (기본값: 'af_bella')
                - speed (float): 음성 속도 (기본값: 1.0)
        
        Returns:
            합성된 오디오 데이터 (numpy array, float32, [-1.0, 1.0], 24000 Hz)
        
        Raises:
            RuntimeError: 모델이 로드되지 않았거나 합성 실패
            ValueError: 잘못된 입력 파라미터
        """
        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요."
            )
        
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 합성할 수 없습니다.")
        
        try:
            print(f"텍스트 합성 중: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            # 파라미터 설정
            voice = kwargs.get('voice', self.voice)
            speed = kwargs.get('speed', self.speed)
            
            # Kokoro 파이프라인으로 음성 생성
            # generator는 (graphemes, phonemes, audio) 튜플을 yield
            generator = self.model(
                text,
                voice=voice,
                speed=speed
            )
            
            # 모든 오디오 청크를 수집
            audio_chunks = []
            for gs, ps, audio_chunk in generator:
                # torch.Tensor인 경우 numpy로 변환
                if isinstance(audio_chunk, torch.Tensor):
                    audio_chunk = audio_chunk.cpu().numpy()
                audio_chunks.append(audio_chunk)
            
            # 모든 청크 연결
            if len(audio_chunks) == 0:
                raise RuntimeError("음성 생성 결과가 없습니다.")
            
            audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            
            # 데이터 타입 확인 및 변환
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 오디오가 2D 배열인 경우 1D로 변환
            if audio.ndim > 1:
                if audio.shape[0] < audio.shape[1]:
                    audio = audio[0]  # 첫 번째 채널 사용
                else:
                    audio = audio.flatten()
            
            # 정규화 ([-1.0, 1.0] 범위로)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            # Kokoro는 24000 Hz 사용 (config의 22050 무시)
            actual_sr = 24000
            print(f"✓ 음성 합성 완료 (길이: {len(audio)} samples, 시간: {len(audio)/actual_sr:.2f}초)")
            return audio
            
        except Exception as e:
            raise RuntimeError(
                f"음성 합성 중 오류 발생: {e}. "
                f"입력 텍스트: \"{text[:30]}...\"\n"
                "espeak-ng가 제대로 설치되지 않았을 수 있습니다."
            ) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Kokoro 모델의 상세 정보를 반환합니다.
        
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
            "model_type": "huggingface",
            "voice": self.voice,
            "speed": self.speed,
            "model_size": "82M parameters"
        })
        return base_info


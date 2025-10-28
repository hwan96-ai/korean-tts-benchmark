"""Coqui TTS 모델 래퍼 모듈."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import numpy as np
import torch
import soundfile as sf

from models.base import BaseTTS


class CoquiTTS(BaseTTS):
    """Coqui TTS (XTTS-v2) 모델 래퍼 클래스.
    
    Mozilla의 오픈소스 Coqui TTS 라이브러리를 사용하여
    멀티링구얼 텍스트를 음성으로 변환합니다.
    XTTS-v2 모델을 사용하며 한국어를 지원합니다.
    
    Attributes:
        device: 모델을 실행할 디바이스
        config: 모델 설정 정보
        sample_rate: 오디오 샘플링 레이트
        output_dir: 생성된 오디오 저장 디렉토리
        model_name_full: Coqui TTS 모델 전체 경로
    """
    
    def __init__(self, device: str = 'auto') -> None:
        """CoquiTTS 초기화.
        
        Args:
            device: 모델을 실행할 디바이스 ('auto', 'cuda', 'cpu')
        
        Raises:
            FileNotFoundError: config 파일을 찾을 수 없는 경우
            ValueError: 잘못된 device 값 또는 config 파싱 실패
        """
        super().__init__(model_name="Coqui-TTS-XTTS-v2")
        
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
                self.config: Dict[str, Any] = all_config.get('coqui', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 오류: {e}") from e
        
        # config에서 설정 읽기
        self.model_name_full: str = self.config.get(
            'model_name', 
            'tts_models/multilingual/multi-dataset/xtts_v2'
        )
        self.sample_rate: int = self.config.get('sample_rate', 24000)
        self.language: str = self.config.get('language', 'ko')
        
        # 출력 디렉토리 설정
        output_dir_path = self.config.get('output_dir', 'data/output/coqui')
        self.output_dir = Path(__file__).parent.parent / output_dir_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ CoquiTTS 초기화 완료 (device: {self.device})")
    
    def load_model(self) -> None:
        """Coqui TTS 모델을 로드합니다.
        
        XTTS-v2 멀티링구얼 모델을 로드하고 지정된 디바이스로 전송합니다.
        
        Raises:
            ImportError: Coqui TTS 라이브러리가 설치되지 않은 경우
            RuntimeError: 모델 로드 실패
        """
        try:
            print(f"Coqui TTS 모델 로드 중...")
            print(f"  모델: {self.model_name_full}")
            
            # Coqui TTS import
            try:
                from TTS.api import TTS
            except ImportError as e:
                raise ImportError(
                    "Coqui TTS가 설치되지 않았습니다.\n"
                    "설치 방법:\n"
                    "  pip install TTS\n"
                    "자세한 정보: https://github.com/coqui-ai/TTS"
                ) from e
            
            # 모델 로드
            self.model = TTS(model_name=self.model_name_full)
            
            # GPU 사용 가능 시 설정
            if self.device == 'cuda':
                self.model.to('cuda')
                print(f"✓ GPU로 모델 전송 완료")
            
            print(f"✓ {self.model_name} 로드 완료")
            print(f"  디바이스: {self.device}")
            print(f"  샘플레이트: {self.sample_rate} Hz")
            
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Coqui TTS 모델 로드 실패: {e}\n"
                "XTTS-v2 모델 다운로드에 시간이 걸릴 수 있습니다."
            ) from e
    
    def synthesize(
        self,
        text: str,
        **kwargs: Any
    ) -> np.ndarray:
        """한국어 텍스트를 음성으로 변환합니다.
        
        Coqui TTS XTTS-v2 모델을 사용하여 텍스트를 음성으로 변환합니다.
        XTTS-v2는 zero-shot voice cloning 모델이므로 참조 음성이 필요합니다.
        
        Args:
            text: 합성할 텍스트
            **kwargs: 추가 모델 파라미터
                - language: 언어 코드 (기본값: 'ko')
                - speaker_wav: 참조 음성 파일 경로 (필수)
                              지정하지 않으면 기본 프롬프트 사용
        
        Returns:
            합성된 오디오 데이터 (numpy array, float32, [-1.0, 1.0])
        
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
        
        output_file: Optional[str] = None
        
        try:
            print(f"텍스트 합성 중: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            # 파라미터 설정
            language = kwargs.get('language', self.language)
            speaker_wav = kwargs.get('speaker_wav', None)
            
            # speaker_wav가 없으면 기본 프롬프트 사용
            if speaker_wav is None:
                # 한국어 기본 프롬프트 경로
                default_prompt = Path(__file__).parent.parent / "data" / "input" / "prompts" / "korean_female.wav"
                if default_prompt.exists():
                    speaker_wav = str(default_prompt)
                    print(f"  기본 한국어 프롬프트 사용: {default_prompt.name}")
                else:
                    raise ValueError(
                        f"참조 음성이 필요합니다.\n"
                        f"XTTS-v2는 zero-shot voice cloning 모델입니다.\n"
                        f"'speaker_wav' 파라미터로 참조 음성을 제공하거나,\n"
                        f"기본 프롬프트를 생성하세요: {default_prompt}"
                    )
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                output_file = tmp.name
            
            # Voice cloning 모드 (XTTS-v2는 항상 이 모드)
            print(f"  Voice cloning 모드 (언어: {language})")
            print(f"  참조 음성: {Path(speaker_wav).name}")
            self.model.tts_to_file(
                text=text,
                language=language,
                speaker_wav=speaker_wav,
                file_path=output_file
            )
            
            # 오디오 로드
            audio, sr = sf.read(output_file)
            
            # 샘플레이트 변환 (필요시)
            if sr != self.sample_rate:
                print(f"  샘플레이트 변환: {sr} Hz → {self.sample_rate} Hz")
                try:
                    import librosa
                    audio = librosa.resample(
                        audio,
                        orig_sr=sr,
                        target_sr=self.sample_rate
                    )
                except ImportError as e:
                    raise ImportError(
                        "librosa가 설치되지 않았습니다.\n"
                        "설치: pip install librosa"
                    ) from e
            
            # Float32 변환 및 정규화
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 스테레오인 경우 모노로 변환
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # 정규화
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            duration = len(audio) / self.sample_rate
            print(f"✓ 음성 합성 완료 (길이: {len(audio)} samples, 시간: {duration:.2f}초)")
            
            return audio
            
        except ImportError:
            raise
        except ValueError as e:
            raise ValueError(f"입력 파라미터 오류: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Coqui TTS 합성 실패: {e}\n"
                f"입력 텍스트: \"{text[:50]}...\""
            ) from e
        finally:
            # 임시 파일 정리
            if output_file is not None and os.path.exists(output_file):
                try:
                    os.unlink(output_file)
                except Exception as e:
                    print(f"  경고: 임시 파일 삭제 실패: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Coqui TTS 모델의 상세 정보를 반환합니다.
        
        Returns:
            모델 정보를 담은 딕셔너리
        """
        base_info = super().get_model_info()
        base_info.update({
            "device": self.device,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "output_dir": str(self.output_dir),
            "model_type": "coqui_tts",
            "model_name_full": self.model_name_full,
            "features": ["멀티링구얼", "Voice Cloning"],
            "github": "https://github.com/coqui-ai/TTS"
        })
        return base_info


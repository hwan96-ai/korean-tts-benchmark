"""Zyphra Zonos TTS 모델 래퍼 모듈."""

from pathlib import Path
from typing import Any, Dict
import yaml
import numpy as np
import torch
import torchaudio

from models.base import BaseTTS


class ZonosTTS(BaseTTS):
    """Zyphra Zonos-v0.1 TTS 모델 래퍼 클래스.
    
    Hugging Face의 Zyphra/Zonos-v0.1-transformer 모델을 사용하여
    텍스트를 음성으로 변환합니다.
    
    주의: 이 모델은 영어 중심이며 한국어 지원이 제한적입니다.
    
    Attributes:
        device: 모델을 실행할 디바이스
        config: 모델 설정 정보
        sample_rate: 오디오 샘플링 레이트
        output_dir: 생성된 오디오 저장 디렉토리
    """
    
    def __init__(self, device: str = 'auto') -> None:
        """ZonosTTS 초기화.
        
        Args:
            device: 모델을 실행할 디바이스 ('auto', 'cuda', 'cpu')
        
        Raises:
            FileNotFoundError: config 파일을 찾을 수 없는 경우
            ValueError: 잘못된 device 값 또는 config 파싱 실패
        """
        super().__init__(model_name="Zonos-v0.1")
        
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
                self.config: Dict[str, Any] = all_config.get('zonos', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 오류: {e}") from e
        
        # config에서 설정 읽기
        self.model_path: str = self.config.get('model_path', 'Zyphra/Zonos-v0.1-transformer')
        self.sample_rate: int = self.config.get('sample_rate', 44100)
        self.language: str = self.config.get('language', 'en')
        
        # 고급 설정
        zonos_config = self.config.get('config', {})
        self.cfg_scale: float = zonos_config.get('cfg_scale', 2.0)
        self.max_new_tokens: int = zonos_config.get('max_new_tokens', 2580)
        self.min_p: float = zonos_config.get('min_p', 0.1)
        
        # 출력 디렉토리 설정
        output_dir_path = self.config.get('output_dir', 'data/output/zonos')
        self.output_dir = Path(__file__).parent.parent / output_dir_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ ZonosTTS 초기화 완료 (device: {self.device})")
        print(f"  ⚠️  경고: 한국어 지원이 제한적입니다 (영어 중심 모델)")
    
    def load_model(self) -> None:
        """Zonos 모델을 로드합니다.
        
        Zyphra의 Zonos-v0.1 transformer 모델을 로드합니다.
        
        주의: 이 모델은 실험적 구현입니다.
        실제 Zonos 라이브러리 설치가 필요합니다.
        
        Raises:
            ImportError: zonos 라이브러리가 설치되지 않은 경우
            RuntimeError: 모델 로드 실패
        """
        try:
            print(f"Zonos 모델 로드 중... (from: {self.model_path})")
            print("⚠️  경고: Zonos 모델은 실험적 지원입니다.")
            
            # zonos 라이브러리 import
            try:
                from zonos.model import Zonos
            except ImportError as e:
                raise ImportError(
                    "zonos 라이브러리가 설치되지 않았습니다.\n"
                    "설치: pip install zonos\n"
                    "자세한 정보: https://github.com/Zyphra/Zonos"
                ) from e
            
            # Zonos 모델 로드
            self.model = Zonos.from_pretrained(
                self.model_path,
                device=self.device
            )
            
            # CUDA 사용 시 bfloat16으로 변환
            if self.device == 'cuda':
                try:
                    self.model = self.model.bfloat16()
                except Exception as e:
                    print(f"  경고: bfloat16 변환 실패, float32로 실행: {e}")
            
            # 평가 모드로 설정
            self.model.eval()
            
            print(f"✓ Zonos 모델 로드 완료 (device: {self.device})")
            
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Zonos 모델 로드 실패: {e}"
            ) from e
    
    def synthesize(
        self,
        text: str,
        **kwargs: Any
    ) -> np.ndarray:
        """텍스트를 음성으로 변환합니다.
        
        Zonos 모델을 사용하여 텍스트를 음성으로 변환합니다.
        
        Args:
            text: 합성할 텍스트
            **kwargs: 추가 모델 파라미터
                - language: 언어 코드 (기본값: 'ko')
                - cfg_scale: Classifier-Free Guidance scale (기본값: self.cfg_scale)
                - max_new_tokens: 생성할 최대 토큰 수 (기본값: self.max_new_tokens)
                - min_p: 최소 확률 threshold (기본값: self.min_p)
        
        Returns:
            합성된 오디오 데이터 (numpy array, shape: [samples])
        
        Raises:
            RuntimeError: 모델이 로드되지 않았거나 합성 실패
            ValueError: 잘못된 입력 파라미터
            ImportError: zonos.conditioning 모듈을 찾을 수 없는 경우
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
            
            # conditioning 준비
            try:
                from zonos.conditioning import make_cond_dict
            except ImportError as e:
                raise ImportError(
                    "zonos.conditioning 모듈을 찾을 수 없습니다.\n"
                    "zonos 라이브러리가 올바르게 설치되었는지 확인하세요."
                ) from e
            
            # 파라미터 설정
            language = kwargs.get('language', 'ko')
            cfg_scale = kwargs.get('cfg_scale', self.cfg_scale)
            max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
            min_p = kwargs.get('min_p', self.min_p)
            
            # conditioning 딕셔너리 생성
            cond_dict = make_cond_dict(
                text=text,
                language=language
            )
            
            # 모델에 conditioning 전달
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # 음성 코드 생성
            with torch.no_grad():
                codes = self.model.generate(
                    prefix_conditioning=conditioning,
                    cfg_scale=cfg_scale,
                    max_new_tokens=max_new_tokens,
                    sampling_params=dict(min_p=min_p),
                    progress_bar=False  # 진행 표시줄 비활성화
                )
                
                # autoencoder로 디코딩
                wavs = self.model.autoencoder.decode(codes).cpu()
            
            # numpy로 변환
            audio = wavs.squeeze().numpy()
            
            # float32로 변환 및 정규화
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 값 범위 확인 및 클리핑 (-1.0 ~ 1.0)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            print(f"✓ 음성 합성 완료 (길이: {len(audio)} samples, "
                  f"시간: {len(audio)/self.sample_rate:.2f}초)")
            
            return audio
            
        except ImportError:
            raise
        except ValueError as e:
            raise ValueError(f"입력 파라미터 오류: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"음성 합성 중 오류 발생: {e}. "
                f"입력 텍스트: \"{text[:100]}\""
            ) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Zonos 모델의 상세 정보를 반환합니다.
        
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
            "cfg_scale": self.cfg_scale,
            "max_new_tokens": self.max_new_tokens,
            "min_p": self.min_p,
            "warning": "한국어 지원 제한적 (영어 중심)"
        })
        return base_info

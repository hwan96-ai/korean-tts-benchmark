"""CosyVoice2 TTS 모델 래퍼 모듈."""

from pathlib import Path
from typing import Any, Dict
import sys
import yaml
import numpy as np
import torch

from models.base import BaseTTS


class CosyVoiceTTS(BaseTTS):
    """CosyVoice-300M-SFT TTS 모델 래퍼 클래스.
    
    Alibaba의 FunAudioLLM/CosyVoice-300M-SFT 모델을 사용하여
    멀티링구얼 텍스트를 음성으로 변환합니다.
    한국어를 공식 지원하며 韩语女 화자를 제공합니다.
    
    Attributes:
        device: 모델을 실행할 디바이스
        cosyvoice_root: CosyVoice 설치 경로
        model: 로드된 CosyVoice2 모델
        sample_rate: 오디오 샘플링 레이트
        output_dir: 생성된 오디오 저장 디렉토리
    """
    
    def __init__(self, device: str = 'auto') -> None:
        """CosyVoiceTTS 초기화.
        
        Args:
            device: 모델을 실행할 디바이스 ('auto', 'cuda', 'cpu')
        
        Raises:
            FileNotFoundError: config 파일을 찾을 수 없는 경우
            ValueError: 잘못된 device 값 또는 config 파싱 실패
        """
        super().__init__(model_name="CosyVoice-300M-SFT")
        
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
        
        # CosyVoice 경로 설정
        self.cosyvoice_root = Path.home() / 'CosyVoice'
        
        # config 파일 로드
        try:
            config_path = Path(__file__).parent.parent / "config" / "models_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                all_config = yaml.safe_load(f)
                self.config: Dict[str, Any] = all_config.get('cosyvoice', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 오류: {e}") from e
        
        # config에서 설정 읽기
        self.sample_rate: int = self.config.get('sample_rate', 24000)
        self.language: str = self.config.get('language', 'ko')
        
        # 출력 디렉토리 설정
        output_dir_path = self.config.get('output_dir', 'data/output/cosyvoice')
        self.output_dir = Path(__file__).parent.parent / output_dir_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ CosyVoiceTTS 초기화 완료 (device: {self.device})")
    
    def load_model(self) -> None:
        """CosyVoice-300M-SFT 모델을 로드합니다.
        
        홈 디렉토리의 ~/CosyVoice 설치를 사용하여
        CosyVoice-300M-SFT 모델을 로드합니다.
        
        Raises:
            RuntimeError: CosyVoice가 설치되지 않았거나 모델 로드 실패
            ImportError: 필요한 라이브러리가 설치되지 않은 경우
        """
        try:
            print(f"CosyVoice-300M-SFT 모델 로드 중...")
            
            # 경로 확인
            if not self.cosyvoice_root.exists():
                raise RuntimeError(
                    "CosyVoice가 설치되지 않았습니다.\n"
                    "설치 방법:\n"
                    "1. git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ~/CosyVoice\n"
                    "2. cd ~/CosyVoice && pip install -r requirements.txt\n"
                    "3. pip install modelscope\n"
                    "4. Python에서 모델 다운로드:\n"
                    "   from modelscope import snapshot_download\n"
                    "   snapshot_download('iic/CosyVoice-300M-SFT', cache_dir='pretrained_models')"
                )
            
            # sys.path 추가 (Matcha-TTS)
            matcha_path = self.cosyvoice_root / 'third_party' / 'Matcha-TTS'
            if str(matcha_path) not in sys.path:
                sys.path.insert(0, str(matcha_path))
            
            # CosyVoice root도 sys.path에 추가
            if str(self.cosyvoice_root) not in sys.path:
                sys.path.insert(0, str(self.cosyvoice_root))
            
            # CosyVoice import (CosyVoice, not CosyVoice2)
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice
            except ImportError as e:
                raise ImportError(
                    "CosyVoice 라이브러리 import 실패.\n"
                    "~/CosyVoice가 올바르게 설치되었는지 확인하세요.\n"
                    f"에러: {e}"
                ) from e
            
            # 모델 디렉토리
            model_dir = self.cosyvoice_root / 'pretrained_models' / 'CosyVoice-300M-SFT'
            
            if not model_dir.exists():
                raise RuntimeError(
                    f"모델 디렉토리를 찾을 수 없습니다: {model_dir}\n"
                    "modelscope로 모델을 다운로드하세요:\n"
                    "  from modelscope import snapshot_download\n"
                    "  snapshot_download('iic/CosyVoice-300M-SFT', "
                    f"cache_dir='{self.cosyvoice_root}/pretrained_models')"
                )
            
            # 모델 로드
            print(f"  모델 디렉토리: {model_dir}")
            self.model = CosyVoice(
                str(model_dir),
                load_jit=False,
                load_trt=False
            )
            
            # 사용 가능한 화자 목록 확인
            if hasattr(self.model, 'list_available_spks'):
                available_spks = self.model.list_available_spks()
                print(f"✓ {self.model_name} 로드 완료")
                print(f"  디바이스: {self.device}")
                print(f"  샘플레이트: {self.sample_rate} Hz")
                print(f"  사용 가능한 화자: {available_spks}")
            else:
                print(f"✓ {self.model_name} 로드 완료")
                print(f"  디바이스: {self.device}")
                print(f"  샘플레이트: {self.sample_rate} Hz")
            
        except RuntimeError:
            raise
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"CosyVoice-300M-SFT 모델 로드 실패: {e}\n"
                "설치 및 모델 다운로드를 확인하세요."
            ) from e
    
    def synthesize(
        self,
        text: str,
        **kwargs: Any
    ) -> np.ndarray:
        """한국어 텍스트를 음성으로 변환합니다.
        
        CosyVoice2-0.5B 모델을 사용하여 한국어 텍스트를 음성으로 변환합니다.
        자동으로 <|ko|> 태그를 추가하여 한국어로 처리합니다.
        
        Args:
            text: 합성할 한국어 텍스트
            **kwargs: 추가 모델 파라미터
                - mode: 'sft' (기본) 또는 'zero_shot' (voice cloning, 미지원)
                - spk_id: 화자 ID (sft 모드에서 사용, 기본값: 'default')
        
        Returns:
            합성된 오디오 데이터 (numpy array, float32, [-1.0, 1.0])
        
        Raises:
            RuntimeError: 모델이 로드되지 않았거나 합성 실패
            ValueError: 잘못된 입력 파라미터
            NotImplementedError: Zero-shot 모드 요청 시
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
            
            # 한국어 태그 추가
            if not text.startswith('<|'):
                text_with_tag = f'<|ko|>{text}'
                print(f"  한국어 태그 추가: <|ko|>")
            else:
                text_with_tag = text
            
            mode = kwargs.get('mode', 'sft')
            
            # SFT 모드 (기본 음성)
            if mode == 'sft':
                # kwargs에서 화자 지정 또는 한국어 화자 자동 선택
                spk_id = kwargs.get('spk_id', None)
                
                if spk_id is None:
                    # 한국어 화자 우선 선택
                    if hasattr(self.model, 'list_available_spks'):
                        available_spks = self.model.list_available_spks()
                        # 한국어 화자 찾기
                        korean_spks = [s for s in available_spks if '韩' in s or '한' in s or 'korean' in s.lower()]
                        if korean_spks:
                            spk_id = korean_spks[0]
                            print(f"  한국어 화자 사용: {spk_id}")
                        elif available_spks:
                            spk_id = available_spks[0]
                            print(f"  기본 화자 사용: {spk_id} (사용 가능: {available_spks})")
                    
                    if spk_id is None:
                        raise ValueError("사용 가능한 화자를 찾을 수 없습니다.")
                else:
                    print(f"  SFT 모드 (화자: {spk_id})")
                
                output_gen = self.model.inference_sft(
                    text_with_tag,
                    spk_id=spk_id,
                    stream=False
                )
            else:
                # Zero-shot 모드는 구현 생략
                raise NotImplementedError(
                    "Zero-shot 모드는 아직 지원하지 않습니다. "
                    "mode='sft'를 사용하세요."
                )
            
            # 첫 번째 출력 추출
            audio_tensor = None
            for i, output_dict in enumerate(output_gen):
                if i == 0:
                    audio_tensor = output_dict['tts_speech']
                    break
            
            if audio_tensor is None:
                raise RuntimeError("음성 생성 결과가 없습니다.")
            
            # Tensor to numpy
            audio = audio_tensor.cpu().numpy().squeeze()
            
            # Float32 변환 및 정규화
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            duration = len(audio) / self.sample_rate
            print(f"✓ 음성 합성 완료 (길이: {len(audio)} samples, 시간: {duration:.2f}초)")
            
            return audio
            
        except NotImplementedError:
            raise
        except ValueError as e:
            raise ValueError(f"입력 파라미터 오류: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"CosyVoice 합성 실패: {e}\n"
                f"입력 텍스트: \"{text[:50]}...\""
            ) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """CosyVoice2 모델의 상세 정보를 반환합니다.
        
        Returns:
            모델 정보를 담은 딕셔너리
        """
        base_info = super().get_model_info()
        base_info.update({
            "device": self.device,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "output_dir": str(self.output_dir),
            "model_type": "cosyvoice2",
            "cosyvoice_root": str(self.cosyvoice_root),
            "model_version": "CosyVoice2-0.5B",
            "supported_languages": ["한국어(ko)", "중국어(zh)", "영어(en)", "일본어(ja)"],
            "github": "https://github.com/FunAudioLLM/CosyVoice"
        })
        return base_info

"""CosyVoice2 TTS 모델 래퍼 모듈."""

from pathlib import Path
from typing import Any, Dict
import tempfile
import yaml
import numpy as np
import torch
import torchaudio

from models.base import BaseTTS


class CosyVoiceTTS(BaseTTS):
    """CosyVoice2-0.5B TTS 모델 래퍼 클래스.
    
    Alibaba의 FunAudioLLM/CosyVoice2-0.5B 모델을 사용하여
    멀티링구얼 텍스트를 음성으로 변환합니다.
    
    Attributes:
        device: 모델을 실행할 디바이스
        config: 모델 설정 정보
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
        super().__init__(model_name="CosyVoice2-0.5B")
        
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
                self.config: Dict[str, Any] = all_config.get('cosyvoice', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 오류: {e}") from e
        
        # config에서 설정 읽기
        self.model_path: str = self.config.get('model_path', 'FunAudioLLM/CosyVoice2-0.5B')
        self.sample_rate: int = self.config.get('sample_rate', 24000)
        self.language: str = self.config.get('language', 'ko')
        
        # 고급 설정
        cosy_config = self.config.get('config', {})
        self.temperature: float = cosy_config.get('temperature', 1.0)
        self.speed: float = cosy_config.get('speed', 1.0)
        
        # 출력 디렉토리 설정
        output_dir_path = self.config.get('output_dir', 'data/output/cosyvoice')
        self.output_dir = Path(__file__).parent.parent / output_dir_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 사용 가능한 화자 목록 (load_model에서 설정)
        self.available_speakers: list = []
        
        print(f"✓ CosyVoiceTTS 초기화 완료 (device: {self.device})")
    
    def load_model(self) -> None:
        """CosyVoice 모델을 로드합니다.
        
        CosyVoice 공식 라이브러리를 사용하여 모델을 로드합니다.
        GitHub: https://github.com/FunAudioLLM/CosyVoice
        
        Raises:
            ImportError: cosyvoice 라이브러리를 찾을 수 없는 경우
            RuntimeError: 모델 로드 실패
        """
        try:
            print(f"CosyVoice 모델 로드 중... (from: {self.model_path})")
            print("⚠️  경고: CosyVoice는 공식 라이브러리 설치가 필요합니다.")
            print("  설치 방법:")
            print("    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git")
            print("    cd CosyVoice")
            print("    pip install -r requirements.txt")
            
            # CosyVoice 라이브러리 import
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice
            except ImportError as e:
                raise ImportError(
                    "CosyVoice 라이브러리가 설치되지 않았습니다.\n"
                    "설치 방법:\n"
                    "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
                    "  cd CosyVoice\n"
                    "  conda create -n cosyvoice python=3.8\n"
                    "  conda activate cosyvoice\n"
                    "  pip install -r requirements.txt\n"
                    "자세한 정보: https://github.com/FunAudioLLM/CosyVoice"
                ) from e
            
            # CosyVoice 모델 로드
            # load_jit=False, load_trt=False로 설정하여 기본 PyTorch 모델 사용
            print(f"  모델 로딩 중... (경로: {self.model_path})")
            
            # fp16 설정 (CUDA 사용 시)
            fp16 = (self.device == 'cuda')
            
            self.model = CosyVoice(
                self.model_path,
                load_jit=False,
                load_trt=False,
                fp16=fp16
            )
            
            # 사용 가능한 화자 목록 가져오기 (SFT 모델의 경우)
            if hasattr(self.model, 'list_available_spks'):
                self.available_speakers = self.model.list_available_spks()
                print(f"  사용 가능한 화자: {len(self.available_speakers)}명")
            else:
                self.available_speakers = []
            
            print(f"✓ CosyVoice 모델 로드 완료 (device: {self.device})")
            
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"CosyVoice 모델 로드 실패: {e}\n"
                "CosyVoice는 공식 라이브러리 설치가 필요합니다.\n"
                "GitHub: https://github.com/FunAudioLLM/CosyVoice\n"
                "설치 가이드를 따라주세요."
            ) from e
    
    def synthesize(
        self,
        text: str,
        **kwargs: Any
    ) -> np.ndarray:
        """텍스트를 음성으로 변환합니다.
        
        CosyVoice 공식 라이브러리를 사용하여 텍스트를 음성으로 변환합니다.
        - SFT 모드: speaker 파라미터로 화자 선택
        - Zero-shot 모드: prompt_wav 파라미터로 프롬프트 음성 제공
        
        Args:
            text: 합성할 텍스트 (한국어는 <|ko|> 태그 추가)
            **kwargs: 추가 모델 파라미터
                - speaker: 화자 이름 (SFT 모드, 기본값: '중문여' 또는 첫 번째 화자)
                - prompt_wav: 프롬프트 음성 경로 (Zero-shot 모드)
                - prompt_text: 프롬프트 텍스트 (Zero-shot 모드)
                - stream: 스트리밍 모드 (기본값: False)
        
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
            
            # 파라미터 설정
            speaker = kwargs.get('speaker', None)
            prompt_wav = kwargs.get('prompt_wav', None)
            prompt_text = kwargs.get('prompt_text', None)
            stream = kwargs.get('stream', False)
            
            # 한국어 + 프롬프트 파일이 있으면 자동으로 Zero-shot 모드 사용
            if self.language == 'ko' and prompt_wav is None:
                korean_prompt_path = Path(__file__).parent.parent / 'data/input/prompts/korean_female.wav'
                if korean_prompt_path.exists():
                    prompt_wav = str(korean_prompt_path)
                    prompt_text = "안녕하세요"  # 프롬프트 텍스트
                    print(f"  ✓ 한국어 프롬프트 음성 발견 → Zero-shot 모드 사용")
            
            # 한국어 텍스트에 언어 태그 추가 (필요시)
            if self.language == 'ko' and not text.startswith('<|'):
                text_with_tag = f'<|ko|>{text}'
            else:
                text_with_tag = text
            
            # 오디오 생성
            if prompt_wav is not None and prompt_text is not None:
                # Zero-shot 모드
                print(f"  Zero-shot 모드 (프롬프트: {prompt_text[:20]}...)")
                
                # 프롬프트 음성 로드 (16kHz로 리샘플링)
                prompt_speech_16k = self._load_wav(prompt_wav, 16000)
                
                # inference_zero_shot 호출
                audio_generator = self.model.inference_zero_shot(
                    text_with_tag,
                    prompt_text,
                    prompt_speech_16k,
                    stream=stream
                )
            else:
                # SFT 모드
                # 화자 선택
                if speaker is None:
                    # 한국어 텍스트면 한국어 화자 우선 선택
                    if self.language == 'ko' and '韩语女' in self.available_speakers:
                        speaker = '韩语女'
                        print(f"  한국어 화자 사용: {speaker}")
                    elif self.available_speakers:
                        speaker = self.available_speakers[0]
                        print(f"  기본 화자 사용: {speaker}")
                    else:
                        speaker = '中文女'  # 기본 화자
                        print(f"  기본 화자 사용: {speaker}")
                else:
                    print(f"  SFT 모드 (화자: {speaker})")
                
                # inference_sft 호출
                audio_generator = self.model.inference_sft(
                    text_with_tag,
                    speaker,
                    stream=stream
                )
            
            # 결과 수집
            audio_chunks = []
            for i, result in enumerate(audio_generator):
                if isinstance(result, dict) and 'tts_speech' in result:
                    audio_chunk = result['tts_speech']
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_chunk = audio_chunk.cpu().numpy()
                    audio_chunks.append(audio_chunk)
                else:
                    # 직접 텐서가 반환된 경우
                    if isinstance(result, torch.Tensor):
                        audio_chunk = result.cpu().numpy()
                        audio_chunks.append(audio_chunk)
            
            # 모든 청크 결합
            if len(audio_chunks) == 0:
                raise RuntimeError("음성 생성 결과가 없습니다.")
            
            audio = np.concatenate(audio_chunks, axis=-1) if len(audio_chunks) > 1 else audio_chunks[0]
            
            # 다차원 배열인 경우 1차원으로 변환
            if audio.ndim > 1:
                if audio.shape[0] < audio.shape[1]:
                    audio = audio[0]  # 첫 번째 채널
                else:
                    audio = audio.flatten()
            
            # float32로 변환
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 값 범위 확인 및 정규화 (-1.0 ~ 1.0)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            print(f"✓ 음성 합성 완료 (길이: {len(audio)} samples, "
                  f"시간: {len(audio)/self.sample_rate:.2f}초)")
            
            return audio
            
        except ValueError as e:
            raise ValueError(f"입력 파라미터 오류: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"음성 합성 중 오류 발생: {e}. "
                f"입력 텍스트: \"{text[:30]}...\""
            ) from e
    
    def _load_wav(self, wav_path: str, target_sr: int) -> torch.Tensor:
        """WAV 파일을 로드하고 리샘플링합니다.
        
        Args:
            wav_path: WAV 파일 경로
            target_sr: 목표 샘플레이트
        
        Returns:
            리샘플링된 오디오 텐서
        """
        # torchaudio로 로드
        waveform, sr = torchaudio.load(wav_path)
        
        # 리샘플링
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        return waveform
    
    def get_model_info(self) -> Dict[str, Any]:
        """CosyVoice 모델의 상세 정보를 반환합니다.
        
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
            "model_type": "cosyvoice_official",
            "temperature": self.temperature,
            "speed": self.speed,
            "available_speakers": len(self.available_speakers),
            "github": "https://github.com/FunAudioLLM/CosyVoice"
        })
        return base_info


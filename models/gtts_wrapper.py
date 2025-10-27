"""Google Text-to-Speech (gTTS) 래퍼 모듈."""

import io
import tempfile
from pathlib import Path
from typing import Any, Dict
import yaml
import numpy as np
from gtts import gTTS
from pydub import AudioSegment

from models.base import BaseTTS


class GTTSWrapper(BaseTTS):
    """Google TTS API 래퍼 클래스.
    
    gTTS 라이브러리를 사용하여 Google Text-to-Speech API로
    한국어 텍스트를 음성으로 변환합니다.
    
    gTTS는 API 기반이므로 별도의 모델 로드가 필요 없으며,
    인터넷 연결이 필요합니다.
    
    Attributes:
        language: 음성 언어 코드 (기본값: 'ko')
        tld: Google TTS API의 TLD (기본값: 'com')
        config: 모델 설정 정보
        sample_rate: 오디오 샘플링 레이트
        output_dir: 생성된 오디오 저장 디렉토리
    """
    
    def __init__(self) -> None:
        """GTTSWrapper 초기화.
        
        config 파일에서 설정을 로드하고 출력 디렉토리를 생성합니다.
        
        Raises:
            FileNotFoundError: config 파일을 찾을 수 없는 경우
            ValueError: config 파일 파싱 실패
        """
        super().__init__(model_name="Google-TTS")
        
        # config 파일 로드
        try:
            config_path = Path(__file__).parent.parent / "config" / "models_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                all_config = yaml.safe_load(f)
                self.config: Dict[str, Any] = all_config.get('gtts', {})
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파일 파싱 오류: {e}") from e
        
        # config에서 설정 읽기
        self.language: str = self.config.get('language', 'ko')
        self.sample_rate: int = self.config.get('sample_rate', 24000)
        
        # gTTS 고급 설정
        gtts_config = self.config.get('config', {})
        self.tld: str = gtts_config.get('tld', 'com')
        self.default_slow: bool = gtts_config.get('slow', False)
        
        # 출력 디렉토리 설정
        output_dir_path = self.config.get('output_dir', 'data/output/gtts')
        self.output_dir = Path(__file__).parent.parent / output_dir_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ GTTSWrapper 초기화 완료 (language: {self.language})")
    
    def load_model(self) -> None:
        """모델을 로드합니다.
        
        gTTS는 API 기반으로 동작하므로 별도의 모델 로드가 필요하지 않습니다.
        이 메서드는 API 연결 가능 여부를 확인합니다.
        
        Raises:
            ConnectionError: 인터넷 연결 확인 실패
            RuntimeError: gTTS API 접근 실패
        """
        try:
            print("Google TTS API 연결 확인 중...")
            
            # 간단한 테스트 문장으로 API 연결 확인
            test_tts = gTTS(text="테스트", lang=self.language, tld=self.tld)
            
            # 임시 파일로 테스트 (실제로 생성하지 않음)
            # BytesIO를 사용하여 메모리에서만 처리
            with io.BytesIO() as fp:
                test_tts.write_to_fp(fp)
                fp.seek(0)
                if len(fp.read()) == 0:
                    raise RuntimeError("Google TTS API에서 빈 응답을 받았습니다.")
            
            # API 기반이므로 model 속성을 True로 설정 (로드됨 표시)
            self.model = True
            
            print("✓ Google TTS API 연결 확인 완료")
            
        except Exception as e:
            # 네트워크 오류인 경우
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise ConnectionError(
                    f"인터넷 연결을 확인할 수 없습니다. "
                    f"Google TTS API는 인터넷 연결이 필요합니다. 에러: {e}"
                ) from e
            else:
                raise RuntimeError(
                    f"Google TTS API 접근 중 오류 발생: {e}"
                ) from e
    
    def synthesize(
        self,
        text: str,
        slow: bool = False,
        **kwargs: Any
    ) -> np.ndarray:
        """텍스트를 음성으로 변환합니다.
        
        Google TTS API를 사용하여 텍스트를 음성으로 변환하고,
        MP3 형식의 응답을 numpy array로 변환합니다.
        
        Args:
            text: 합성할 한국어 텍스트
            slow: 느린 속도로 읽기 여부 (기본값: False)
            **kwargs: 추가 gTTS 파라미터
        
        Returns:
            합성된 오디오 데이터 (numpy array, shape: [samples])
        
        Raises:
            RuntimeError: API가 초기화되지 않았거나 합성 실패
            ValueError: 잘못된 입력 파라미터
            ConnectionError: 네트워크 연결 오류
        """
        # API 초기화 확인
        if self.model is None:
            raise RuntimeError(
                "Google TTS API가 초기화되지 않았습니다. "
                "load_model()을 먼저 호출하세요."
            )
        
        # 입력 검증
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 합성할 수 없습니다.")
        
        try:
            print(f"텍스트 합성 중: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            # gTTS 객체 생성
            tts = gTTS(
                text=text,
                lang=self.language,
                slow=slow or self.default_slow,
                tld=self.tld
            )
            
            # MP3 데이터를 메모리에 생성
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # pydub를 사용하여 MP3를 AudioSegment로 로드
            audio_segment = AudioSegment.from_mp3(mp3_fp)
            
            # 샘플레이트 변환 (필요한 경우)
            if audio_segment.frame_rate != self.sample_rate:
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
            
            # 모노로 변환 (스테레오인 경우)
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # numpy array로 변환
            # pydub의 raw_data는 int16 형식
            samples = np.array(audio_segment.get_array_of_samples())
            
            # float32로 변환하고 정규화 (-1.0 ~ 1.0 범위)
            audio = samples.astype(np.float32)
            audio = audio / np.iinfo(np.int16).max
            
            print(f"✓ 음성 합성 완료 (길이: {len(audio)} samples, "
                  f"시간: {len(audio)/self.sample_rate:.2f}초)")
            
            return audio
            
        except ConnectionError as e:
            raise ConnectionError(
                f"Google TTS API 연결 실패. 인터넷 연결을 확인하세요. 에러: {e}"
            ) from e
        except ValueError as e:
            raise ValueError(f"입력 파라미터 오류: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"음성 합성 중 오류 발생: {e}. "
                f"입력 텍스트: \"{text[:100]}\""
            ) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Google TTS의 상세 정보를 반환합니다.
        
        Returns:
            모델 정보를 담은 딕셔너리
            - name: 모델 이름
            - loaded: API 초기화 여부
            - type: 모델 타입
            - language: 언어 코드
            - sample_rate: 샘플링 레이트
            - tld: Google TTS TLD
            - output_dir: 출력 디렉토리
            - model_type: API 기반 모델임을 표시
        """
        base_info = super().get_model_info()
        base_info.update({
            "language": self.language,
            "sample_rate": self.sample_rate,
            "tld": self.tld,
            "output_dir": str(self.output_dir),
            "model_type": "api",
            "requires_internet": True
        })
        return base_info


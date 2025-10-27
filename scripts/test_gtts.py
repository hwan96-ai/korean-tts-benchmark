"""Google TTS (gTTS) 모델 테스트 스크립트.

이 스크립트는 GTTSWrapper 클래스의 기본 사용법을 보여줍니다.
Google TTS API를 사용하여 음성 합성을 수행합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.gtts_wrapper import GTTSWrapper


def main() -> None:
    """Google TTS 모델 테스트 메인 함수."""
    try:
        print("=" * 60)
        print("Google TTS (gTTS) 모델 테스트")
        print("=" * 60)
        
        # 1. 모델 인스턴스 생성
        print("\n[1단계] 모델 인스턴스 생성")
        tts = GTTSWrapper()
        
        # 2. API 연결 확인 (모델 로드)
        print("\n[2단계] Google TTS API 연결 확인")
        print("⚠️  인터넷 연결이 필요합니다.\n")
        tts.load_model()
        
        # 3. 모델 정보 확인
        print("\n[3단계] 모델 정보 확인")
        info = tts.get_model_info()
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # 4. 음성 합성 테스트
        print("\n[4단계] 음성 합성 테스트")
        
        test_sentences = [
            ("안녕하세요, 구글 TTS 테스트입니다.", False),
            ("한국어 음성 합성이 매우 자연스럽습니다.", False),
            ("느린 속도로 읽어봅니다.", True),  # slow=True
        ]
        
        for idx, (text, slow) in enumerate(test_sentences, 1):
            speed_text = "느림" if slow else "보통"
            print(f"\n  테스트 {idx}: \"{text}\" (속도: {speed_text})")
            
            # 음성 합성
            audio = tts.synthesize(text, slow=slow)
            
            # 오디오 저장
            output_path = tts.output_dir / f"test_{idx}_{'slow' if slow else 'normal'}.wav"
            tts.save_audio(audio, str(output_path), sample_rate=tts.sample_rate)
            
            print(f"  ✓ 저장 완료: {output_path}")
        
        # 5. 추가 테스트: 긴 문장
        print("\n[5단계] 긴 문장 테스트")
        long_text = (
            "구글 텍스트 투 스피치는 인공지능 기반의 음성 합성 기술입니다. "
            "다양한 언어를 지원하며, 한국어도 매우 자연스럽게 읽어줍니다. "
            "API 형태로 제공되어 별도의 모델 다운로드 없이 즉시 사용할 수 있습니다."
        )
        
        print(f"  텍스트: \"{long_text[:50]}...\"")
        audio = tts.synthesize(long_text)
        output_path = tts.output_dir / "test_long_sentence.wav"
        tts.save_audio(audio, str(output_path), sample_rate=tts.sample_rate)
        print(f"  ✓ 저장 완료: {output_path}")
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")
        print(f"생성된 파일 위치: {tts.output_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


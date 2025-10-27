"""Zonos-v01 모델 테스트 스크립트.

이 스크립트는 ZonosTTS 클래스의 기본 사용법을 보여줍니다.
실제 모델을 다운로드하고 로드하여 음성 합성을 수행합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.zonos import ZonosTTS


def main() -> None:
    """Zonos-v01 모델 테스트 메인 함수."""
    try:
        print("=" * 60)
        print("Zonos-v01 TTS 모델 테스트")
        print("=" * 60)
        
        # 1. 모델 인스턴스 생성
        print("\n[1단계] 모델 인스턴스 생성")
        tts = ZonosTTS(device='auto')
        
        # 2. 모델 로드
        print("\n[2단계] 모델 로드")
        print("⚠️  첫 실행 시 Hugging Face에서 모델을 다운로드합니다.")
        print("   (약 1.5GB, 시간이 걸릴 수 있습니다)\n")
        tts.load_model()
        
        # 3. 모델 정보 확인
        print("\n[3단계] 모델 정보 확인")
        info = tts.get_model_info()
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # 4. 음성 합성 테스트
        print("\n[4단계] 음성 합성 테스트")
        
        test_sentences = [
            "안녕하세요, 반갑습니다.",
            "한국어 음성 합성 테스트입니다.",
            "오늘 날씨가 정말 좋네요."
        ]
        
        for idx, text in enumerate(test_sentences, 1):
            print(f"\n  테스트 {idx}: \"{text}\"")
            
            # 음성 합성
            audio = tts.synthesize(text, speaker_id=0, speed=1.0)
            
            # 오디오 저장
            output_path = tts.output_dir / f"test_{idx}.wav"
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


import torch
from nemo.collections.asr.models import ASRModel
from nemo_forced_aligner.aligner import ForcedAligner

def generate_word_timestamps(audio_path, transcript_text, language="sme", device="cuda" if torch.cuda.is_available() else "cpu"):
    if not audio_path or not transcript_text:
        return []

    # 加载预训练ASR模型（需替换为支持目标语言的模型）
    asr_model = ASRModel.from_pretrained(
        model_name="stt_en_conformer_ctc_large",  # 替换为合适的模型
        map_location=device
    )
    
    # 初始化对齐器
    aligner = ForcedAligner(
        asr_model=asr_model,
        device=device
    )
    
    # 执行对齐
    try:
        alignments = aligner.align(
            audio_path=audio_path,
            text=transcript_text,
            language=language
        )
        
        # 转换为与现有格式兼容的时间戳
        return [
            {
                "start": float(seg.start_time),
                "end": float(seg.end_time),
                "text": seg.text
            } 
            for seg in alignments.segments
        ]
    except Exception as e:
        print(f"Alignment failed: {str(e)}")
        return []
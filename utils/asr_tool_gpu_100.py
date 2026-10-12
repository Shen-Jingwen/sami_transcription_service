import numpy as np
from transformers import pipeline
from librosa import resample
import torch
import soundfile as sf
import os
import glob

# 仅保留large模型
ASR_MODEL = "shenjingwen/wav2vec2-ft-9h-1"
ASR_MODEL = "shenjingwen/wav2vec2-ft-9h"
ASR_MODEL = "shenjingwen/wav2vec2-ft-9h2000" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-9h800" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-9h2800" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-9h1200"  
ASR_MODEL = "shenjingwen/wav2vec2-ft-30h-2800" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-30h-3600" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-30h-6800" 
ASR_MODEL = "shenjingwen/wav2vec2-base-gie30h-18000" 
ASR_MODEL = "GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned"

# GPU环境检查
def check_gpu_env():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please check your GPU/torch installation.")
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    return torch.device("cuda:0")

class CustomASR_GPU:
    def __init__(self):
        self.device = check_gpu_env()
        self.pipe = self._load_model_to_gpu()
        self.chunk_length_s = 100

    def _load_model_to_gpu(self):
        """GPU专属模型加载"""
        print(f"Loading {ASR_MODEL} to GPU...")
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=ASR_MODEL,
                device=self.device,
                return_timestamps='word',
                torch_dtype=torch.float32,
                model_kwargs={"cache_dir": "./model_cache"},
                decoder_kwargs={ 
                    "beam_size": 1,
                    "ctc_zero_infinity": False
                }
            )
            print(f"✅ Model loaded to GPU successfully")
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load model to GPU: {str(e)}")

    def transcribe_gpu(self, audio_path):
        """使用HuggingFace pipeline进行转录（与test_asr.py一致）"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {str(e)}")

        # 预处理：转单声道 + 重采样到16kHz
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != 16000:
            audio = resample(audio, orig_sr=sr, target_sr=16000)

        # 直接调用pipeline（与test_asr.py的transcribe_with_official_chunk一致）
        with torch.no_grad():
            result = self.pipe(
                audio,
                chunk_length_s=self.chunk_length_s,
                stride_length_s=5,
                # return_timestamps='word'
            )
        
        full_text = result.get("text", "").strip()
        
        # 提取时间戳
        timestamps = []
        for chunk in result.get("chunks", []):
            timestamps.append({
                "text": chunk["text"].strip(),
                "start": round(chunk["timestamp"][0], 3),
                "end": round(chunk["timestamp"][1], 3)
            })
        
        print(f"len={len(timestamps)}={len(full_text.split(' '))}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        return full_text, timestamps

# 批量处理
def batch_process_mp3_gpu(input_dir: str, output_root: str = "asr_9h"):
    asr = CustomASR_GPU()

    if not os.path.exists(input_dir):
        raise RuntimeError(f"Input directory {input_dir} not found")

    mp3_files = glob.glob(os.path.join(input_dir, "*.mp3")) + glob.glob(os.path.join(input_dir, "*.wav"))
    if not mp3_files:
        print(f"No audio files found in {input_dir}")
        return

    total = len(mp3_files)
    for idx, mp3_path in enumerate(mp3_files, 1):
        print(f"\n[{idx}/{total}] Processing: {mp3_path}")
        file_name = os.path.basename(mp3_path).replace(".mp3", "").replace(".wav", "")
        output_dir = os.path.join(output_root, file_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            full_text, timestamps = asr.transcribe_gpu(mp3_path)
        except Exception as e:
            print(f"❌ Failed to process {mp3_path}: {str(e)}")
            torch.cuda.empty_cache()
            continue

        # 保存结果
        with open(os.path.join(output_dir, "transcription.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

        with open(os.path.join(output_dir, "word_timestamps.txt"), "w", encoding="utf-8") as f:
            f.write("单词\t开始时间(秒)\t结束时间(秒)\n")
            f.write("-" * 50 + "\n")
            for ts in timestamps:
                f.write(f"{ts['text']}\t{ts['start']}\t{ts['end']}\n")

        print(f"✅ Saved to {output_dir}")

if __name__ == "__main__":
    # INPUT_DIR = "/m/home/home0/01/shenj4/unix/RA_project/ms_thesis/example/test"
    INPUT_DIR = "/m/home/home0/01/shenj4/unix/RA_project/ms_thesis/example/giellagas_41/test_labeled"
    output_root = "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_gpu_output2"
    try:
        batch_process_mp3_gpu(input_dir=INPUT_DIR, output_root=output_root)
    except Exception as e:
        print(f"❌ Batch process failed: {str(e)}")
        torch.cuda.empty_cache()
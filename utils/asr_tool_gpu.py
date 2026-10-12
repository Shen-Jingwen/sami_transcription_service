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
# ASR_MODEL = "shenjingwen/wav2vec2-ft-9h-s2000" 
# ASR_MODEL = "shenjingwen/wav2vec2-ft-9h-s3200" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-30h-2800" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-30h-3600" 
ASR_MODEL = "shenjingwen/wav2vec2-ft-30h-6800" 
# ASR_MODEL = "shenjingwen/wav2vec2-ft-50h-4800" 
# ASR_MODEL = "shenjingwen/wav2vec2-ft-50h-9600" 
# ASR_MODEL = "shenjingwen/wav2vec2-ft-50h-21600" 
# ASR_MODEL = "shenjingwen/wav2vec2-ft-50h-24000" 
# ASR_MODEL = "shenjingwen/wav2vec2-ft-50h-34200" 
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
        self.chunk_length_s = 120  # Length of audio chunks to process
        
        self.min_overlap_words = 3  # 仅处理≥3个单词的重叠
        self.check_overlap_words = 10  # 每次检查前10个单词的重叠
        self.overlap_ratio = 1.5
        self.overlap_ratio2 = 2

    def _load_model_to_gpu(self):
        """GPU专属模型加载：优化显存+数据类型"""
        print(f"Loading {ASR_MODEL} to GPU...")
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=ASR_MODEL,
                device=self.device,  # 直接指定GPU设备
                return_timestamps='word',
                torch_dtype=torch.float16,  # 半精度减少显存占用
                model_kwargs={"cache_dir": "./model_cache"}  # 模型缓存避免重复下载
            )
            print(f"✅ Model loaded to GPU successfully")
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load model to GPU: {str(e)}")

    def preprocess_audio(self, audio, sr):
        """GPU适配预处理：张量移至GPU+优化格式"""
        # 转单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # 转float32（GPU计算兼容）
        audio = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        # 重采样到16kHz
        if sr != 16000:
            audio = resample(audio, orig_sr=sr, target_sr=16000)
        # 转换为GPU张量（关键：数据移至GPU）
        audio_tensor = torch.from_numpy(audio).to(self.device)
        return audio_tensor

    def transcribe_gpu(self, audio_path):
        """纯GPU转录：无分块/线程/临时文件"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {str(e)}")

        # 预处理+移至GPU
        audio_gpu = self.preprocess_audio(audio, sr)

        # GPU批量转录（核心：一次性处理，无分块）
        with torch.no_grad():  # 禁用梯度计算，减少显存占用
            result = self.pipe(
                audio_gpu.cpu().numpy(),  # pipeline暂不直接支持GPU张量，转numpy但模型在GPU
                return_timestamps='word',
                chunk_length_s=self.chunk_length_s,  # 适度分块（GPU显存友好） 待处理overlap
                stride_length_s=(self.overlap_ratio, self.overlap_ratio2)  # 重叠步长优化转录精度
            )

        # 提取结果
        # full_text = result["text"].strip()
        # timestamps = []
        # for chunk in result.get("chunks", []):
            # tt=chunk["text"].strip()
            # print(f"{tt}")
            # timestamps.append({
                # "text": tt,
                # "start": round(chunk["timestamp"][0], 2),
                # "end": round(chunk["timestamp"][1], 2)
            # })
        # peak_memory = torch.cuda.max_memory_allocated()/1024**3
        # print(f"Peak GPU memory: {peak_memory:.2f}GB")
        text = ""
        last_chunk_words = []
        timestamps = []
        global_offset = 0.0
        last_chunk_end = 0.0

        # 遍历pipeline返回的chunks（按时间分块的转录结果）
        for chunk_result in result.get("chunks", []):
            # 提取当前chunk的文本和时间戳
            original_chunk_text  = chunk_result["text"].strip()
            current_words = original_chunk_text .split()
            overlap = 0

            # 处理分块导致的重叠/截断（完全复用custom_asr的逻辑）
            if last_chunk_words:
                max_possible_overlap = min(len(last_chunk_words), len(current_words), self.check_overlap_words)
                for i in range(max_possible_overlap, 0, -1):
                    last_part = " ".join(last_chunk_words[-i:]).lower().strip(".,!?")
                    current_part = " ".join(current_words[:i]).lower().strip(".,!?")
                    # print(f"""overlap fix last_part={last_part} current_part={current_part} start={chunk_result["timestamp"]}""")
                    if current_part.startswith(last_part):
                        overlap = i
                        if not current_part == last_part:
                            # 上一个chunk的最后一个单词不完整，截断修复
                            overlap = i - 1
                            text = " ".join(text.split()[:-1])
                            timestamps.pop()
                            # print(f"""overlap fix last_part={last_part} current_part={current_part} start={round(chunk_result["timestamp"][0], 3)}""")
                        break
                    else:
                        # 检测上一个chunk末尾多截断的情况
                        last_part_2 = " ".join(last_chunk_words[-i-1:-1]).lower().strip(".,!?") if i+1 <= len(last_chunk_words) else ""
                        if last_part_2 and current_part.startswith(last_part_2):
                            overlap = i
                            text = " ".join(text.split()[:-1])
                            timestamps.pop()
                            print(f"""overlap fix last_part={last_part} current_part={current_part} last_part_2={last_part_2} start={round(chunk_result["timestamp"][0], 3)}""")
                            break
                        else:
                            # 检测当前chunk开头截断的情况
                            current_part_2 = " ".join(current_words[1:i+1]).lower().strip(".,!?") if i+1 <= len(current_words) else ""
                            if current_part_2.startswith(last_part):
                                overlap = i - 1
                                current_words = current_words[1:]
                                text = " ".join(text.split()[:-1])
                                timestamps.pop()
                                print(f"""overlap fix last_part={last_part} current_part={current_part} current_part_2={current_part_2} start={round(chunk_result["timestamp"][0], 3)}""")
                                break

            # 移除重叠部分的文本
            chunk_text = ""
            if overlap >= self.min_overlap_words:
                chunk_text = " ".join(current_words[overlap:]).strip()
            else:
                chunk_text = original_chunk_text.strip()

            # 拼接有效文本并更新历史单词
            if chunk_text:
                text = f"{text} {chunk_text}" if text else chunk_text
                last_chunk_words = current_words[-self.check_overlap_words:]


            # 处理时间戳（偏移校准，避免重叠）
            adjusted_ts = {
                "text": chunk_result["text"].strip(),
                "start": round(chunk_result["timestamp"][0] + global_offset, 3),
                "end": round(chunk_result["timestamp"][1] + global_offset, 3)
            }

            # 跳过重叠部分的时间戳
            if overlap == 0 or not (adjusted_ts["text"].split()[:overlap] == current_words[:overlap]):
                timestamps.append(adjusted_ts)
            elif len(timestamps) != len(text.strip().split(' ')):
                if chunk_result['timestamp'][0]>=timestamps[-1]["end"] and chunk_result['text']==timestamps[-1]["text"]:
                    timestamps.append(adjusted_ts)
            
            if len(timestamps) != len(text.strip().split(' ')):
            # if text.strip().split(' ')[-1] !=  timestamps[-1]["text"]:  
                print(f"timestamps={timestamps}")
                print(f"text={text}")
                print(f"chunk_result={chunk_result}")
                break

            # 更新全局时间偏移
            last_chunk_end = chunk_result["timestamp"][1]
            # global_offset += last_chunk_end - (self.chunk_length_s * self.overlap_ratio) 

        # 最终文本清理
        full_text = text.strip()
        print(f"len={len(timestamps)}={len(full_text.split(' '))}")


        # 清理GPU缓存（关键：避免显存泄漏）
        torch.cuda.empty_cache()
        return full_text, timestamps

# 批量处理（GPU优化版）
def batch_process_mp3_gpu(input_dir: str, output_root: str = "asr_9h"):
    # 初始化GPU ASR（仅加载一次模型）
    asr = CustomASR_GPU()

    # 检查输入目录
    if not os.path.exists(input_dir):
        raise RuntimeError(f"Input directory {input_dir} not found")

    # 遍历MP3文件
    mp3_files = glob.glob(os.path.join(input_dir, "*.mp3"))+glob.glob(os.path.join(input_dir, "*.wav"))
    if not mp3_files:
        # mp3_files = glob.glob(os.path.join(input_dir, "*.wav"))
        if not mp3_files:
            print(f"No MP3 files found in {input_dir}")
            return

    # 逐个处理
    total = len(mp3_files)
    for idx, mp3_path in enumerate(mp3_files, 1):
        print(f"\n[{idx}/{total}] Processing: {mp3_path}")
        file_name = os.path.basename(mp3_path).replace(".mp3", "")
        output_dir = os.path.join(output_root, file_name)
        os.makedirs(output_dir, exist_ok=True)

        # GPU转录
        try:
            full_text, timestamps = asr.transcribe_gpu(mp3_path)
        except Exception as e:
            print(f"❌ Failed to process {mp3_path}: {str(e)}")
            torch.cuda.empty_cache()  # 出错也清理显存
            continue

        # 保存结果
        # 1. 纯文本
        with open(os.path.join(output_dir, "transcription.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

        # 2. 时间戳
        with open(os.path.join(output_dir, "word_timestamps.txt"), "w", encoding="utf-8") as f:
            f.write("单词\t开始时间(秒)\t结束时间(秒)\n")
            f.write("-" * 50 + "\n")
            for ts in timestamps:
                f.write(f"{ts['text']}\t{ts['start']}\t{ts['end']}\n")

        print(f"✅ Saved to {output_dir}")

# 运行入口
if __name__ == "__main__":
    INPUT_DIR = "/m/home/home0/01/shenj4/unix/RA_project/ms_thesis/example/test2"
    # INPUT_DIR = "/u/01/shenj4/unix/RA_project/ms_thesis/example/giellagas_41/test"
    # INPUT_DIR = "/u/01/shenj4/unix/RA_project/ms_thesis/example/test"
    # INPUT_DIR = "/u/01/shenj4/unix/RA_project/ms_thesis/example/giellagas_41/test_labeled"
    
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_9h_1"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_9h2000"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_9h800"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_9h2800"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_9h1200"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_9hs2000"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_9hs3200"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_30h2800"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_30h3200"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_30h3600"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_30h6800"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_50h4800"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_50h9600"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_50h21600"
    output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_gpu_output2"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_50h24000"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_50h31800"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_50h34200"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_base"
    # output_root= "/u/01/shenj4/unix/RA_project/ms_thesis/example/asr_base_gie30h18000"
    try:
        batch_process_mp3_gpu(input_dir=INPUT_DIR, output_root=output_root)
    except Exception as e:
        print(f"❌ Batch process failed: {str(e)}")
        torch.cuda.empty_cache()  # 最终清理显存
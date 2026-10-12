import os
import glob
import json
import re
import torch

# 彻底清理一次显存缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 导入 ctc_forced_aligner 的核心组件
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# =====================================================================
# 1. 核心强制对齐函数（封装自您的原有逻辑）
# =====================================================================
def generate_word_timestamps(audio_path, transcript_text, language="sme", device="cuda" if torch.cuda.is_available() else "cpu", batch_size=16):
    """
    输入音频路径和纯台词文本，利用 GPU 加速生成高精度单词级别时间戳
    """
    if not audio_path or not transcript_text:
        return []
        
    # 加载对齐模型与分词器（根据 GPU/CPU 自动选择半精度或单精度）
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # 读取音频波形并加载至指定设备
    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)
    text = transcript_text.strip()

    # 1. 生成发射矩阵 (Emissions)
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    # 2. 文本预处理（规范化、小写化，内部会自动安全滤除发音无关的标点符号）
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
    )

    # 3. 求解最优对齐路径
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    # 4. 获取跨度区间并后处理得到最终秒级时间戳
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    return word_timestamps


# =====================================================================
# 2. 批量匹配与标签动态过滤逻辑
# =====================================================================
def batch_align_with_reference(audio_dir, text_dir, output_root_dir, language="sme"):
    """
    自动匹配同名的音频和 txt 文本。
    输出：
    1. 原始 word_timestamps.json 矩阵
    2. 带段落时间戳的 transcription.txt (格式: start - end speaker: 内容)
    """
    os.makedirs(output_root_dir, exist_ok=True)
    
    audio_extensions = ['*.wav', '*.mp3', '*.WAV']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
        
    print(f"🎵 找到 {len(audio_files)} 个音频文件，开始匹配同名 Reference 文本...")
    
    success_count = 0
    for idx, audio_path in enumerate(audio_files, 1):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        # if "17" not in base_name:
            # continue
        text_path = os.path.join(text_dir, f"{base_name}.txt")
        
        print(f"\n[{idx}/{len(audio_files)}] 正在处理文件: {base_name}")
        
        if not os.path.exists(text_path):
            print(f"  ⚠️ 跳过：未找到对应的参考文本文件 -> {text_path}")
            continue
            
        try:
            # 1. 结构化读取：不仅要干净台词，还要记录每一个 speaker 的台词是由多少个单词组成的
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()
            
            clean_lines = []        # 传给对齐器的纯台词行
            paragraph_meta = []     # 存储每行对应的 (speaker_name, word_count)
            
            for line in raw_lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                # 提取说话人标签（捕获名字和台词内容）
                match = re.match(r'^\s*([a-zA-Z0-9\u00C0-\u017F\s]{1,25})\s*[:：](.*)$', line_str)
                if match:
                    speaker = match.group(1).strip()
                    content = match.group(2).strip()
                else:
                    # 如果某一行没有冒号（可能是上一个人的换行文本），默认归为上一人，若无上一人则设为未知
                    speaker = paragraph_meta[-1][0] if paragraph_meta else "unknown"
                    content = line_str
                
                if content:
                    clean_lines.append(content)
                    # 计算这一轮对话包含多少个单词（按空格切分）
                    word_count = len(content.split())
                    paragraph_meta.append((speaker, word_count))
            
            # 将合并后的纯台词送入强制对齐器
            transcript_text = " ".join(clean_lines)
            
            if not transcript_text.strip():
                print(f"  ⚠️ 跳过：参考文本中无有效台词")
                continue
                
            print(f"  🎬 正在调取 GPU 生成强制对齐时间戳...")
            word_timestamps = generate_word_timestamps(
                audio_path=audio_path,
                transcript_text=transcript_text,
                language=language
            )
            
            # 2. 创建输出目录
            file_output_dir = os.path.join(output_root_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # 格式 1：保存原始标准的结构化 JSON 文件
            output_json_path = os.path.join(file_output_dir, "word_timestamps.json")
            with open(output_json_path, "w", encoding="utf-8") as f_out:
                json.dump(word_timestamps, f_out, ensure_ascii=False, indent=2)
                
            # 格式 2：按段落单词数滑动切片，生成带时间戳的 transcription.txt
            output_txt_path = os.path.join(file_output_dir, "transcription.txt")
            
            word_pointer = 0  # 单词流全局指针
            total_words_aligned = len(word_timestamps)
            
            with open(output_txt_path, "w", encoding="utf-8") as f_txt:
                for speaker, count in paragraph_meta:
                    if count == 0:
                        continue
                    
                    # 确定当前段落对应的单词切片区间 [start_idx : end_idx]
                    start_idx = word_pointer
                    end_idx = min(word_pointer + count, total_words_aligned)
                    
                    # 如果指针越界了，说明对齐工具漏掉了后面的词，做兜底防崩处理
                    if start_idx >= total_words_aligned:
                        break
                        
                    paragraph_words = word_timestamps[start_idx:end_idx]
                    
                    if paragraph_words:
                        # 核心获取：段落第一个单词的开始时间，最后一个单词的结束时间
                        p_start = paragraph_words[0].get("start", 0.0)
                        p_end = paragraph_words[-1].get("end", 0.0)
                        
                        # 把这批单词重新拼回原汁原味的文本内容（优先从 ctc-aligner 的 text 中取词）
                        p_content = " ".join(item.get("text", item.get("word", "")) for item in paragraph_words)
                        
                        # 写入规范化格式：start - end speaker: 内容
                        f_txt.write(f"*{p_start:.3f} - {p_end:.3f}* {speaker}: {p_content}\n")
                    
                    # 指针向后平移
                    word_pointer += count
            
            print(f"  ✅ 成功！JSON 与段落 TXT 已保存至：{file_output_dir}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 该文件对齐失败，错误原因: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n?? 批量强制对齐运行完毕！成功处理了 {success_count}/{len(audio_files)} 个文件。")
    
# =====================================================================
# 3. 自动化任务执行入口
# =====================================================================
if __name__ == "__main__":
    # 填入你在 Triton 集群上的真实工作路径
    tt='/m/home/home0/01/shenj4/unix/RA_project/ms_thesis'
    # AUDIO_DIRECTORY = tt+"/example/test1"   
    AUDIO_DIRECTORY = tt+"/example/16k_yle_train"   
    # TEXT_DIRECTORY = tt+"/example/giellagas_41/docx_pre/0"
    TEXT_DIRECTORY = tt+"/example/16k_yle_train/txt"
    OUTPUT_DIRECTORY = tt+"/example/train/ctcfa_outputs_yle"      # 最终生成的时间戳结果（包含json和txt）存放路径
    if "triton_scratch" in TEXT_DIRECTORY: 
        AUDIO_DIRECTORY =   os.path.expanduser(AUDIO_DIRECTORY)
        TEXT_DIRECTORY = os.path.expanduser(TEXT_DIRECTORY)
    print(TEXT_DIRECTORY)
    # 一键启动
    batch_align_with_reference(
        audio_dir=AUDIO_DIRECTORY,
        text_dir=TEXT_DIRECTORY,
        output_root_dir=OUTPUT_DIRECTORY,
        language="sme"                            # 北萨米语 ISO 代码
    )
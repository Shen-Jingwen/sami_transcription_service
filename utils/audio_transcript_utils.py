import json, re
import time
import tempfile
import soundfile as sf
import shutil
from bs4 import BeautifulSoup
# from whisper_asr import WhisperAutomaticSpeechRecognizer
from asr_instance import asr
from custom_diarization import SpeakerDiarizer
from .forced_alignment import generate_word_timestamps



def save_transcript(transcript_text):
    if not transcript_text or not isinstance(transcript_text, str):
        print(f"Error: Invalid transcript text, type()={type(transcript_text)} transcript_text={transcript_text}")
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix=f"transcript_{timestamp}_")
    temp_file.close()
    # print(transcript_text)
    # print(temp_file.name)
    # print(temp_file)
    with open(temp_file.name, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    return temp_file.name


def save_audio(audio_data):
    if audio_data is None:
        print("Error: No audio data provided")
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"recording_{timestamp}_")
    temp_file.close()
    print(f"Saving audio to: {temp_file.name}")

    if isinstance(audio_data, tuple):
        sr, y = audio_data
        sf.write(temp_file.name, y, sr)
    else:
        shutil.copy(audio_data, temp_file.name)

    return temp_file.name



def resolve_text_to_save(edited_text, plain_text):
    return save_transcript(edited_text if edited_text.strip() else plain_text)


def time_str_to_seconds(time_str):
    """
    time transfer（0:00、00:00:00）to seconds
    """
    # print(f"time_str_to_seconds={time_str}")
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:  # 0:11
        minutes, seconds = parts
        return minutes * 60 + seconds 
    elif len(parts) == 3:  # 1:10:03
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds*1.0 
    else:
        raise ValueError(f"time error: {time_str}")

   
def seconds_to_time_str(s):
    """
    time transfer seconds to (0:00、00:00:00)
    """
    if s is None or s == 0:
        return "0:00"
    try:
        s = float(s)
    except (ValueError, TypeError):
        return "0:00"
    hours = int(s / 3600)
    minutes = int((s % 3600) / 60)
    seconds = int(s % 60)
    if hours > 0:
        # HH:MM:SS
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        # MM:SS
        return f"{minutes:02d}:{seconds:02d}"    



def get_word_timestamp(word, previousWord, nextWord, timestamps):
    # print (f"""double click {word}  pre/next {previousWord} {nextWord}  timestamps={timestamps} """)
    if not word or not timestamps:
        return 0.0, "0:00"
    timestamps = list(timestamps)
    for index, item in enumerate(timestamps):
        if item['text'].strip() == word.strip():
            print (f"""index {index}  timestamps[index]={timestamps[index]}""")
            if index == 0:
                if timestamps[index+1]['text'].strip() == nextWord.strip():
                    word_timestamp = float(item['start'])
                    str_time=seconds_to_time_str(word_timestamp)
                    print (f"""double click {word}  word_timestamp={word_timestamp}={str_time} pre/next {previousWord} {nextWord}""")
                    # return float(item['start'])
                    return word_timestamp, str_time
            elif nextWord in [ None, "None"]:
                if timestamps[index-1]['text'].strip() == previousWord.strip():
                    word_timestamp = float(item['start'])
                    str_time=seconds_to_time_str(word_timestamp)
                    print (f"""double click {word}  word_timestamp={word_timestamp}={str_time} pre/next {previousWord} {nextWord}""")
                    # return float(item['start'])
                    return word_timestamp, str_time
            else:
                if timestamps[index-1]['text'].strip() == previousWord.strip() and timestamps[index+1]['text'].strip() == nextWord.strip():
                    word_timestamp = float(item['start'])
                    str_time=seconds_to_time_str(word_timestamp)
                    print (f"""double click {word}  word_timestamp={word_timestamp}={str_time} pre/next {previousWord} {nextWord}""")
                    # return float(item['start'])
                    return word_timestamp, str_time
    return 0.0, "0:00"


def process_audio_with_timestamps(audio_file):
    if not audio_file:
        yield "", []
        return
    # asr._stop_event.set()
    # if asr._current_process and asr._current_process.is_alive():
        # asr._current_process.join(timeout=1.0)
    
    asr.stop_current_task()
    asr._stop_event.clear()
    transcript_text = ""
    timestamps = []
    last_result = None
    try:  
        for result in asr.transcribe_with_diarization_file(audio_file):
            if asr._stop_event.is_set():  
                break
            transcript_text = result[0]
            # timestamps = generate_word_timestamps(audio_file, transcript_text)
            timestamps = result[3] if len(result) > 3 else []
            raw_asr_text = result[4] if len(result) > 4 else transcript_text
            last_result = (transcript_text, timestamps, raw_asr_text)
            yield transcript_text, timestamps, raw_asr_text
        if not asr._stop_event.is_set() and last_result:
            yield last_result
    except Exception as e:
        duration = sf.info(audio_file).duration if audio_file else 1000.0
        yield transcript_text, [(0.0, duration, transcript_text)] if transcript_text else [], transcript_text


def update_highlight(current_time, timestamps, original_text):
    original_text = original_text.replace("<br><br>", "\n\n").replace("<br>", "\n")
    if not timestamps or current_time is None:
        return original_text
    try:
        current_time = float(current_time)
    except (TypeError, ValueError):
        return original_text
        
    has_speaker = any(
        isinstance(w, dict) and w.get("speaker") and w.get("speaker") != "UNKNOWN"
        for w in timestamps
    )
    
    if not has_speaker:
        highlighted_html = ""
        current_pos = 0
        for w in timestamps:
            if not isinstance(w, dict):
                continue
            word = w.get("text", "")
            start = float(w.get("start", 0))
            if not word:
                continue
            pattern = r'(?:(?<=\s)|^)' + re.escape(word) + r'(?=\s|$)'
            m = re.search(pattern, original_text[current_pos:])
            if m is None:
                continue
            word_pos = current_pos + m.start()
            if word_pos > current_pos:
                highlighted_html += original_text[current_pos:word_pos]
            if start <= current_time + 1.5:
                highlighted_html += f'<span class="highlight">{word}</span>'
            else:
                highlighted_html += f'<span class="word">{word}</span>'
            current_pos = word_pos + len(word)
        if current_pos < len(original_text):
            highlighted_html += original_text[current_pos:]
        return f"<span>{highlighted_html}</span>"
        
    # 按 speaker 分组，每组记录 start/end/words
    segments = []
    cur = None
    for item in timestamps:
        if not isinstance(item, dict):
            continue
        spk = item.get("speaker", "UNKNOWN")
        word = item.get("text", "")
        start = float(item.get("start", 0))
        end = float(item.get("end", 0))

        if cur is None or spk != cur["speaker"]:
            if cur is not None:
                segments.append(cur)
            cur = {"speaker": spk, "start": start, "end": end, "words": [(word, start)]}
        else:
            cur["words"].append((word, start))
            cur["end"] = end
    if cur is not None:
        segments.append(cur)

    # 拼 HTML
    parts = []
    current_pos = 0
    for seg in segments:
        spk_label = SpeakerDiarizer.speaker_to_label(seg["speaker"])
        start_str = SpeakerDiarizer._fmt_time(seg["start"])
        end_str = SpeakerDiarizer._fmt_time(seg["end"])
        for word, start in seg["words"]:
            if not word:
                continue
            pattern = r'(?:(?<=\s)|^)' + re.escape(word) + r'(?=\s|$)'
            m = re.search(pattern, original_text[current_pos:])
            if m is None:
                continue
            word_pos = current_pos + m.start()
            if word_pos > current_pos:
                parts.append(original_text[current_pos:word_pos])
            if start <= current_time + 1.5:
                parts.append(f'<span class="highlight">{word}</span> ')
            else:
                parts.append(f'<span class="word">{word}</span> ')
            current_pos = word_pos + len(word)
    if current_pos < len(original_text):
        parts.append(original_text[current_pos:])
    return f"<span>{''.join(parts)}</span>"
    



def normalize_timestamps(raw_timestamps):
    """
    把 generate_word_timestamps 的返回值统一成 [{"text": str, "start": float, "end": float}, ...]
    兼容 tuple ("word", (start, end)) 和 dict 两种格式。
    """
    normalized = []
    if not raw_timestamps:
        return normalized
    for w in raw_timestamps:
        if isinstance(w, dict):
            normalized.append(w)
        elif isinstance(w, tuple) and len(w) == 2:
            word, span = w
            if isinstance(span, (tuple, list)) and len(span) == 2:
                s, e = span
                normalized.append({"text": word, "start": float(s), "end": float(e)})
    return normalized


def align_text_with_audio(audio_path, text):
    """
    用编辑后的文本重新跑 forced alignment，返回统一格式的 timestamps。
    失败时返回 []。
    """
    if not audio_path or not text:
        return []
    try:
        raw = generate_word_timestamps(audio_path, text)
        return normalize_timestamps(raw)
    except Exception as e:
        print(f"align_text_with_audio failed: {e}")
        return []    
    
# current_time = 1.0
# timestamps = [{'start':0.42, 'end':0.861, 'text':'ceahkki'},{'start':0.92, 'end':1.061, 'text':'lei'}]
# original_text = "<span>ceahkki lei<span>"
# highlighted_text = update_highlight(current_time, timestamps, original_text)
# print(highlighted_text)



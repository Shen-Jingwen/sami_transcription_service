import numpy as np
from transformers import pipeline
from librosa import resample
import whisperx
import torch
import tempfile
import soundfile as sf
import os
import threading
import multiprocessing
import queue
from utils.forced_alignment import generate_word_timestamps
from custom_diarization import SpeakerDiarizer, DEFAULT_DIARIZATION_MODEL
# from utils.punctuation_restorer import PunctuationRestorer
import logging

logging.basicConfig(
    filename="asr_model_load.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

ASR_MODELS = [ 
    "GetmanY1/wav2vec2-base-sami-cont-pt-22k-finetuned",
    "GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned",
    "shenjingwen/wav2vec2_par20h_ylegie35h",
    "shenjingwen/wav2vec2-ft-50h-24000", #'combine 30h giellagas + 20h parliament'
    "shenjingwen/wav2vec2-par20h-gie30h-3600", #'2 stage: 20h parliament -> 30h giellagas'
    # "GetmanY1/wav2vec2-large-ebranch-sami-18k-finetuned-experimental"
]
DEFAULT_ASR_MODEL = ASR_MODELS[1]


class CustomAutomaticSpeechRecognizer:
    def __init__(self, model_name=ASR_MODELS[1], device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.torch_dtype = torch.float32
        self.model_name = None
        self.pipe = None
        # self.punctuation_restorer = PunctuationRestorer()
        self._stop_event = threading.Event()  # stop process last audio once upload new audios
        # self._stop_event = multiprocessing.Event()
        # self._stop_event = None  

        self._current_process = None
        self._result_queue = queue.Queue()
        
        self.diarizer = None            
        self.use_diarization = False
        self.chunk_length_s = 100  # Length of audio chunks to process
        self.stride_length_s = 5
        self.change_model(model_name or DEFAULT_ASR_MODEL)
        
        self.incremental_mode = False
        self.overlap_ratio = 0.1
        self.incremental_chunk_s = 90
        self.min_overlap_words = 3 #remove overlap only when overlap >= min_overlap_words (set to 1 if no need)
        self.check_overlap_words = 10 #number of words to check overlap for each new chunk
        

    def stop_current_task(self):
        """安全停止当前任务（兼容 Thread 和 Process）"""
        self._stop_event.set()
        if hasattr(self, '_current_process') and self._current_process is not None:
            if hasattr(self._current_process, 'is_alive') and self._current_process.is_alive():
                if hasattr(self._current_process, 'terminate'):
                    # Process 对象
                    self._current_process.terminate()
                    self._current_process.join(timeout=1.0)
                else:
                    # Thread 对象
                    self._current_process.join(timeout=1.0)
                    
    def change_model(self, model_name):
        if model_name not in ASR_MODELS:
            model_name = DEFAULT_ASR_MODEL
            
        if model_name == self.model_name: 
            if model_name == None:
                model_name = self.model_name = DEFAULT_ASR_MODEL
            print(f"Model already set to {model_name}, skipping reload")
            return
        print(f"Changing model from {self.model_name} to {model_name}")
        
        self._stop_event.set()
        if self._current_process and self._current_process.is_alive():
            self._current_process.join(timeout=1.0)
            if self._current_process.is_alive():
                self._current_process.terminate()
            
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()
            
        self.model_name = model_name
        def load_model(target_model):
            try:
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=target_model,
                    device=self.device,
                    return_timestamps='word',
                    torch_dtype=self.torch_dtype
                )
                print(f"Successfully loaded model: {target_model}")
                logging.info(f"Successfully loaded model: {target_model}")
                return pipe
            except Exception as e:
                print(f"Error loading model {target_model}: {str(e)}")
                logging.error(f"Error loading model {target_model}: {str(e)}")
                return None

        self.pipe = load_model(self.model_name) # first try
        if self.pipe is None:
            print(f"Retrying to load original model {self.model_name} (1 more attempt)...")
            self.pipe = load_model(self.model_name)  # first retry
        if self.pipe is None:
            print(f"Retrying to load original model {self.model_name} (2 more attempt)...")
            self.pipe = load_model(self.model_name)  # second retry
        
        if self.pipe is None:
            fallback_model = ASR_MODELS[0] if self.model_name == ASR_MODELS[1] else ASR_MODELS[1]
            print(f"Original model failed after retry, trying fallback model: {fallback_model}")
            self.pipe = load_model(fallback_model)
            self.model_name = fallback_model  
        
        if self.pipe is None:
            final_error = f"Both original model ({self.model_name}) and fallback model ({fallback_model}) failed to load!"
            logging.error(final_error)
            raise RuntimeError(final_error)
        
        self._stop_event.clear()

    def preprocess_audio(self, audio, sr):
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # target_dtype = np.float16 if self.torch_dtype == torch.float16 else np.float32
        # audio = audio.astype(target_dtype)
        # Resample if sampling rate is not 16kHz
        if sr != 16000:
            audio = resample(audio, orig_sr=sr, target_sr=16000)
        return audio
          
    def transcribe(self, stream, result_queue):
        sr, audio = stream
        audio = self.preprocess_audio(audio, sr)
        if stream is None:
            raise ValueError("The 'stream' parameter cannot be None.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            sf.write(temp_audio_file.name, audio, 16000)
            temp_file_name = temp_audio_file.name

        text = ""
        timestamps = []
        
        if not self.incremental_mode:
            for chunk_result in self._chunk_transcribe(audio, sr):
                if self._stop_event.is_set():
                    os.remove(temp_file_name)
                    return
                chunk_text = chunk_result["text"].strip()
                
                if chunk_text:
                    text = chunk_text
                
                if "chunks" in chunk_result:
                    for ts in chunk_result["chunks"]:
                        adjusted_ts = {
                            "text": ts["text"],
                            "start": ts["timestamp"][0] ,
                            "end": ts["timestamp"][1] 
                        }
                        timestamps.append(adjusted_ts)
                    
                # current_text = self.punctuation_restorer.restore(text) ##punctuation, need better solution
                current_text = text
                raw_asr_text = text
                result_queue.put((current_text, stream, current_text, timestamps, raw_asr_text))
        else:
            last_chunk_words = []
            global_offset = 0.0
            last_chunk_end = 0.0
            for chunk_result in self._chunk_transcribe_incremental(audio, sr):
                if self._stop_event.is_set():
                    os.remove(temp_file_name)
                    return
                chunk_text = chunk_result["text"].strip()
                current_words = chunk_text.split()
                overlap = 0
                
                # handle overlap part caused by chunk process
                if last_chunk_words:
                    max_possible_overlap = min(len(last_chunk_words), len(current_words), 10)
                    for i in range(max_possible_overlap, 0, -1):
                        last_part = " ".join(last_chunk_words[-i:]).lower().strip(".,!?")
                        current_part = " ".join(current_words[:i]).lower().strip(".,!?")
                        if current_part.startswith(last_part):
                            overlap = i
                            if not current_part == last_part:
                                # the last word in last chunk is not complete
                                overlap = i-1
                                text = " ".join(text.split()[:-1])
                            break
                        else:
                            # the last word in last chunk is rubbish (not complete and split to 2 words)
                            last_part_2 = " ".join(last_chunk_words[-i-1:-1]).lower().strip(".,!?")
                            if current_part.startswith(last_part_2):
                                overlap = i
                                text = " ".join(text.split()[:-1])
                                break
                            else:
                                # the first word in current chunk is rubbish
                                current_part_2 = " ".join(current_words[1:i+1]).lower().strip(".,!?")
                                if current_part_2.startswith(last_part):
                                    overlap = i-1
                                    current_words = current_words[1:]
                                    text = " ".join(text.split()[:-1])
                                    break
                        
                
                if overlap >= self.min_overlap_words:
                    # remove overlap directly
                    chunk_text = " ".join(current_words[overlap:])
                
                if chunk_text:
                    text += " " + chunk_text
                    last_chunk_words = current_words[-self.check_overlap_words:]
                
                temp_timestamps = []
                if "chunks" in chunk_result:
                    for idx, ts in enumerate(chunk_result["chunks"]):
                        if overlap > 0 and idx < overlap:
                            continue
                        adjusted_ts = {
                            "text": ts["text"],
                            "start": ts["timestamp"][0] + global_offset,
                            "end": ts["timestamp"][1] + global_offset
                        }
                        temp_timestamps.append(adjusted_ts)
                    
                    if chunk_result["chunks"]:
                        last_chunk_end = chunk_result["chunks"][-1]["timestamp"][1]
                        global_offset += last_chunk_end - (self.chunk_length_s * self.overlap_ratio)
                
                # current_text = self.punctuation_restorer.restore(text) ##punctuation, need better solution
                current_text = text
                raw_asr_text = text
                result_queue.put((current_text, stream, current_text, temp_timestamps, raw_asr_text))  
    
        try:
            final_timestamps = generate_word_timestamps(temp_file_name, text)
        except Exception as e:
            print(f"Error generating timestamps: {e}")
            final_timestamps = timestamps
            
        normalized = []
        for w in final_timestamps:
            if isinstance(w, dict):
                normalized.append(w)
            elif isinstance(w, tuple) and len(w) == 2:
                word, (s, e) = w
                normalized.append({"text": word, "start": s, "end": e})
        final_timestamps = normalized

        # final_text = self.punctuation_restorer.restore(text) ##punctuation, need better solution
        final_text = text
        print(f"111111111111111111111: {final_text}")
        
        # ===== diarization =====
        if getattr(self, "use_diarization", False) and self.diarizer is not None:
            # ASR 模型移到 CPU，腾 GPU
            if self.pipe is not None and hasattr(self.pipe, "model"):
                try:
                    self.pipe.model.to("cpu")
                except Exception:
                    pass
            torch.cuda.empty_cache()

            try:
                speaker_turns = self.diarizer.diarize(temp_file_name)
            except Exception as e:
                print(f"Diarization failed: {e}")
                speaker_turns = []

            # diarizer 移回 CPU
            self.diarizer.to("cpu")
            torch.cuda.empty_cache()

            # 词级对齐
            for w in final_timestamps:
                if isinstance(w, dict) and "start" in w and "end" in w:
                    w["speaker"] = SpeakerDiarizer.assign_speaker(w["start"], w["end"], speaker_turns)
            segments = SpeakerDiarizer.merge_words_by_speaker(final_timestamps)
            final_text = SpeakerDiarizer.format_segments(segments)

            # ASR 模型搬回 GPU
            if self.pipe is not None and hasattr(self.pipe, "model"):
                try:
                    self.pipe.model.to(self.device)
                except Exception:
                    pass


        print(f"111111111111111111111: {final_text}")
        os.remove(temp_file_name)
        result_queue.put((final_text, stream, final_text, final_timestamps, raw_asr_text))
            

    def _chunk_transcribe_incremental(self, audio, sr):
        """手动分块 + overlap，逐 chunk yield，WER 会受影响"""
        audio_list = audio.tolist()
        chunk_size = int(self.chunk_length_s * sr)
        incremental_chunk_size = int(self.incremental_chunk_s * sr)
        

        for i in range(0, len(audio_list), incremental_chunk_size):
            if self._stop_event.is_set():
                return
            start = max(0, i - int(self.overlap_ratio * incremental_chunk_size))
            end = min(len(audio_list), start + chunk_size)
            chunk = audio_list[start:end]

            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

            chunk_np = np.array(chunk, dtype=np.float32)
            output = self.pipe(chunk_np, return_timestamps='word')
            print(f"_chunk_transcribe_incremental {i // incremental_chunk_size}: {output}")
            yield output

              
    def _chunk_transcribe(self, audio, sr):
        """Generator that yields partial transcription results"""
        
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.ascontiguousarray(audio)

        output = self.pipe(
            audio,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
            return_timestamps='word'
        )
        print(f"""_chunk_transcribe={output}""")
        yield output
        
    
    def transcribe_with_diarization_file(self,filepath: str):
        self._stop_event.set()
        if self._current_process and self._current_process.is_alive():
            self._current_process.join(timeout=0.5)
            if self._current_process.is_alive():
                self._current_process.terminate()
        self._stop_event.clear()
        
        audio = whisperx.load_audio(filepath, 16000)
        # print(f"audio: {audio}  filepath= {filepath}.")
        if audio is None:
            print(f"Error: Failed to load audio from {filepath}.")  
        # stop_event = threading.Event()   
        
        result_queue = queue.Queue()
        # result_queue = multiprocessing.Queue()
        self._current_process = threading.Thread (
            target=self.transcribe,
            args=((16000, audio), result_queue)
        )
        self._current_process.start()
        
        while not self._stop_event.is_set():
            try:
                yield result_queue.get(timeout=0.1)
            except queue.Empty:
                if not self._current_process.is_alive():
                    break
                    
        # return self.transcribe(
            # (16000, audio), None, "", False
        # )
        
        

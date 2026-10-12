import numpy as np
import time
import whisperx
from scipy.signal import resample
import time
import os
import time
import torch

class WhisperAutomaticSpeechRecognizer:
    # device = "cuda"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8"  # change to if more gpu memory available
    batch_size = 4 
    model = whisperx.load_model(
        "medium", device, language="en", compute_type=compute_type
        ,
        asr_options={
            "max_new_tokens": 1024,
            "clip_timestamps": True,
            "hallucination_silence_threshold": 0.2,
        }
    )
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=os.environ.get('HF_TOKEN'), device=device
    )
    existing_speaker = None

    @staticmethod
    def downsample_audio_scipy(audio: np.ndarray, original_rate, target_rate=16000):
        if original_rate == target_rate:
            return audio

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Check if audio has one channel
        if len(audio.shape) != 1:
            raise ValueError("Input audio must have only one channel.")

        # Calculate the number of samples in the downsampled audio
        num_samples = int(len(audio) * target_rate / original_rate)
        downsampled_audio = resample(audio, num_samples)

        return downsampled_audio

    @staticmethod
    def transcribe_with_diarization_file(filepath: str):
        audio = whisperx.load_audio(filepath, 16000)
        return WhisperAutomaticSpeechRecognizer.transcribe_with_diarization(
            (16000, audio), None, "", False
        )

    @staticmethod
    def transcribe_with_diarization(
        stream, full_stream, full_transcript, streaming=True
    ):
        start_time = time.time()
        sr, y = stream
        if streaming:
            sr, y = stream
            y = WhisperAutomaticSpeechRecognizer.downsample_audio_scipy(y, sr)
            y = y.astype(np.float32)
            y /= 32768.0

        if full_transcript is None:
            full_transcript = ""
        transcribe_result = WhisperAutomaticSpeechRecognizer.model.transcribe(
            y, batch_size=WhisperAutomaticSpeechRecognizer.batch_size
        )
        diarize_segments = WhisperAutomaticSpeechRecognizer.diarize_model(y)

        diarize_result = whisperx.assign_word_speakers(
            diarize_segments, transcribe_result
        )

        new_transcript = ""
        for segment in diarize_result["segments"]:
            current_speaker = ""
            default_first_speaker = "SPEAKER_00"
            try:
                current_speaker = segment["speaker"]
            except KeyError:
                current_speaker = default_first_speaker
            if WhisperAutomaticSpeechRecognizer.existing_speaker == None:
                try:
                    WhisperAutomaticSpeechRecognizer.existing_speaker = current_speaker
                except KeyError:
                    WhisperAutomaticSpeechRecognizer.existing_speaker = default_first_speaker
                new_transcript += f"\n {WhisperAutomaticSpeechRecognizer.existing_speaker}  - "
            if current_speaker != WhisperAutomaticSpeechRecognizer.existing_speaker and current_speaker is not default_first_speaker:
                WhisperAutomaticSpeechRecognizer.existing_speaker = current_speaker
                new_transcript += f"\n {WhisperAutomaticSpeechRecognizer.existing_speaker}  - "
            new_transcript = new_transcript + segment["text"]
        full_transcript = full_transcript + new_transcript
        end_time = time.time()
        if streaming:
            time.sleep(5 - (end_time - start_time))
        return full_transcript, stream, full_transcript

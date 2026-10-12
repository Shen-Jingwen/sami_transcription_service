from custom_asr import CustomAutomaticSpeechRecognizer, ASR_MODELS, DEFAULT_ASR_MODEL
from custom_diarization import SpeakerDiarizer, DIARIZATION_MODELS, DEFAULT_DIARIZATION_MODEL

import torch
torch.cuda.empty_cache()


asr = CustomAutomaticSpeechRecognizer(DEFAULT_ASR_MODEL)
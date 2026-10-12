
## add wen 20261012

Implemented a Northern Sami speech transcription service with speaker diarization.
Frontend built with Gradio Blocks, provides audio upload, ASR model selection,
speaker diarization, transcript display, editing, and saving.

##Service Link
http://195.148.21.110:7860/

![Frontend screenshot](sami_transcription_service/readme/frontend.png)

Key features:

- Offline speaker-attributed transcription with word-level timestamp alignment and audio-text highlighting.
- Users can switch between different ASR models and diarization models at runtime.
		Transcription and speaker attribution pipeline:
		  1. The ASR model produces the transcription. (custom_asr.py)
		  2. Forced alignment produces word-level timestamps based on ASR transcription. (utils/forced_alignment.py, utils/audio_transcript_utils.py)
		  3. The diarization model produces speaker turns - speaker + start/end timestamps. (custom_diarization.py)
		  4. Word-level timestamps and speaker turns are aligned to assign speaker label on ASR transcription. (custom_asr.py)
		  5. Final output (speaker-attributed transcript) is combined as `*start-end* speaker: transcription`. (custom_diarization.py)
		ASR and diarization are logically independent, but executed sequentially in 
	the code (ASR → forced alignment → diarization → merge), because both models need 
	GPU memory and cannot run in parallel on a single GPU. Frontend (app.py): ASR outputs first 
	for early feedback, then replaced by speaker-attributed transcript once diarization finishes.
		

- Long audio is processed in chunks by the ASR pipeline - 100s chunks with 5s overlap. (custom_asr.py)
		Handled by HuggingFace ASR pipeline — the pipeline merges chunk results
	with proper stride handling and returns the final transcription as a whole.
	Chunking happens entirely inside the pipeline for long-audio ASR processing.
	It does not change how results are returned; the pipeline does not output
	chunk by chunk.
	

- Optional incremental ASR output mode, with a WER trade-off warning. (app.py)
		This only affects how ASR results are segmented and streamed to the frontend. 
		The HuggingFace ASR pipeline does not natively support chunk-by-chunk output, 
	so in this mode the audio is manually split into chunks and each chunk is
	transcribed separately. Overlap between chunks is handled in code, which slightly increases WER.
	  
- Transcript editing: edit text, re-align with forced alignment, and save the result. (app.py)



##Working Directory
/sami_transcription_service

## Dependencies & Conda Environment
/sami_transcription_service/requirements.txt: pip-installed Python dependencies.
/sami_transcription_service/environment.yml: Update hugging face token here. `conda env create -f environment.yml`

##activate environment (command)
`conda activate sami_asr`

##start/restart the service (command)
`./start.sh --daemon`

##Logging and Debugging (command)
`tail -f log.txt`




## ASR
custom_asr.py
- https://huggingface.co/GetmanY1/wav2vec2-base-sami-cont-pt-22k-finetuned
- https://huggingface.co/GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned
- Additional fine-tuned Sámi models under `shenjingwen/` on Hugging Face.

## Diarization
custom_diarization.py
- `shenjingwen/sami-diarizen-ft-006`
- `shenjingwen/sami-diarizen-ft-006L`
- `shenjingwen/sami-diarizen-ftl12.7h-ft-0.28b`


# References
- https://huggingface.co/shenjingwen
- https://github.com/SubtleParesh/live-speech-reference-search
- https://github.com/MahmoudAshraf97/ctc-forced-aligner/tree/main
- https://github.com/BUTSpeechFIT/DiariZen
- https://huggingface.co/GetmanY1/wav2vec2-base-sami-cont-pt-22k-finetuned
- https://huggingface.co/GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned

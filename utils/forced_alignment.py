import torch
torch.cuda.empty_cache()
import json

from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)


def generate_word_timestamps(audio_path, transcript_text, language="sme", device="cuda" if torch.cuda.is_available() else "cpu", batch_size=16):
    if not audio_path or not transcript_text:
        return []
        
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float32,
    )

    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)


    # with open(text_path, "r") as f:
        # lines = f.readlines()
    # text = "".join(line for line in lines).replace("\n", " ").strip()
    text = transcript_text.strip()

    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    # print (f"word_timestamps={json.dump(word_timestamps, f, ensure_ascii=False, indent=2)}")

    # output_json_path = "word_timestamps.json"  
    # with open(output_json_path, "w", encoding="utf-8") as f:
        # json.dump(word_timestamps, f, ensure_ascii=False, indent=2)
    return word_timestamps


##test
audio_path = "audios/80_99_north_s_00_003.wav"
text_path = "audios/003.txt"
language = "sme" # Northern Sami ISO-639-3 Language code 
# word_timestamps = generate_word_timestamps(audio_path, text_path, language)


# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import os
import torch
import gradio as gr
# from vector_search import HybridVectorSearch
# from whisper_asr import WhisperAutomaticSpeechRecognizer
from custom_asr import CustomAutomaticSpeechRecognizer, ASR_MODELS, DEFAULT_ASR_MODEL
from custom_diarization import SpeakerDiarizer, DIARIZATION_MODELS, DEFAULT_DIARIZATION_MODEL

from asr_instance import asr
import time
import tempfile
import json, re
import soundfile as sf
import shutil
from utils.audio_transcript_utils import *


CSS="""
    .custom-btn {
        # width: 60px !important;    
        # height: 40px !important; 
        display: flex !important;          
        font-size: 0.9rem !important;
        border-radius: 0.375rem !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .file-btn {  
        display: flex !important;  
        border-radius: 0.375rem !important;
    }
    .transcriptBox {
        position: relative;
    }
    .btn-container {
        position: absolute;
        bottom: 0.5rem;
        right: 0.5rem;
        display: flex;
        gap: 0.25rem;
    }
    .gr-textbox { overflow-y: auto; }
    .transcript-stream {
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0.5; }
        to { opacity: 1; }
    }
    
    .highlight {
        background-color: #e0f2fe !important;
        border-radius: 3px !important;
        padding: 1px 3px;
        # transition: background-color 0.2s ease;
    }
    .transcript-container {
        overflow-y: auto;
        height: 100%;
        padding: 10px;
        # padding-right: 1rem;
    }
    .transcript-line {
        # padding: 4px;
        # cursor: pointer;
        min-height: 1.2em;
        line-height: 1.2;
    }
    .transcript-line.highlight {
        background-color: #e0f2fe;
        transition: background-color 0.3s ease;
    }
    .edit-container {
        position: relative;
        margin-top: 1rem;
    }
    .edit-textarea {
        min-height: 100px;
    }
    .edit-controls {
        margin-top: 0.5rem;
        display: flex;
        justify-content: flex-end;
        gap: 0.5rem;
    }
    .incremental-warning label {
        color: #d32f2f !important;
    }
"""


def strip_speaker_tags(text):
    """
    去掉 *00:00:00.36-00:00:12.70* S1: 这类标签，只留纯文本。
    """
    if not isinstance(text, str):
        return text
    # 去掉 *时间-时间* S数字: 或 S数字: 标签
    text = re.sub(r'\*[\d:\.\-]+\*\s*S\d+:\s*', '', text)
    # 去掉纯 S数字: 标签
    text = re.sub(r'\bS\d+:\s*', '', text)
    return text.strip()


def clean_text_for_alignment(text):
    """
    forced alignment 前的文本清洗：去 HTML、去说话人标签、去多余空白。
    """
    if not isinstance(text, str):
        return text
    # 去 HTML
    text = remove_tags(text)
    # 去说话人标签
    text = strip_speaker_tags(text)
    # 换行变空格
    text = text.replace("\n", " ")
    # 多空格变单空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def retranscribe_on_model_change(model_name, audio_input_value, mode, enable_diarization, diar_model, incremental_mode):
    print(f'''model_name  {model_name}, {asr.model_name}''')
    if audio_input_value is None:
        print(model_name) 
        return gr.skip(), gr.skip(), gr.skip(), model_name, gr.skip(), gr.skip()
    asr.incremental_mode = incremental_mode 
    if model_name != asr.model_name:
        asr._stop_event.set()
        if asr._current_process and asr._current_process.is_alive():
            asr._current_process.join(timeout=0.5)
            if asr._current_process.is_alive():
                asr._current_process.terminate()
        asr.change_model(model_name)
        asr._stop_event.clear()
    if enable_diarization:
        if asr.diarizer is None:
            asr.diarizer = SpeakerDiarizer(model_name=diar_model)
        elif asr.diarizer.model_name != diar_model:
            asr.diarizer.unload()
            asr.diarizer.model_name = diar_model
            asr.diarizer.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            asr.diarizer.load()
        asr.use_diarization = True
    else:
        asr.use_diarization = False
        
    
    if mode == "live":
        if audio_input_value is not None:
            sr, audio = audio_input_value
            audio = asr.preprocess_audio(audio, sr)
            output = asr.pipe(audio, chunk_length_s=30, stride_length_s=5)['text']
            return output, audio_input_value, output
        return "", None, ""
    elif mode == "offline":
        cache = {"audio_path": audio_input_value, "text": '', "raw_asr_text": '', "timestamps": []}
        if audio_input_value is not None:
            
            for transcript_text, timestamps, raw_asr_text in process_audio_with_timestamps(audio_input_value):
                if asr._stop_event.is_set():
                    break
                duration = sf.info(audio_input_value).duration
                if enable_diarization:
                    html_text = transcript_text.replace("\n\n", "<br><br>").replace("\n", "<br>")
                else:
                    html_text = f'<span>{transcript_text}</span>'
                cache = {"audio_path": audio_input_value, "text": transcript_text, "raw_asr_text": raw_asr_text, "timestamps": timestamps}
                yield html_text, transcript_text, timestamps, duration, save_file_transcript(html_text),cache
            if not asr._stop_event.is_set():
                yield html_text, transcript_text, timestamps, duration, save_file_transcript(html_text),cache
        return "",'', [], 0, None, cache

def toggle_edit_mode(edit_mode, current_text):
    new_mode = not edit_mode
    if isinstance(current_text, str):
        plain_text = current_text.replace("<br><br>", "\n\n").replace("<br>", "\n")
        plain_text = remove_tags(plain_text)
    else:
        plain_text = current_text
    html_text = plain_text.replace("\n\n", "<br><br>").replace("\n", "<br>")    
    return (
        new_mode, 
        gr.HTML(visible=not new_mode, value=f'<span>{html_text}</span>'),
        gr.Textbox(visible=new_mode, interactive=True, lines=40, value=plain_text),  
        gr.Button(visible=not new_mode), 
        gr.Button(visible=new_mode, variant="secondary", value="Cancel"),  
        plain_text
    )

def edit_done(current_time, timestamps, transcript_text, edit_mode, cache):
    text_to_save = transcript_text if isinstance(transcript_text, str) else str(transcript_text)
    html_text = text_to_save.replace("\n\n", "<br><br>").replace("\n", "<br>")
    html_text = f'<span>{html_text}</span>'
    
    print(f"edit_done= text_to_save {text_to_save}")
    print(f"edit_done= html_text {html_text}")
    print(f"edit_done= gr.HTML {gr.HTML(value=html_text, visible=True)}")
    print("transcript_text:", repr(transcript_text[:300]) if isinstance(transcript_text, str) else transcript_text)
    if cache and cache.get("text") and clean_text_for_alignment(text_to_save) == clean_text_for_alignment(cache["text"]):
        return (
            False,                          # edit_mode 退出
            gr.update(),                    # transcript_html 不动
            gr.Textbox(visible=False, value=text_to_save),   # transcript_edit 隐藏
            gr.Button(visible=True),        # edit_button 显示
            gr.Button(visible=False),       # cancel_edit_btn 隐藏
            gr.update(),                    # original_text 不动
            gr.update(),                    # edit_text_state 不动
            gr.update(),                    # asr_cache 不动
            gr.update(),                    # audio_timestamps 不动
            gr.update(),                    # saved_transcript_file 不动
        )
        
    asr_plain_text = clean_text_for_alignment(text_to_save)
    new_timestamps = align_text_with_audio(cache["audio_path"], asr_plain_text) 
    if new_timestamps:
        cache["text"] = asr_plain_text
        cache["raw_asr_text"] = asr_plain_text
        cache["timestamps"] = new_timestamps
            
    return (
        False,
        gr.HTML(value=html_text, visible=True),
        gr.Textbox(visible=False, value=text_to_save),
        gr.Button(visible=True),   # show Edit
        gr.Button(visible=False),  # hide Cancel
        text_to_save,
        text_to_save,
        cache,
        new_timestamps,
        save_transcript(text_to_save)
    )



def cancel_edit(edit_mode, original_text):
    plain_text = original_text if isinstance(original_text, str) else str(original_text)
    html_text = plain_text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return (
        False,  
        gr.HTML(value=f'<span>{html_text}</span>', visible=True),  
        gr.Textbox(visible=False, value=plain_text),  
        gr.Button(visible=True), 
        gr.Button(visible=False),  
        plain_text 
    )
 
def remove_tags(text):
    tag_pattern = re.compile(r'<[^>]+>')
    return tag_pattern.sub('', text)
 
# def save_edited_transcript(transcript_text, edit_mode, original_text):
    # text_to_save = transcript_text if isinstance(transcript_text, str) else str(transcript_text)
    # html_text = text_to_save.replace("\n\n", "<br><br>").replace("\n", "<br>")
    # html_text = f'<span>{html_text}</span>'
    
    # saved_file = save_transcript(text_to_save)
    # if not edit_mode:
        # return (
            # not edit_mode,  
            # gr.HTML(value=html_text, visible=True),  
            # gr.Textbox(visible=False, value=text_to_save),  
            # gr.Button(visible=True),  # show Edit button
            # gr.Button(visible=False),  # hide Save button
            # gr.Button(visible=False),  # hide Cancel button
            # text_to_save,  
            # text_to_save, 
            # saved_file  
        # ) 
    # else:
        # return ( 
            # edit_mode,  
            # gr.HTML(value=html_text, visible=False),  
            # gr.Textbox(visible=True, value=text_to_save),  
            # gr.Button(visible=True),  # show Edit button
            # gr.Button(visible=False),  # hide Save button
            # gr.Button(visible=False),  # hide Cancel button
            # text_to_save,  
            # text_to_save, 
            # saved_file  
        # ) 

def save_file_transcript(original_text):
    text_to_save = original_text if isinstance(original_text, str) else str(original_text)
    text_to_save = text_to_save.replace("<br><br>", "\n\n").replace("<br>", "\n")
    saved_file = save_transcript(text_to_save)
    return saved_file  

def offline_audio_handler(audio_file, enable_diarization, diar_model, incremental_mode):
    if not audio_file: 
        return "", "", "", [], None, 0
    # asr._stop_event.set()
    # if asr._current_process and asr._current_process.is_alive():
        # asr._current_process.join(timeout=1.0)
        # if asr._current_process.is_alive():
            # asr._current_process.terminate()
    
    current_audio.value = audio_file
    duration = sf.info(audio_file).duration if audio_file else 0
    asr._stop_event.clear()
    asr.incremental_mode = incremental_mode
    if enable_diarization:
        if asr.diarizer is None:
            asr.diarizer = SpeakerDiarizer(model_name=diar_model)
        elif asr.diarizer.model_name != diar_model:
            asr.diarizer.unload()
            asr.diarizer.model_name = diar_model
            asr.diarizer.load()
        asr.use_diarization = True
    else:
        asr.use_diarization = False
    
    html_text = ""
    timestamps = []
    cache = {
        "audio_path": audio_file, 
        "text": '', 
        "raw_asr_text": '',
        "timestamps": timestamps
    }
    for transcript_text, timestamps, raw_asr_text in process_audio_with_timestamps(audio_file):
        if asr._stop_event.is_set():  # stop once recieve new audios
            break
        if enable_diarization:
            html_text = transcript_text.replace("\n\n", "<br><br>").replace("\n", "<br>")
        else:
            words = transcript_text.split()
            html_text = ' '.join(f'<span class="word">{w}</span>' for w in words)
        cache = {"audio_path": audio_file, "text": transcript_text, "raw_asr_text": raw_asr_text, "timestamps": timestamps}    
        # print(f"transcript_text: {repr(transcript_text[:200])}")
        # print(f"html_text: {repr(html_text[:200])}")
        yield html_text, transcript_text, transcript_text, timestamps, audio_file, duration, save_file_transcript(html_text),cache

    if not asr._stop_event.is_set():
        yield html_text, html_text, html_text, timestamps, audio_file, duration, save_file_transcript(html_text),cache


def rediarize_with_cache(audio_file, enable_diarization, diar_model, cache, prev_enabled, rediarize=False):
    if not audio_file or not cache:
        return "", [], 0, "", None
    if cache.get("audio_path") != audio_file:
        # 缓存不是这个音频，走完整流程
        return "", [], 0, "", None
    if not enable_diarization and prev_enabled:
        raw_asr_text = cache.get("raw_asr_text", cache["text"])
        html_text = raw_asr_text
        for w in cache["timestamps"]:
            if isinstance(w, dict):
                w.pop("speaker", None)
        return html_text, cache["timestamps"], sf.info(audio_file).duration, raw_asr_text, save_file_transcript(html_text)
    
    if not rediarize and not (enable_diarization and not prev_enabled):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
    # 配置 diarizer
    if asr.diarizer is None:
        asr.diarizer = SpeakerDiarizer(model_name=diar_model)
    elif asr.diarizer.model_name != diar_model:
        asr.diarizer.unload()
        asr.diarizer.model_name = diar_model
        asr.diarizer.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        asr.diarizer.load()
        
    asr.use_diarization = True
    torch.cuda.empty_cache()

    # 重跑 diarization
    try:
        speaker_turns = asr.diarizer.diarize(audio_file)
    except Exception as e:
        print(f"Diarization failed: {e}")
        speaker_turns = []


    # 对齐
    timestamps = cache["timestamps"]
    for w in timestamps:
        if isinstance(w, dict) and "start" in w and "end" in w:
            w["speaker"] = SpeakerDiarizer.assign_speaker(w["start"], w["end"], speaker_turns)
    segments = SpeakerDiarizer.merge_words_by_speaker(timestamps)
    final_text = SpeakerDiarizer.format_segments(segments)

    # ASR 模型搬回 GPU
    if asr.pipe is not None and hasattr(asr.pipe, "model"):
        try:
            asr.pipe.model.to(asr.device)
        except Exception:
            pass

    html_text = final_text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    duration = sf.info(audio_file).duration
    print("timestamps len:", len(timestamps))
    print("speaker labels:", set(w.get("speaker") for w in timestamps if isinstance(w, dict)))
    return html_text, timestamps, duration, final_text, save_file_transcript(final_text)

def update_text_highlight(current_time, timestamps, transcript_text, transcript_html, edit_mode, enable_diarization):
    # print(f"audio_time value: {type(current_time)} {current_time}") 
    print("transcript_text:", repr(transcript_text[:200]) if isinstance(transcript_text, str) else transcript_text)
    print("transcript_html:", repr(transcript_html[:200]) if isinstance(transcript_html, str) else transcript_html)
    if isinstance(current_time, str): 
        if ":" in current_time:
            current_time=time_str_to_seconds(current_time)
            print(f"audio_time value: {type(current_time)} {current_time}")
    # if not timestamps or current_time is None or current_time == 0:
        # return transcript_html if isinstance(transcript_html, str) else ""
    if not timestamps or current_time is None:
        return transcript_text
    if not enable_diarization:    
        plain = remove_tags(transcript_text) if isinstance(transcript_text, str) else transcript_text
        highlighted_text = update_highlight(current_time, timestamps, plain)
    else:
        highlighted_text = update_highlight(current_time, timestamps, transcript_text)
    highlighted_text = highlighted_text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    print(f"update_text_highlight={highlighted_text}")
    # return highlighted_text, remove_tags(transcript_text)
    return highlighted_text
        
# transcript box and edit button
def create_transcript_ui():
    with gr.Column(scale=25):
        # html for display
        transcript_html = gr.HTML(
            value="Transcript will appear here",
            label="Transcript",
            show_label=True,
            elem_id="transcript_html"
        )
        # Textbox for edit mode 
        transcript_edit = gr.Textbox(
            lines=40,
            placeholder="Transcript",
            max_lines=40,
            label="Transcript (Edit Mode)",
            show_label=True,
            visible=False,
            interactive=True,
            elem_id="transcript_textbox"
        )
    with gr.Column(scale=1):
        edit_button = gr.Button("Edit Transcript", elem_classes="custom-btn", elem_id="edit")
        edit_done_button = gr.Button("Save/Exit Edit", elem_classes="custom-btn", elem_id="edit_done")
        cancel_edit_btn = gr.Button("Cancel", visible=False, elem_classes="custom-btn", variant="secondary")
        saved_transcript_file = gr.File(label="saved_transcript", visible=True, file_count="single", type="filepath", elem_classes="file-btn")
        # save_transcript_btn = gr.Button("Save Current Transcript", visible=False, elem_classes="custom-btn") 
        # current_transcript_file = gr.File(label="new_saved_transcript", visible=False, file_count="single", type="filepath", elem_classes="file-btn")
    return transcript_html, transcript_edit, edit_button, edit_done_button, cancel_edit_btn, saved_transcript_file

# edit button event
def bind_edit_buttons(edit_button, edit_done_button, cancel_edit_btn, edit_mode, transcript_html, transcript_edit, original_text, current_time=None, timestamps=None):
    edit_button.click(
        toggle_edit_mode,
        inputs=[edit_mode, original_text],
        outputs=[edit_mode, transcript_html, transcript_edit, edit_button, cancel_edit_btn, original_text],
        show_progress=False
    )
    edit_done_button.click(
        edit_done,
        inputs=[played_duration, audio_timestamps, edit_text_state, edit_mode, asr_cache],
        outputs=[edit_mode, transcript_html, transcript_edit, edit_button, cancel_edit_btn, original_text, edit_text_state, asr_cache, audio_timestamps, saved_transcript_file],
        show_progress="minimal"
    )
    cancel_edit_btn.click(
        cancel_edit,
        inputs=[edit_mode, original_text],
        outputs=[edit_mode, transcript_html, transcript_edit, edit_button, cancel_edit_btn, original_text],
        show_progress=False
    )
    
  

with gr.Blocks(css=CSS) as demo:
    full_stream = gr.State()
    transcript = gr.State(value="")
    chats = gr.State(value=[])
    edit_mode = gr.State(value=False)
    just_saved = gr.State(False)
    edit_text_state = gr.State("")
    original_text = gr.State(value="")
    
    audio_timestamps = gr.State([])
    current_audio = gr.State(None)
    asr_cache = gr.State(None)
    prev_diar_enabled = gr.State(False)
    with gr.Column():
        model_selector = gr.Dropdown(
            choices=ASR_MODELS,
            value=DEFAULT_ASR_MODEL,
            label="Select ASR Model"
        ) 
        incremental_mode_checkbox = gr.Checkbox(
            value=False,
            label="Incremental ASR output (may increase WER)",
            elem_classes="incremental-warning"
        )
    enable_diarization = gr.Checkbox(
        value=False,
        label="Enable Speaker Diarization"
    )
    diarization_model_selector = gr.Dropdown(
        choices=DIARIZATION_MODELS,
        value=DEFAULT_DIARIZATION_MODEL,
        label="Select Diarization Model",
        visible=False
    )
    '''
    with gr.Tab("Live Mode"):
        xxxx
    '''     
    
    with gr.Tab("Offline Mode"):
        with gr.Row(variant="panel", equal_height=True):
            audio_input = gr.Audio(
                sources=["upload"], 
                type="filepath",
                interactive=True,
                elem_id="offline_audio"
            )
            audio_duration = gr.Number(value=0.0, visible=False, elem_id="audio_duration")
            played_duration = gr.Number(value=5.0, visible=False, elem_id="played_duration")
            audio_timestamps = gr.State([])
            
        with gr.Row(variant="panel", equal_height=True):
            transcript_html, transcript_edit, edit_button, edit_done_button, cancel_edit_btn, saved_transcript_file = create_transcript_ui()
        audio_input.clear(
            offline_audio_handler,
            [audio_input, enable_diarization, diarization_model_selector, incremental_mode_checkbox],
            [transcript_html, transcript_edit, original_text, audio_timestamps, current_audio, audio_duration, saved_transcript_file],
            show_progress=True,
            queue=True
        )
        # audio_input.upload(
            # offline_audio_handler,
            # [audio_input, enable_diarization, diarization_model_selector, incremental_mode_checkbox],
            # [transcript_html, transcript_edit, original_text, audio_timestamps, current_audio, audio_duration, saved_transcript_file],
            # show_progress=True,
            # queue=True,  # Important for generators
        # )
        audio_input.change(
            offline_audio_handler,
            [audio_input, enable_diarization, diarization_model_selector, incremental_mode_checkbox],
            [transcript_html, transcript_edit, original_text, audio_timestamps, current_audio, audio_duration, saved_transcript_file, asr_cache],
            show_progress=True,
            queue=True,  # Important for generators
        )
        
        
        enable_diarization.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[enable_diarization],
            outputs=[diarization_model_selector],
            show_progress=False,
        ).then(
            rediarize_with_cache,
            inputs=[audio_input, enable_diarization, diarization_model_selector, asr_cache, prev_diar_enabled],
            outputs=[transcript_html, audio_timestamps, audio_duration, original_text, saved_transcript_file],
            show_progress=True
        ).then(
            lambda enabled: enabled,
            inputs=[enable_diarization],
            outputs=[prev_diar_enabled],
            show_progress=False
        )
        
        transcript_edit.change(
            lambda x: x,
            inputs=[transcript_edit],
            outputs=[edit_text_state],
            show_progress=False,
        )
        bind_edit_buttons(edit_button, edit_done_button, cancel_edit_btn, edit_mode, transcript_html, transcript_edit, original_text, played_duration, audio_timestamps)
        
        
        ### start alignment and highlight update ###
        hidden_trigger = gr.Button(visible=False, elem_id="hidden_trigger")
        hidden_trigger.click(
            time_str_to_seconds,  
            inputs=[played_duration],
            outputs=[played_duration],
            js="""() => {
                return [document.getElementById('time').textContent];
            }"""
        )      
        hidden_trigger.click(
            update_text_highlight,
            inputs=[played_duration, audio_timestamps, original_text, transcript_html, edit_mode, enable_diarization],
            outputs=[],
            # js="""
                # function(played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode, enable_diarization) {
                    # const transcriptElement = document.getElementById('transcript_html');
                    # const transcriptElement_edit = document.getElementById('transcript_textbox');
                    # //if (transcriptElement) {
                    # //    transcriptElement.innerHTML = transcript_html;
                    # //}
                    # return [played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode, enable_diarization];
                # }
                # """    
            js="""
                function(played_duration, audio_timestamps, transcript_text, transcript_html, edit_mode, enable_diarization) {
                    const audioElement = document.getElementById('offline_audio');
                    if (audioElement.dataset.intervalId) {
                        clearInterval(parseInt(audioElement.dataset.intervalId));
                    }
                    const intervalId = setInterval(() => {
                        const timeElement = document.getElementById('time');
                        if (timeElement) {
                            const played_duration = timeElement.textContent;
                            document.getElementById("hidden_trigger").click();
                        }
                    }, 1000);
                    audioElement.dataset.intervalId = intervalId;
                    return [played_duration, audio_timestamps, transcript_text, transcript_html, edit_mode, enable_diarization];
                }
            """
        ) 
        audio_input.play(
            time_str_to_seconds,  
            inputs=[played_duration],
            outputs=[played_duration],
            js="""
                function() {
                    const audioElement = document.getElementById('offline_audio');
                    if (audioElement.dataset.intervalId) {
                        clearInterval(parseInt(audioElement.dataset.intervalId));
                        console.log('clean:', audioElement.dataset.intervalId);
                    }
                    const intervalId = setInterval(() => {
                        const timeElement = document.getElementById('time');
                        if (timeElement) {
                            const played_duration = timeElement.textContent;
                            console.log('time:', played_duration);
                            
                            document.getElementById("hidden_trigger").click();
                            return [played_duration];  
                        }
                        return ["0:00"];  // default
                    }, 1000);  

                    document.getElementById('offline_audio').dataset.intervalId = intervalId;

                    // initial
                    const timeElement = document.getElementById('time');
                    return timeElement ? [timeElement.textContent] : ["0:00"];
                }
            """    
        )
        
        audio_input.play(
            update_text_highlight,
            inputs=[played_duration, audio_timestamps, original_text, transcript_html, edit_mode, enable_diarization],
            show_progress="minimal",
            outputs=[],
            # js="""
                # function(played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode, enable_diarization) {
                    # const transcriptElement = document.getElementById('transcript_html');
                    # const transcriptElement_edit = document.getElementById('transcript_textbox');
                    # //if (transcriptElement) { 
                    # //    transcriptElement.innerHTML = transcript_html;
                    # //}
                    # return [played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode, enable_diarization];
                # }
                # """  
            js="""
                function(played_duration, audio_timestamps, transcript_text, transcript_html, edit_mode, enable_diarization) {
                    const audioElement = document.getElementById('offline_audio');
                    if (audioElement.dataset.intervalId) {
                        clearInterval(parseInt(audioElement.dataset.intervalId));
                    }
                    const intervalId = setInterval(() => {
                        const timeElement = document.getElementById('time');
                        if (timeElement) {
                            const played_duration = timeElement.textContent;
                            document.getElementById("hidden_trigger").click();
                        }
                    }, 1000);
                    audioElement.dataset.intervalId = intervalId;
                    return [played_duration, audio_timestamps, transcript_text, transcript_html, edit_mode, enable_diarization];
                }
            """
            
        ) 

        played_duration.change(
            update_text_highlight,
            inputs=[played_duration, audio_timestamps, original_text, transcript_html, edit_mode, enable_diarization],
            outputs=[transcript_html]
        )
        
        audio_input.pause(
            None,
            inputs=None,
            outputs=None,
            js="""
                function() {
                    const audioElement = document.getElementById('offline_audio');
                    if (audioElement.dataset.intervalId) {
                        clearInterval(parseInt(audioElement.dataset.intervalId));
                    }
                    return [];
                }
            """
        )
        
        ### end alignment and highlight update ###

        
        model_selector.input(
            retranscribe_on_model_change,
            inputs=[model_selector, audio_input, gr.State("offline"), enable_diarization, diarization_model_selector, incremental_mode_checkbox],
            outputs=[transcript_html, original_text, audio_timestamps, audio_duration, saved_transcript_file, asr_cache],
            show_progress=True
        )
          
        diarization_model_selector.change(
            lambda audio, en, dm, cache, prev: rediarize_with_cache(audio, en, dm, cache, prev, True),
            inputs=[audio_input, enable_diarization, diarization_model_selector, asr_cache, prev_diar_enabled],
            outputs=[transcript_html, audio_timestamps, audio_duration, original_text, saved_transcript_file],
            show_progress=True
        )

        ### start double click transcript words then jump to audio position and play ###
        word_click_trigger = gr.Textbox(visible=False, elem_id="word_click_trigger")
        word_timestamp = gr.Number(value=0.0, visible=False, elem_id="word_timestamp")
        str_time=gr.Textbox(visible=False, elem_id="str_time")
        previousWord = gr.Textbox(visible=False, elem_id="previousWord")
        nextWord = gr.Textbox(visible=False, elem_id="nextWord")
        
        
        word_hidden_btn = gr.Button(visible=False, elem_id="word_hidden_btn")
        word_hidden_btn.click(
            None,  
            inputs=None,
            outputs=[word_click_trigger, previousWord, nextWord],
            js="""() => {
                return [
                    document.getElementById('word_click_trigger').textContent,
                    document.getElementById('previousWord').textContent,
                    document.getElementById('nextWord').textContent,
                ];
            }"""
        )      
        
        t_hidden_btn = gr.Button(visible=False, elem_id="t_hidden_btn")
        t_hidden_btn.click(
            None,  
            inputs=[word_timestamp, str_time],
            outputs=[word_timestamp, str_time]
        )   
        
        
        audio_input.change(
            None,  
            inputs=[transcript_html],
            outputs=[word_click_trigger, previousWord, nextWord],
            js="""
                function(transcript_html) {
                
                    function handleWordDoubleClick(e) {
                        const clickedWord = e.target;
                        const word = clickedWord.innerText.trim();
                        const previousWord = clickedWord.previousElementSibling?.innerText.trim() || 'None';
                        const nextWord = clickedWord.nextElementSibling?.innerText.trim() || 'None';
                        const triggerElem = document.getElementById('word_click_trigger');
                        triggerElem.textContent=word;
                        document.getElementById("word_hidden_btn").click();
                        
                        // console.log('e:', previousWord);
                        console.log('double clicked:', word);
                        // console.log('e:', nextWord);
                        
                        document.getElementById('previousWord').textContent=previousWord;
                        document.getElementById('nextWord').textContent=nextWord;
                    }
                    document.getElementById('transcript_html').addEventListener('dblclick', function(e){handleWordDoubleClick(e);});
                    document.getElementById('transcript_textbox').addEventListener('dblclick', function(e){handleWordDoubleClick(e);});

                    word_click_trigger=document.getElementById('word_click_trigger').textContent;
                    previousWord=document.getElementById('previousWord').textContent;
                    nextWord=document.getElementById('nextWord').textContent;
                    return [word_click_trigger, previousWord, nextWord];
                }
            """    
        )
        
        
        word_click_trigger.change(
            get_word_timestamp,
            inputs=[word_click_trigger, previousWord, nextWord, audio_timestamps],
            outputs=[word_timestamp, str_time]
        )
        word_click_trigger.change(
            None,
            inputs=None,
            outputs=None,
            js="""
                function() {
                    document.getElementById("t_hidden_btn").click();
                    return [];
                }
            """    
        )
        str_time.change(
            None,
            inputs=[word_timestamp, str_time, audio_duration],
            outputs=[],
            js="""
                function(word_timestamp, str_time, audio_duration) {
                    if (word_timestamp > 0.0) {
                        const audioElement = document.getElementById('offline_audio');
                        document.getElementById('time').textContent = str_time;
                        const percentage = (word_timestamp / audio_duration) ;
                        console.log('Jumped to str_time:', str_time, audio_duration, percentage);
                        
                        const gradioApp = document.querySelector('gradio-app');
                        const root = gradioApp.shadowRoot || gradioApp;
                        const audioContainer = root.querySelector('#offline_audio');
                        
                        const waveform = document.getElementById('waveform');
                        if (waveform) {
                            const rect = waveform.getBoundingClientRect();
                            if (audio_duration > 0) {
                                const clickX = rect.left + (rect.width * percentage);
                                const clickY = rect.top + (rect.height / 2);
                                
                                const clickEvent = new MouseEvent('click', {
                                    bubbles: true,
                                    cancelable: true,
                                    clientX: clickX,
                                    clientY: clickY,
                                    view: window
                                });
                                
                                waveform.dispatchEvent(clickEvent);
                                console.log('Dispatched click event on waveform');
                            }
                        }
                        
                        //if (audioContainer) {
                            //const playButton = audioContainer.querySelector('button[aria-label="Play"]');
                            //if (playButton) {
                               //playButton.click();
                            //}
                        //}
                    }
                    return [];
                }
            """
        )
        ### end double click transcript words then jump to audio position and play ###
                   
if __name__ == "__main__": 
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        # share=True,
        share=False,
        ssl_verify=False
    )

"""
Speaker diarization module using DiariZen.
Standalone module, called by ASR service.
Handles GPU/CPU placement and memory cleanup internally.
"""

import os
import gc
import time
import torch
from dataclasses import dataclass
from typing import List, Optional
from diarizen.pipelines.inference import DiariZenPipeline


# ===================== 默认模型 =====================
DIARIZATION_MODELS = [
    "shenjingwen/sami-diarizen-ft-006",
    "shenjingwen/sami-diarizen-ft-006L",
    "shenjingwen/sami-diarizen-ftl12.7h-ft-0.28b",
]
DEFAULT_DIARIZATION_MODEL = DIARIZATION_MODELS[1]
FALLBACK_DIARIZATION_MODEL = DIARIZATION_MODELS[-1]

@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class SpeakerDiarizer:
    """
    封装 DiariZen pipeline。
    - 懒加载模型
    - 支持指定 GPU / CPU
    - 支持把模型在 GPU / CPU 之间搬移，方便和 ASR 共享显存
    - 提供 merge_same_speaker 和 align_words_to_speakers
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DIARIZATION_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        rttm_out_dir: Optional[str] = None,
        merge_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.rttm_out_dir = rttm_out_dir
        self.merge_threshold = merge_threshold
        self.pipeline = None

    # ---------- 模型加载 / 卸载 / 设备切换 ----------

    def load(self):
        """懒加载 pipeline 到 self.device"""
        if self.pipeline is not None:
            return self.pipeline

        print(f"[Diarizer] Loading model: {self.model_name} (device={self.device})")
        try:
            self.pipeline = DiariZenPipeline.from_pretrained(
                self.model_name,
                rttm_out_dir=self.rttm_out_dir,
            )
        except Exception as e:
            print(f"[Diarizer] Failed to load {self.model_name}: {e}")
            print(f"[Diarizer] Falling back to {FALLBACK_DIARIZATION_MODEL}")
            self.model_name = FALLBACK_DIARIZATION_MODEL
            self.pipeline = DiariZenPipeline.from_pretrained(
                self.model_name,
                rttm_out_dir=self.rttm_out_dir,
            )

        # 如果 pipeline 有 .to() 就搬过去
        if hasattr(self.pipeline, "to"):
            try:
                dev = torch.device(self.device) if isinstance(self.device, str) else self.device
                self.pipeline.to(dev)
            except Exception as e:
                print(f"[Diarizer] .to({self.device}) failed: {e}")

        return self.pipeline

    def unload(self):
        """释放模型，清空显存"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to(self, device: str):
        """把已加载的 pipeline 搬到指定设备（GPU/CPU 互切）"""
        self.device = device
        if self.pipeline is not None and hasattr(self.pipeline, "to"):
            try:
                self.pipeline.to(device)
            except Exception as e:
                print(f"[Diarizer] Switch to {device} failed: {e}")
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    # ---------- 核心：对整段音频做 diarization ----------

    def diarize(self, audio_path: str, session_name: Optional[str] = None) -> List[SpeakerTurn]:
        """
        对整段音频跑一次 diarization，返回合并后的 speaker turns。
        注意：这是整段一次跑，不要按 chunk 调用。
        """
        if not audio_path or not os.path.exists(audio_path):
            print(f"[Diarizer] Audio not found: {audio_path}")
            return []

        pipe = self.load()
        sess = self._clean_session_name(session_name or os.path.splitext(os.path.basename(audio_path))[0])

        try:
            # DiariZen 返回多个值，第一个是 diarization 结果
            result = pipe(audio_path, sess_name=sess)
            diar_results = result[0] if isinstance(result, tuple) else result
        except torch.cuda.OutOfMemoryError:
            print("[Diarizer] CUDA OOM, retry on CPU")
            self.to("cpu")
            result = self.pipeline(audio_path, sess_name=sess)
            diar_results = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"[Diarizer] diarize failed: {e}")
            return []

        raw_turns: List[SpeakerTurn] = []
        for turn, _, speaker in diar_results.itertracks(yield_label=True):
            raw_turns.append(SpeakerTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            ))

        merged = self.merge_same_speaker(raw_turns)
        print(f"[Diarizer] raw turns={len(raw_turns)} merged turns={len(merged)}")
        return merged

    # ---------- 工具函数 ----------

    def merge_same_speaker(self, turns: List[SpeakerTurn]) -> List[SpeakerTurn]:
        """合并同一说话人、间隔 ≤ merge_threshold 的连续分片"""
        if not turns:
            return []
        turns = sorted(turns, key=lambda t: t.start)
        merged = [turns[0]]
        for cur in turns[1:]:
            last = merged[-1]
            gap = cur.start - last.end
            if cur.speaker == last.speaker and gap <= self.merge_threshold:
                merged[-1] = SpeakerTurn(
                    start=last.start,
                    end=max(last.end, cur.end),
                    speaker=last.speaker,
                )
            else:
                merged.append(cur)
        return merged

    @staticmethod
    def assign_speaker(word_start: float, word_end: float, turns: List[SpeakerTurn]) -> str:
        """找与 [word_start, word_end] 重叠最多的 speaker turn"""
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for t in turns:
            ov_start = max(word_start, t.start)
            ov_end = min(word_end, t.end)
            ov = max(0.0, ov_end - ov_start)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = t.speaker
        return best_speaker

    @staticmethod
    def merge_words_by_speaker(word_timestamps: List[dict]) -> List[dict]:
        """
        输入: [{"text": str, "start": float, "end": float, "speaker": str}, ...]
        输出: [{"speaker": str, "words": [str], "start": float, "end": float}, ...]
        """
        if not word_timestamps:
            return []
        segments = []
        cur = {
            "speaker": word_timestamps[0].get("speaker", "UNKNOWN"),
            "words": [word_timestamps[0]["text"]],
            "start": word_timestamps[0]["start"],
            "end": word_timestamps[0]["end"],
        }
        for w in word_timestamps[1:]:
            spk = w.get("speaker", "UNKNOWN")
            if spk == cur["speaker"]:
                cur["words"].append(w["text"])
                cur["end"] = w["end"]
            else:
                segments.append(cur)
                cur = {
                    "speaker": spk,
                    "words": [w["text"]],
                    "start": w["start"],
                    "end": w["end"],
                }
        segments.append(cur)
        return segments
        
    @staticmethod
    def speaker_to_label(speaker: str) -> str:
        """SPEAKER_00 -> S1, SPEAKER_01 -> S2, ..."""
        import re
        m = re.search(r"(\d+)$", speaker)
        if m:
            return f"S{int(m.group(1))}"
        return speaker
    
    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """秒 -> HH:MM:SS.mmm"""
        if seconds is None:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"
        
    @staticmethod
    def format_segments(segments: List[dict]) -> str:
        """*start-end* S1: text"""
        """*HH:MM:SS.mmm-HH:MM:SS.mmm* S1: text"""
        lines = []
        for seg in segments:
            spk = SpeakerDiarizer.speaker_to_label(seg["speaker"])
            start = SpeakerDiarizer._fmt_time(seg.get("start", 0.0))
            end = SpeakerDiarizer._fmt_time(seg.get("end", 0.0))
            text = " ".join(seg["words"])
            lines.append(f"*{start}-{end}* {spk}: {text}")
        return "\n\n".join(lines)

    @staticmethod
    def _clean_session_name(name: str) -> str:
        return name.replace(" ", "_")


# ===================== 单例（可选） =====================
_default_diarizer: Optional[SpeakerDiarizer] = None

def get_default_diarizer() -> SpeakerDiarizer:
    global _default_diarizer
    if _default_diarizer is None:
        _default_diarizer = SpeakerDiarizer()
    return _default_diarizer
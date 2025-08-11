#!/usr/bin/env python3
"""
MVP transcript -> podcast pipeline using:
 - LangGraph for the "sub-agent" verbal-tag injection
 - Dia (via transformers AutoProcessor + DiaForConditionalGeneration) for TTS

Quickstart (recommended in a new venv):
  pip install -U langgraph langchain langchain-openai transformers torch soundfile
  pydub datasets python-dotenv

Requirements / notes:
 - You will want a CUDA GPU for Dia (the model authors note ~10GB VRAM for the
   full model).
 - You should set OPENAI_API_KEY (or edit to use another init_chat_model
   provider).
 - ffmpeg is required by pydub to concatenate audio.
 - This is MVP code: sanitize inputs and improve error handling for production
   use.
"""

import os
import random
import re
import shutil
import sys
import tempfile
from typing import Dict, List

# TTS / Dia
import torch
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph + LangChain (for LLM)
from langgraph.graph import START, StateGraph
from pydub import AudioSegment
from transformers import AutoProcessor, DiaForConditionalGeneration

# ---- Config / defaults ----
DIA_CHECKPOINT = os.environ.get("DIA_CHECKPOINT", "nari-labs/Dia-1.6B")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL", "openai:gpt-4o-mini")  # changeable
CONTEXT_WINDOW = 2  # prev/next lines to give sub-agent
VERBAL_TAGS = [
    "(laughs)",
    "(clears throat)",
    "(sighs)",
    "(gasps)",
    "(coughs)",
    "(singing)",
    "(sings)",
    "(mumbles)",
    "(beep)",
    "(groans)",
    "(sniffs)",
    "(claps)",
    "(screams)",
    "(inhales)",
    "(exhales)",
    "(applause)",
    "(burps)",
    "(humming)",
    "(sneezes)",
    "(chuckle)",
    "(whistles)",
]
LINE_COMBINERS = [
    "…um,",
    "- uh -",
    "— hmm —",
    #    "— well —",
    #    "— you know —",
    #    "- yeah, uh -"
]
PAUSE_PLACEHOLDER = "[insert-verbal-tag-for-pause]"
MAX_TAG_RATE = 0.15  # sparing: 15% of lines may receive an ad-hoc tag (approx)
AVG_WPS = 2.5  # words per second estimate
MAX_NEW_TOKENS_CAP = 1600  # Dia generate cap (adjust per GPU/memory)


# ---- Utilities: parsing and normalization ----
def parse_simple_txt(path: str) -> List[Dict]:
    """
    Very small parser: supports lines that are either:
      [S1] text...
      Name: text...
      or bare text (assumed S1)
    Returns list of {'speaker': 'S1'|'S2', 'text': '...'}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    parsed = []
    speakers_map = {}  # name -> S1/S2
    next_speaker_id = 1

    for ln in raw_lines:
        # bracketed form
        m = re.match(r"^\[?(S\d+)\]?\s*(.*)$", ln)
        if m:
            speaker = m.group(1)
            text = m.group(2).strip()
            parsed.append({"speaker": speaker, "text": text})
            continue
        # "Name: ... " form
        m2 = re.match(r"^([A-Za-z0-9 _\-]+?):\s*(.*)$", ln)
        if m2:
            name = m2.group(1).strip()
            text = m2.group(2).strip()
            if name not in speakers_map:
                tag = f"S{next_speaker_id}"
                speakers_map[name] = tag
                next_speaker_id += 1
            parsed.append({"speaker": speakers_map[name], "text": text})
            continue
        # fallback: treat as continuation of previous speaker if any, else S1
        if parsed:
            parsed[-1]["text"] += " " + ln
        else:
            parsed.append({"speaker": "S1", "text": ln})
    return parsed


def parse_srt(path: str) -> List[Dict]:
    """
    Extremely-primitive .srt parser: returns text lines ignoring timestamps.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # split numbered blocks
    blocks = re.split(r"\n\s*\n", content)
    parsed = []
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        # drop index and timestamps if present
        if re.match(r"^\d+$", lines[0]) and len(lines) >= 2:
            candidate = (
                " ".join(lines[2:]) if "-->" in lines[1] else " ".join(lines[1:])
            )
        else:
            candidate = " ".join(lines)
        # no speaker info in srt - push as S1 by default
        parsed.append({"speaker": "S1", "text": candidate})
    return parsed


def ingest_transcript(path: str) -> List[Dict]:
    if path.lower().endswith(".srt"):
        return parse_srt(path)
    else:
        return parse_simple_txt(path)


# ---- Merge consecutive same-speaker lines with placeholder ----
def merge_consecutive_lines(lines: List[Dict]) -> List[Dict]:
    out = []
    warnings = []
    for ln in lines:
        if out and ln["speaker"] == out[-1]["speaker"]:
            # warn & merge with placeholder
            warnings.append(
                f"Consecutive lines from {ln['speaker']}, merging with placeholder."
            )
            out[-1]["text"] = (
                out[-1]["text"].rstrip()
                + " "
                + PAUSE_PLACEHOLDER
                + " "
                + ln["text"].lstrip()
            )
        else:
            out.append(ln.copy())
    if warnings:
        print("WARN:", *warnings, sep="\n  - ")
    return out


# ---- LangGraph sub-agent (injector) ----
def build_langgraph_injector(llm=None):
    """
    Construct a simple LangGraph StateGraph with a single node `inject_line`.
    The node expects state keys:
      prev_lines: List[str]
      current_line: str
      next_lines: List[str]
      summary: str
      topic: str
    and returns {'modified_line': '...'}
    If llm is None, the node uses the fallback rule-based injector.
    """

    # typed-free approach for simplicity
    def rule_based_injector(state):
        prev_lines = state.get("prev_lines", [])
        cur = state["current_line"]
        next_lines = state.get("next_lines", [])
        # replace placeholder with random varied tag
        if PAUSE_PLACEHOLDER in cur:
            tag = random.choice(LINE_COMBINERS)
            cur = cur.replace(PAUSE_PLACEHOLDER, tag)
        else:
            # maybe insert a sparing tag at start (approx MAX_TAG_RATE)
            if random.random() < MAX_TAG_RATE:
                tag = random.choice(VERBAL_TAGS)
                # ensure format like: [S1] (gasps) rest...
                cur = re.sub(r"^(\[S[12]\])\s*", r"\1 " + tag + " ", cur)
        return {"modified_line": cur}

    def inject_node(state):
        # prefer LLM if provided
        if llm is None:
            return rule_based_injector(state)

        # build a prompt for the LLM
        prev_lines = state.get("prev_lines", [])
        cur = state["current_line"]
        next_lines = state.get("next_lines", [])
        summary = state.get("summary", "")
        topic = state.get("topic", "")

        system = SystemMessage(
            content=(
                "You are a concise transcript editor. You receive a single transcript "
                "line prefixed by a speaker tag ([S1] or [S2]) and the surrounding "
                "context. Return ONLY the updated single line (no commentary). "
                "Rules:\n"
                " - Keep the leading speaker tag exactly as [S1] or [S2].\n"
                " - If the line contains the placeholder "
                "[insert-verbal-tag-for-pause], replace it with one "
                "appropriate verbal tag (choose from the provided set) and do not "
                "add any others.\n"
                " - You may sparsely (<=15% of lines) add a short verbal tag to the "
                "start of the spoken text (immediately after the speaker tag) when "
                "context suggests it (e.g., (gasps), (laughs), …um,). "
                " - Do NOT overuse tags; maintain naturalness and vary the chosen "
                "tag.\n"
                " - Do not alter the main semantic content other than "
                "inserting/replacing verbal tags.\n"
                " - Output must be a single transcript line starting with the "
                "speaker tag.\n"
            )
        )
        human_prompt = (
            f"Prev lines:\n{chr(10).join(prev_lines)}\n\n"
            f"Current line:\n{cur}\n\n"
            f"Next lines:\n{chr(10).join(next_lines)}\n\n"
            f"Conversation summary (short): {summary}\n"
            f"Current topic: {topic}\n\n"
            f"Available verbal tags (example set): {VERBAL_TAGS}\n\n"
            "Return only the modified single line."
        )

        resp = llm.invoke([system, HumanMessage(content=human_prompt)])
        # resp is an AIMessage — extract content
        content = getattr(resp, "content", None)
        if content is None:
            # fallback: convert to str
            content = str(resp)
        # take only first non-empty line (safety)
        for line in content.splitlines():
            if line.strip():
                modified = line.strip()
                break
        else:
            modified = cur
        return {"modified_line": modified}

    builder = StateGraph(dict)  # basic untyped StateGraph for MVP
    builder.add_node("inject_line", inject_node)
    builder.add_edge(START, "inject_line")
    graph = builder.compile()
    return graph


# ---- chunking: make 5-10s mini-transcripts ----
def estimate_seconds_for_text(text: str) -> float:
    words = len(text.split())
    return words / AVG_WPS


def chunk_to_5_10s(lines: List[Dict]) -> List[str]:
    """
    Combine line strings into blocks of 5..10 seconds. Each block is a
    newline-separated transcript with speaker tags preserved (e.g.,
    "[S1] ...\n[S2] ...").
    """
    blocks = []
    buf = []
    buf_seconds = 0.0

    i = 0
    while i < len(lines):
        ln = lines[i]
        sline = (
            f"[{ln['speaker']}] {ln['text']}"
            if not ln["text"].startswith(f"[{ln['speaker']}]")
            else ln["text"]
        )
        sec = estimate_seconds_for_text(ln["text"])
        # if single line longer than 10s: split crude by sentence punctuation
        if sec > 10:
            # naive split
            parts = re.split(r"(?<=[\\.\\?\\!])\\s+", ln["text"])
            parts = [p.strip() for p in parts if p.strip()]
            for p in parts:
                s = estimate_seconds_for_text(p)
                if buf_seconds + s <= 10:
                    buf.append(f"[{ln['speaker']}] {p}")
                    buf_seconds += s
                else:
                    if buf and buf_seconds >= 5:
                        blocks.append("\n".join(buf))
                        buf = []
                        buf_seconds = 0.0
                    # put the long chunk alone if >10s after splitting: accept
                    # it (edge-case)
                    blocks.append(f"[{ln['speaker']}] {p}")
            i += 1
            continue

        # normal flow: append until we are at least 5s but no more than 10s
        if buf_seconds + sec <= 10:
            buf.append(sline)
            buf_seconds += sec
            # if we reached >=5s, finalize a block (but try to accumulate a bit
            # more to favor upper bound)
            if buf_seconds >= 5:
                blocks.append("\n".join(buf))
                buf = []
                buf_seconds = 0.0
            i += 1
        else:
            # buffer would exceed 10s if we add this line: flush buffer if
            # >=5s, else put the line alone (if buffer <5s)
            if buf_seconds >= 5:
                blocks.append("\n".join(buf))
                buf = []
                buf_seconds = 0.0
            else:
                # combine buf + this line even though >10s; prefer to keep
                # continuity (rare)
                buf.append(sline)
                blocks.append("\n".join(buf))
                buf = []
                buf_seconds = 0.0
                i += 1


    # leftover
    if buf:
        # try to merge with previous block if too short
        if blocks and estimate_seconds_for_text("\n".join(buf)) < 5:
            blocks[-1] = blocks[-1] + "\n" + "\n".join(buf)
        else:
            blocks.append("\n".join(buf))
    return blocks


# ---- Dia TTS helpers ----
class DiaTTS:
    def __init__(self, model_checkpoint=DIA_CHECKPOINT, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[DiaTTS] Loading model {model_checkpoint} on device {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(model_checkpoint)
        self.model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(
            self.device
        )

    def text_to_audio_file(self, text: str, out_path: str, max_new_tokens: int = 512):
        # prepare inputs
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(
            self.device
        )
        est_seconds = estimate_seconds_for_text(text)
        # heuristics for tokens: approx 86 tokens ~= 1s (authors' reference); scale up
        desired_tokens = min(MAX_NEW_TOKENS_CAP, max(256, int(est_seconds * 86) + 128))
        desired_tokens = min(desired_tokens, max_new_tokens)
        # generate
        outputs = self.model.generate(**inputs, max_new_tokens=desired_tokens)
        # decode
        decoded = self.processor.batch_decode(outputs)
        # save audio (processor offers save_audio convenience)
        # decoded is a structure that processor.save_audio expects
        self.processor.save_audio(decoded, out_path)
        return out_path


# ---- TOP-LEVEL pipeline ----
def run_pipeline(
    input_path: str,
    out_audio_path: str,
    dia_checkpoint: str = DIA_CHECKPOINT,
    openai_model: str = OPENAI_MODEL_NAME,
):
    # ingest
    lines = ingest_transcript(input_path)
    # normalize to [S1]/[S2] already done in parser; ensure tag formatting
    lines = [
        {
            "speaker": (
                ln["speaker"] if ln["speaker"].startswith("S") else ln["speaker"]
            ),
            "text": ln["text"],
        }
        for ln in lines
    ]
    # merge consecutive
    lines = merge_consecutive_lines(lines)

    # build LLM (optional)
    llm = None
    if os.environ.get("OPENAI_API_KEY"):
        print("[pipeline] Initializing LLM (via init_chat_model) ...")
        try:
            llm = init_chat_model(openai_model)
        except Exception as e:
            print("[pipeline] LLM init failed, falling back to rule-based injector:", e)
            llm = None
    else:
        print(
            "[pipeline] OPENAI_API_KEY not found: using rule-based injector fallback."
        )

    graph = build_langgraph_injector(llm=llm)

    # simple summary / topic (MVP: derive short summary via naive concatenation
    # or user-provided)
    convo_summary = "A short conversational summary (MVP): " + " ".join(
        [ln["text"] for ln in lines[:6]]
    )
    current_topic = lines[0]["text"].split()[0:6] if lines else "general"

    # process each line through LangGraph sub-agent
    processed = []
    for idx, ln in enumerate(lines):
        prev_lines = [
            f"[{l['speaker']}] {l['text']}"
            for l in lines[max(0, idx - CONTEXT_WINDOW) : idx]
        ]
        next_lines = [
            f"[{l['speaker']}] {l['text']}"
            for l in lines[idx + 1 : idx + 1 + CONTEXT_WINDOW]
        ]
        state_in = {
            "prev_lines": prev_lines,
            "current_line": f"[{ln['speaker']}] {ln['text']}",
            "next_lines": next_lines,
            "summary": convo_summary,
            "topic": (
                " ".join(current_topic)
                if isinstance(current_topic, list)
                else str(current_topic)
            ),
        }
        try:
            result = graph.invoke(state_in)
            # result should include 'modified_line'
            modified = result.get("modified_line") or state_in["current_line"]
        except Exception as e:
            print("[pipeline] LangGraph injector failed for line", idx, ":", e)
            # fallback to rule-based local
            modified = state_in["current_line"]
            if PAUSE_PLACEHOLDER in modified:
                modified = modified.replace(
                    PAUSE_PLACEHOLDER, random.choice(VERBAL_TAGS)
                )
        processed.append(
            {"speaker": ln["speaker"], "text": re.sub(r"^\[S[12]\]\s*", "", modified)}
        )

    # chunk into 5-10s mini transcripts
    mini_transcripts = chunk_to_5_10s(processed)
    print(
        f"[pipeline] Produced {len(mini_transcripts)} mini-transcript blocks "
        f"(5-10s preferred)."
    )

    # TTS each block with Dia
    tts = DiaTTS(model_checkpoint=dia_checkpoint)
    tmp_dir = tempfile.mkdtemp(prefix="dia_mvp_")
    seg_paths = []
    try:
        for i, block in enumerate(mini_transcripts):
            out_wav = os.path.join(tmp_dir, f"block_{i:04d}.wav")
            print(
                f"[TTS] Generating block {i + 1}/{len(mini_transcripts)} -> "
                f"{out_wav} ..."
            )
            tts.text_to_audio_file(block, out_wav, max_new_tokens=MAX_NEW_TOKENS_CAP)
            seg_paths.append(out_wav)

        # concatenate with pydub
        print("[audio] Concatenating audio segments ...")
        combined = None
        for p in seg_paths:
            seg = AudioSegment.from_wav(p)
            if combined is None:
                combined = seg
            else:
                combined += seg

        # export final mp3 (or wav)
        if combined is None:
            raise RuntimeError("No audio segments produced.")
        print(f"[audio] Exporting final to {out_audio_path} ...")
        combined.export(out_audio_path, format="mp3")
        print("[pipeline] Done.")
    finally:
        # cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---- CLI ----
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <srt|vtt|transcript> <mp3 output>")
        sys.exit(1)
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    run_pipeline(input_path, out_path)

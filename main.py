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

import argparse

from src.pipeline import run_pipeline

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a transcript to a podcast using Dia TTS."
    )
    parser.add_argument(
        "input_path", help="Path to the input transcript file (srt, txt)."
    )
    parser.add_argument("output_path", help="Path to save the final MP3 audio file.")
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible voice selection."
    )
    parser.add_argument(
        "--s1-voice", type=str, help="Path to an audio prompt file for Speaker 1."
    )
    parser.add_argument(
        "--s2-voice", type=str, help="Path to an audio prompt file for Speaker 2."
    )

    args = parser.parse_args()

    # Build a dictionary of voice prompts
    voice_prompts = {}
    if args.s1_voice:
        voice_prompts["S1"] = args.s1_voice
    if args.s2_voice:
        voice_prompts["S2"] = args.s2_voice

    # Pass the prompts to the pipeline
    run_pipeline(
        args.input_path, args.output_path, voice_prompts=voice_prompts, seed=args.seed
    )

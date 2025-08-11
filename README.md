# BomDia – Transcript-to-Podcast Pipeline

> “Turn any script into a studio-quality morning talk show.”

BomDia is a minimal-viable pipeline that converts a plain-text or `.srt` transcript into an ultra-realistic **dialogue podcast** using [Dia 1.6B](https://github.com/nari-labs/dia) – a state-of-the-art text-to-speech model – and LangGraph for intelligent verbal-tag injection.

---

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/your-org/bomdia.git
cd bomdia

# 2. Create venv with uv (or pip)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt   # or pip install -r requirements.txt

# 3. Set your OpenAI API key (optional, for smarter tag injection)
export OPENAI_API_KEY="sk-..."

# 4. Run
python main.py simple-transcript.txt output.mp3 \
  --s1-voice alice.wav \
  --s2-voice bob.wav
```

That’s it. `output.mp3` now contains a natural conversation between **S1** and **S2**, complete with laughs, sighs, and realistic pacing—just like a morning radio talk show.

---

## Input Formats

| File type | Auto-detected? | Speaker labels |
|-----------|----------------|----------------|
| `.txt`    | ✓              | `[S1]` / `[S2]` or `Name:` |
| `.srt`    | ✓              | All mapped to `S1` (you can rename later) |

Example (`simple-transcript.txt`):

```
Alice: Hey, I thought we should —
Bob: Yeah, I was thinking the same.
Alice: Oh? So, what do you want to get then?
Bob: No! what? I thought you meant we should grab something to eat.
```

---

## Pipeline Walk-through

1. **Ingest**
   Parses the transcript into JSON lines:
   `{speaker: "S1", text: "Hello there"}`

2. **Merge consecutive same-speaker lines**
   If two `S1` lines appear back-to-back they’re merged with
   `[insert-verbal-tag-for-pause]` so the dialogue alternates correctly.

3. **LangGraph sub-agent**
   For every line the agent receives
   - the 2 previous lines
   - the 2 next lines
   - a short running summary
   - current topic

   It decides—sparingly—where to inject verbal tags such as
   `(laughs)`, `…um,`, `(sighs)`, etc.
   `[insert-verbal-tag-for-pause]` is replaced with the most contextually appropriate tag.

4. **Chunking**
   Breaks the dialogue into **5-10 second** mini-transcripts.
   Each chunk still starts with `[S1]` or `[S2]` as required by Dia.

5. **Dia TTS**
   Feeds every mini-transcript to Dia (GPU ≈ 4-10 GB VRAM).
   Voice-cloning is optional: supply 5–10 s WAV files with `--s1-voice` / `--s2-voice`.

6. **Concatenate**
   Combines all WAV chunks into the final MP3 with `pydub`.

---

## CLI Reference

```
python main.py INPUT OUTPUT [--options]

positional:
  input_path       transcript file (txt | srt)
  output_path      final MP3

optional:
  --seed N         deterministic voice selection
  --s1-voice FILE  5-10 s WAV prompt for speaker 1
  --s2-voice FILE  5-10 s WAV prompt for speaker 2
```

---

## Configuration

All tunables live in `config/app.toml`, e.g.

```toml
[model]
dia_checkpoint = "nari-labs/Dia-1.6B-0626"
openai_model   = "openai:gpt-4o-mini"

[pipeline]
context_window = 2          # lines of context for sub-agent
max_tag_rate   = 0.15       # 15 % of lines may receive a tag
avg_wps        = 2.5        # words per second estimate
```

Override with env vars:
`DIA_CHECKPOINT`, `CONTEXT_WINDOW`, etc.

---

## Voice Prompt Tips

- **Length**: 5–10 s
- **Format**: 16-bit, 22 kHz or 44 kHz WAV (mono preferred)
- **Content**: A single clear sentence is enough; Dia will learn timbre & style.

---

## System Requirements

| Component | Spec |
|-----------|------|
| Python    | 3.10+ |
| GPU       | CUDA 12.6+ (≥ 10 GB VRAM for full float16) |
| OS        | Linux / macOS (Windows WSL2 supported) |

---

## Project Layout

```
bomdia/
├── main.py               # CLI entry point
├── src/
│   ├── pipeline.py       # orchestrates everything
│   ├── components/
│   │   ├── transcript_parser/
│   │   ├── verbal_tag_injector/
│   │   └── audio_generator/
├── config/
│   ├── app.toml
│   └── prompts.toml
├── tests/                # root integration tests
└── docs/
    └── components/       # per-system documentation
```

---

## Development & Testing

```bash
uv run black src/ shared/ main.py
uv run ruff check --fix  src/ shared/ main.py
uv run mypy  src/ shared/ main.py
uv run bandit -c pyproject.toml -r src/ shared/ main.py
```

Minimum 25 % line coverage enforced.

---

## License

Apache 2.0 – see [LICENSE](LICENSE).

---

## Disclaimer

This project uses a high-fidelity voice synthesis model.
**Misuse** (impersonation, fake news, harassment) is **strictly forbidden**.
By using BomDia you agree to the [Dia disclaimer](https://github.com/nari-labs/dia#-disclaimer).

---

## Contributing

PRs welcome!
Start in `docs/components/` if you add new subsystems, and keep the no-magic-variables rule sacred.

---

🪷 “BomDia” is a nod to the Portuguese greeting for “good morning,” often heard on morning talk shows in Portugal and Brazil.

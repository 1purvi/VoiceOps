# ZentroMart Voice Support Agent — VoiceOps 2026

Capstone for the LLM & Generative AI course (Consulting & Analytics Club, IIT Guwahati × Gnani.ai).
Author: **Dharmana Purvi**
Repository: <PUBLIC_GITHUB_URL>

A customer-support agent for ZentroMart (a fictional Indian online store) that answers from company
documents, holds a multi-turn conversation with order lookup and ticket tools, and talks over voice with
Gnani speech-to-text and text-to-speech.

```
customer audio → Gnani STT → agent (LLM + RAG + tools) → Gnani TTS → spoken reply
```

| Step | What | Result |
| --- | --- | --- |
| 1 | Prompt-only LLM on `practice/questions.csv` | 12 / 52 correct (23.1%) |
| 2 | RAG over `kb/` (hybrid dense + BM25) | 52 / 52 correct (100%) |
| 3 | Agent with tools + memory on `practice/conversations.csv` | 96 / 96 turns, 23 / 23 conversations |
| 4 | Gnani STT / TTS voice wrapper | English, Hindi and Hinglish calls end to end |

Full write-up: [`report.pdf`](report.pdf). Demo: [`demo.mp4`](demo.mp4).

## Repository layout

```
├── README.md
├── .env.example            copy to .env and fill in keys (never commit .env)
├── requirements.txt
├── run.py                  the one entry point: web-app call, or batch run over a folder of audio
├── report.pdf
├── demo.mp4
├── src/
│   ├── config.py           paths, .env loading, today = 20 Oct 2026
│   ├── llm.py              Step 1: system prompt, LLM call, JSON parsing
│   ├── metrics.py          scorer shared by Steps 1–3
│   ├── eval_cache.py       cached evaluation runs for Steps 1–2
│   ├── step1_baseline.ipynb
│   ├── rag_pipeline.py     Step 2: chunking, hybrid retrieval, date table, grounded answer
│   ├── step2_rag.ipynb
│   ├── agent.py            Step 3: tool-using agent + conversation evaluator
│   ├── step3_conversations.py
│   ├── voice.py            Step 4: Gnani STT/TTS around the Step 3 agent
│   ├── web_app.py          Gradio call UI started by run.py
│   ├── batch_run.py        batch mode used by run.py --audio_dir
│   ├── call_handsfree.py   optional terminal call (voice activity detection)
│   ├── call_push_to_talk.py optional terminal call (press Enter to talk)
│   └── data_validation.py  dataset checks
└── sample_outputs/
    ├── run/                    replies from run.py (text + audio + chunks used)
    ├── live_call_2026-10-31/   a recorded live call (audio + JSON per turn)
    └── evaluation/             Step 1–3 evaluation results (CSV)
```

## Setup

1. Python 3.10+.
2. Install dependencies:
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download `Capstone_Project_Dataset_FTTT_2026.zip` (link in the capstone brief) and extract it into the
   repository root, so that `Capstone_Project_Dataset_FTTT_2026/kb/` exists. The dataset is not committed.
4. Copy `.env.example` to `.env` and fill in `GNANI_API_KEY` and the `LLM_*` values.

The first run builds the vector index (`outputs/vector_index.pkl`), which takes about a minute.

## Run

Everything starts from one file, `run.py`, in the repository root.

**Talk to the agent (web app)**

```
python run.py
```

The browser opens at http://127.0.0.1:7860, a call starts and the agent greets you by voice. Record a
question with the mic and the agent answers in text and speech; follow-up questions keep the context. Say
"bye" to end the call, or press **New call**. Every turn is saved to `outputs/live_calls/<timestamp>/`.
Options: `--port 8000`, `--share` (public Gradio link), `--voice Kaveri`.

**Batch run (organisers)**

```
python run.py --audio_dir path/to/audio_folder
```

Each audio file (`.wav .mp3 .m4a .flac .ogg .aac`) is a separate call. Add `--conversation` to treat the
files, sorted by name, as one multi-turn call, and `--out_dir` to change the output folder (default
`sample_outputs/run`). For every input file it writes:

- `<name>.reply.wav` — the spoken reply (Gnani TTS)
- `<name>.json` — transcript, reply text, decision, tool calls and the document chunks used (id, source, text)

plus `results.csv` and `results.jsonl` for the whole folder.

**Evaluation (Steps 1–3):** open `src/step1_baseline.ipynb` and `src/step2_rag.ipynb`, or run
`python src/step3_conversations.py`. Saved results are in `sample_outputs/evaluation/`.

## Tech used

- **LLM:** `nvidia/nemotron-3.5-super-vl-120b-a12b` via the NVIDIA API (OpenAI-compatible; any provider works)
- **Retrieval:** `sentence-transformers` (`intfloat/multilingual-e5-base`) + `rank-bm25`, fused with reciprocal rank fusion
- **Tools:** provided `lookup_order` and `create_ticket`, plus `search_kb`, `date_calc`, `final_answer`
- **Speech:** Gnani only — STT `prisma-v3.0`, TTS `timbre-v2.5` (voice Kaveri) via `gnani-vachana`
- **Interface:** terminal (`sounddevice`) and Gradio

## Rules followed

- Gnani is the only speech service; there is no other STT/TTS, even as a fallback.
- Only the synthetic ZentroMart data is used.
- Answers come from the documents at run time; nothing is hardcoded.
- API keys live in `.env`, which is git-ignored.

## Building the submission ZIP

```
cd ..
zip -r dharmana_purvi_voiceops.zip dharmana_purvi_voiceops \
    -x "*/__pycache__/*" "*/.DS_Store" "*/.env" "*/.venv/*" "*/outputs/*" "*/Capstone_Project_Dataset_FTTT_2026/*"
```

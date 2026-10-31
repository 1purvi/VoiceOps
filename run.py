"""ZentroMart voice support agent: the one command to run everything.

    python run.py                                   # web app: opens the browser, starts a call, ask by mic
    python run.py --audio_dir path/to/audio         # batch: every audio file = a separate call
    python run.py --audio_dir path/to/call --conversation   # batch: files in name order = one call

Web app: the agent greets you by voice; record a question with the mic, hear the spoken reply.
Say "bye" to end the call, or press "New call". Every turn is saved to outputs/live_calls/<timestamp>/.

Batch (organiser test): for each audio file writes <name>.reply.wav and <name>.json (transcript, reply,
decision, tools used and the document chunks used), plus results.csv and results.jsonl.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def check_setup():
    """Stop early with a clear message if .env, keys or the dataset are missing."""
    problems = []
    if not (ROOT / ".env").exists():
        problems.append("No .env file: copy .env.example to .env and fill in the keys.")
    import config                                                  # loads .env
    for key in ("GNANI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "EMBED_MODEL"):
        if not os.getenv(key):
            problems.append(f"{key} is empty in .env.")
    if not config.KB_DIR.exists():
        problems.append(f"Dataset not found at {config.DATASET_DIR}: extract "
                        "Capstone_Project_Dataset_FTTT_2026.zip into the repository root.")
    if problems:
        sys.exit("Setup incomplete:\n  - " + "\n  - ".join(problems))


def warm_up():
    """Load the embedding model and build / load the vector index once, before the first question."""
    from rag_pipeline import retrieve
    print("Loading the knowledge base index...")
    retrieve("return window")


def main():
    ap = argparse.ArgumentParser(description="ZentroMart voice support agent")
    ap.add_argument("--audio_dir", help="batch mode: folder of customer audio files")
    ap.add_argument("--out_dir", help="batch mode: where replies are saved (default sample_outputs/run)")
    ap.add_argument("--conversation", action="store_true",
                    help="batch mode: treat the files (sorted by name) as one conversation")
    ap.add_argument("--voice", help="Gnani timbre-v2.5 voice (default Kaveri)")
    ap.add_argument("--port", type=int, default=7860, help="web app port (default 7860)")
    ap.add_argument("--share", action="store_true", help="web app: also create a public Gradio link")
    args = ap.parse_args()

    check_setup()
    warm_up()

    if args.audio_dir:                                             # organiser batch run
        import batch_run
        argv = ["--audio_dir", args.audio_dir, "--out_dir", args.out_dir or str(ROOT / "sample_outputs" / "run")]
        if args.conversation:
            argv.append("--conversation")
        if args.voice:
            argv += ["--voice", args.voice]
        batch_run.main(argv)
    else:                                                          # web app with a live call
        if args.voice:
            os.environ["TTS_VOICE"] = args.voice
        import web_app
        print(f"Opening the call at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
        web_app.launch(port=args.port, share=args.share)


if __name__ == "__main__":
    main()

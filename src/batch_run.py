"""Batch mode used by run.py: answer every customer audio file in a folder, by voice.

    python run.py --audio_dir path/to/audio                 # each file = a separate call (test questions)
    python run.py --audio_dir path/to/call --conversation   # files in name order = one call (multi-turn)

For each input file it writes to --out_dir (default outputs/run_outputs/):
    <name>.reply.wav   the spoken reply (Gnani TTS)
    <name>.json        transcript, reply text, decision, tools used and the document chunks used
and a summary results.csv + results.jsonl for the whole folder.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE if (HERE / "voice.py").exists() else HERE / "src"     # works from VoiceOps/ or VoiceOps/src/
sys.path.insert(0, str(SRC))

from voice import VoiceAgent, AUDIO_EXTENSIONS, TTS_VOICE  # noqa: E402
from config import OUT_DIR  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="ZentroMart voice agent: audio folder -> spoken replies")
    ap.add_argument("--audio_dir", default=str(OUT_DIR / "audio_test"),
                    help="folder with customer audio (.wav .mp3 .m4a .flac .ogg .aac), default outputs/call_check")
    ap.add_argument("--out_dir", default=str(OUT_DIR / "audio_test"), help="where replies are saved")
    ap.add_argument("--conversation", action="store_true",
                    help="treat the files (sorted by name) as one conversation instead of separate calls")
    ap.add_argument("--voice", default=TTS_VOICE, help="Gnani timbre-v2.5 voice (default Kaveri)")
    args = ap.parse_args(argv)

    audio_dir, out_dir = Path(args.audio_dir), Path(args.out_dir)
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        sys.exit(f"No audio files found in {audio_dir} ({', '.join(sorted(AUDIO_EXTENSIONS))})")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} audio file(s) | {'one conversation' if args.conversation else 'separate calls'} "
          f"| output: {out_dir}")

    records, va = [], VoiceAgent(args.voice) if args.conversation else None
    for i, f in enumerate(files, 1):
        agent = va if args.conversation else VoiceAgent(args.voice)       # fresh memory per call
        rec = agent.turn(f, out_dir / f"{f.stem}.reply.wav")
        rec["file"] = f.name
        (out_dir / f"{f.stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        records.append(rec)
        print(f"[{i}/{len(files)}] {f.name} ({rec['seconds']}s, {rec['decision']})\n"
              f"   heard : {rec['transcript']}\n   reply : {rec['reply']}\n"
              f"   chunks: {[c['id'] for c in rec['chunks']]}" + (f"\n   ERROR : {rec['error']}" if rec["error"] else ""))

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["file", "transcript", "stt_language", "reply", "decision", "used_sections", "reply_audio",
            "unclear_audio", "error", "seconds"]
    with open(out_dir / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({**r, "used_sections": " ; ".join(r["used_sections"])})
    errors = sum(1 for r in records if r["error"])
    print(f"\nDone: {len(records)} files, {errors} with errors. Results in {out_dir}/results.csv")


if __name__ == "__main__":
    main()

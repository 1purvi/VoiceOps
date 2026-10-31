"""Live voice call with the ZentroMart agent: speak into the mic, hear the reply, say "bye" to end.

    python src/call_push_to_talk.py                    # live mic call
    python src/call_push_to_talk.py --voice Kaveri     # pick the TTS voice

Each turn: press Enter to start talking, press Enter again to stop. The agent transcribes, answers,
and plays its reply. The call ends after a turn where you say bye / goodbye / alvida / बाय / अलविदा.

Everything is saved to outputs/live_calls/<timestamp>/ :
    turn01.customer.wav   your recording
    turn01.reply.wav      the agent's spoken reply
    turn01.json           transcript, reply, decision, chunks used
    call.jsonl            all turns of the call
"""
import argparse
import json
import queue
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = Path(__file__).resolve().parent
SRC = HERE if (HERE / "voice.py").exists() else HERE / "src"
sys.path.insert(0, str(SRC))

from voice import VoiceAgent, TTS_VOICE  # noqa: E402
from config import OUT_DIR  # noqa: E402

SAMPLE_RATE = 16000
BYE = re.compile(r"\b(bye|goodbye|good bye|alvida|tata)\b|बाय|अलविदा", re.IGNORECASE)


def record(path: Path) -> bool:
    """Record from the mic between two Enter presses. Returns False if nothing was captured."""
    input("\nPress Enter and speak...")
    chunks: "queue.Queue[np.ndarray]" = queue.Queue()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        callback=lambda data, *_: chunks.put(data.copy())):
        input("Recording... press Enter to stop.")
    frames = [chunks.get() for _ in range(chunks.qsize())]
    if not frames:
        return False
    sf.write(path, np.concatenate(frames), SAMPLE_RATE, subtype="PCM_16")
    return True


def play(path: Path):
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        data, sr = sf.read(path)
        sd.play(data, sr)
        sd.wait()


def main():
    ap = argparse.ArgumentParser(description="Live mic call with the ZentroMart voice agent")
    ap.add_argument("--voice", default=TTS_VOICE, help="Gnani timbre-v2.5 voice (default Kaveri)")
    ap.add_argument("--out_dir", default=str(OUT_DIR / "live_calls"), help="parent folder for call recordings")
    args = ap.parse_args()

    call_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    call_dir.mkdir(parents=True, exist_ok=True)
    print(f"Live call started. Saving to {call_dir}\nSay 'bye' to end, or Ctrl+C to quit.")

    agent = VoiceAgent(args.voice)        # one agent = one conversation memory
    turn = 0
    with open(call_dir / "call.jsonl", "w", encoding="utf-8") as log:
        try:
            while True:
                turn += 1
                customer_wav = call_dir / f"turn{turn:02d}.customer.wav"
                if not record(customer_wav):
                    print("Nothing recorded, try again.")
                    turn -= 1
                    continue

                reply_wav = call_dir / f"turn{turn:02d}.reply.wav"
                rec = agent.turn(customer_wav, reply_wav)
                rec["file"] = customer_wav.name
                (call_dir / f"turn{turn:02d}.json").write_text(
                    json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()

                print(f"   heard : {rec['transcript']}\n   reply : {rec['reply']} ({rec['decision']})"
                      + (f"\n   ERROR : {rec['error']}" if rec["error"] else ""))
                if reply_wav.exists():
                    play(reply_wav)

                if BYE.search(rec["transcript"] or ""):
                    break
        except KeyboardInterrupt:
            print("\nCall stopped.")

    print(f"Call ended after {turn} turn(s). Files in {call_dir}")


if __name__ == "__main__":
    main()
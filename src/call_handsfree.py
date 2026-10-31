
import argparse
import io
import itertools
import queue
import re
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = Path(__file__).resolve().parent
SRC = HERE if (HERE / "voice.py").exists() else HERE / "src"     # works from VoiceOps/ or VoiceOps/src/
sys.path.insert(0, str(SRC))
from config import OUT_DIR  # noqa: E402
from voice import (VoiceAgent, TTS_VOICE, PRISMA_MAX_SECONDS, FILLERS, SMALL_TALK, RETRY_PROMPT,  # noqa: E402
                   for_speech, script_language, small_talk, is_unclear, join_wavs)

RATE = 16000
BLOCK = 480                              # 30 ms of audio per block
START_BLOCKS = 4                         # ~120 ms of voice starts a recording
END_SILENCE_BLOCKS = 23                  # ~0.7 s of silence ends it
PRE_ROLL_BLOCKS = 10                     # keep ~300 ms before speech started, so the first word is not cut
MIN_SPEECH_BLOCKS = 12                   # ignore blips shorter than ~0.36 s (coughs, clicks)
MAX_BLOCKS = int(PRISMA_MAX_SECONDS * RATE / BLOCK)
GOODBYE = re.compile(r"\b(bye|goodbye|good bye|alvida|bas itna hi)\b|अलविदा|बाय", re.I)
GREETING = "Namaste! You are connected to ZentroMart support. How can I help you today?"
FAREWELL = "Thank you for calling ZentroMart. Goodbye!"


def rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def to_wav(blocks):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(np.concatenate(blocks).tobytes())
    return buf.getvalue()


def play(wav_bytes):
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    sd.wait()


def calibrate(seconds=1.0):
    """Measure the room's background noise to set the speech threshold."""
    rec = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    levels = [rms(rec[i:i + BLOCK]) for i in range(0, len(rec) - BLOCK, BLOCK)]
    noise = float(np.median(levels)) if levels else 100.0
    return max(noise * 3.0, 350.0)


def listen(threshold):
    """Block until the caller says something; return it as WAV bytes (or None if it was only a blip)."""
    q, pre, frames = queue.Queue(), deque(maxlen=PRE_ROLL_BLOCKS), []
    voiced = silent = 0
    speaking = False
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK,
                        callback=lambda d, *_: q.put(d.copy())):
        while True:
            block = q.get()
            loud = rms(block) > threshold
            if not speaking:
                pre.append(block)
                voiced = voiced + 1 if loud else 0
                if voiced >= START_BLOCKS:
                    speaking, frames = True, list(pre)
                    print("  (hearing you...)", flush=True)
            else:
                frames.append(block)
                silent = 0 if loud else silent + 1
                if silent >= END_SILENCE_BLOCKS or len(frames) >= MAX_BLOCKS:
                    break
    if len(frames) - silent < MIN_SPEECH_BLOCKS:
        return None
    return to_wav(frames)


class PhraseCache:
    """Fixed phrases (greeting, fillers, small talk) are synthesised once in the background and replayed instantly."""

    def __init__(self, speech):
        self.speech, self.audio = speech, {}
        texts = [GREETING, FAREWELL] + [t for lang in ("en", "hi-en", "hi") for t in
                                        FILLERS[lang] + [SMALL_TALK["greet"][lang], SMALL_TALK["thanks"][lang],
                                                         RETRY_PROMPT[lang]]]
        threading.Thread(target=self._warm, args=(texts,), daemon=True).start()

    def _warm(self, texts):
        for t in texts:
            try:
                self.audio[t] = self.speech.speak(for_speech(t))[0]
            except Exception:
                pass

    def get(self, text, wait=False):
        if text not in self.audio and wait:
            self.audio[text] = self.speech.speak(for_speech(text))[0]
        return self.audio.get(text)


def main():
    ap = argparse.ArgumentParser(description="Hands-free voice call with the ZentroMart agent")
    ap.add_argument("--save", action="store_true", help="also save your audio and the replies in outputs/talk/")
    ap.add_argument("--voice", default=TTS_VOICE, help="Gnani timbre-v2.5 voice (default Kaveri)")
    ap.add_argument("--model", help="LLM model for this call only (e.g. a faster one); default: LLM_MODEL in .env")
    ap.add_argument("--no-filler", action="store_true", help="do not play 'one moment...' while thinking")
    ap.add_argument("--fast", action="store_true",
                    help="low hidden reasoning (gpt-oss: reasoning_effort=low; Nemotron/Qwen3: thinking off)")
    args = ap.parse_args()
    import agent as call_tools
    if args.model:
        call_tools.LLM_MODEL = args.model
    if args.fast:
        call_tools.REASONING = "low"
    print(f"LLM: {call_tools.LLM_MODEL}" + (" (low reasoning)" if args.fast else ""))
    save_dir = OUT_DIR / "talk"
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    va = VoiceAgent(args.voice, fast=True)
    cache = PhraseCache(va.speech)
    fillers = {lang: itertools.cycle(v) for lang, v in FILLERS.items()}
    print("Measuring background noise, stay quiet for a second...")
    threshold = calibrate()
    print(f"Ready (speech threshold {threshold:.0f}). Talk normally; say 'bye' or press Ctrl+C to end.\n")
    try:
        print(f"AGENT: {GREETING}")
        play(cache.get(GREETING, wait=True))
    except Exception as e:
        print(f"  (greeting skipped: {type(e).__name__}: {str(e)[:120]})")

    turn = 0
    try:
        while True:
            print("Listening...", flush=True)
            audio = listen(threshold)
            if audio is None:
                continue
            turn += 1
            heard = va.hear(audio)
            text = heard["transcript"]
            print(f"YOU  : {text}")
            if GOODBYE.search(text or "") and len(re.findall(r"\w+", text)) <= 4:   # "bye", "ok bye", "thanks bye"
                break
            quick = small_talk(text) or is_unclear(text)
            filler_on = False
            if not quick and not args.no_filler:                 # say something at once while the agent thinks
                f = cache.get(next(fillers[script_language(text)]))
                if f:
                    data, sr = sf.read(io.BytesIO(f), dtype="float32")
                    sd.play(data, sr)
                    filler_on = True
            out = va.think(text)
            reply = out.get("reply", "")
            print(f"AGENT: {reply}   [{out.get('decision')}]")
            if heard["error"]:
                print("  ERROR:", heard["error"])
            if filler_on:
                sd.wait()
            t_speak, first, parts = time.time(), None, []
            try:
                cached = cache.get(reply) if quick else None
                for wav_bytes in ([cached] if cached else va.speech.speak_stream(for_speech(reply))):
                    first = first if first is not None else round(time.time() - t_speak, 1)
                    parts.append(wav_bytes)
                    play(wav_bytes)                              # mic is closed while the agent speaks
            except Exception as e:
                print(f"  TTS ERROR: {type(e).__name__}: {str(e)[:150]}")
            print(f"  [hear {heard['stt_s']}s | think {out.get('agent_s')}s | first audio {first}s]")
            if args.save:
                stamp = time.strftime("%H%M%S")
                (save_dir / f"{stamp}_t{turn}_customer.wav").write_bytes(audio)
                if parts:
                    (save_dir / f"{stamp}_t{turn}_reply.wav").write_bytes(join_wavs(parts))
            if GOODBYE.search(text or ""):                       # "...that is all, bye": answered, then end
                break
    except KeyboardInterrupt:
        print()
    try:
        print(f"AGENT: {FAREWELL}")
        play(cache.get(FAREWELL, wait=True))
    except Exception:
        pass


if __name__ == "__main__":
    main()

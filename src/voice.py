"""Step 4: the voice layer around the Step 3 agent. Library only (used by run.py, web_app.py and the call_*.py scripts).

  audio --> Speech.transcribe (Gnani STT) --> VoiceAgent.think (Step 3 Agent) --> for_speech --> Speech.speak (Gnani TTS)

  Speech           transcribe(audio)  prisma-v3.0 "indic-auto" detects Hindi / English / Hinglish; falls back to en-IN
                   speak(text)        reply audio (WAV bytes); speak_stream(text) yields it sentence by sentence
  VoiceAgent       hear(audio) -> think(transcript) -> turn(audio, reply_wav) = hear + think + speak, one call's memory
                   fast=True (live calls): instant small talk, KB searched before the LLM, short phone-call answers
  for_speech(text) reply rewritten for the ear: order IDs in groups of 4, rupees spoken, short ticket refs
  helpers          script_language, small_talk, is_unclear, WAV split/join, chunk_texts (sections used)

Only Gnani is used for speech (capstone rule). GNANI_API_KEY is read from .env.
"""
import io
import os
import re
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config  # loads .env into the environment, incl. GNANI_API_KEY
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient, AudioConfig

from agent import Agent
from rag_pipeline import chunk_kb

assert config.ROOT                                       # (config import is needed for .env)
STT_MODEL = os.getenv("STT_MODEL", "prisma-v3.0")        # multilingual model with language auto-detection
STT_FALLBACK_LANGUAGE = os.getenv("STT_LANGUAGE", "en-IN")
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.getenv("TTS_VOICE", "Kaveri")
TTS_SAMPLE_RATE = 24000
PRISMA_MAX_SECONDS = 28                                  # prisma-v3.0 accepts up to 30 s per request
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

STT_TOPIC = ("Customer support call to ZentroMart, an Indian online store for groceries and electronics: orders, "
             "order IDs, delivery, returns, refunds, cancellations, warranty, verification with phone digits or "
             "pincode. The caller may speak English, Hindi or Hinglish.")
STT_KEYWORDS = ["ZentroMart", "refund", "return", "pincode", "order ID", "cancel", "warranty", "Assured", "UPI",
                "ZentroMart Plus", "Express", "reverse pickup", "wallet", "COD"]

HINGLISH_WORDS = {"hai", "hain", "kya", "aap", "aapka", "aapko", "mera", "meri", "mujhe", "nahi", "nahin", "kar",
                  "karein", "kijiye", "kripya", "ka", "ki", "ke", "ko", "mein", "hoon", "tha", "thi", "raha", "rahi",
                  "sakta", "sakti", "chahiye", "kab", "kitne", "din", "paisa", "wapas", "jaldi", "bhi", "aur"}

# "Sorry, I did not catch that" in the customer's language, for empty or unclear audio
RETRY_PROMPT = {
    "hi": "माफ़ कीजिए, मैं आपकी बात ठीक से सुन नहीं पाई। क्या आप दोबारा बोल सकते हैं?",
    "hi-en": "Maaf kijiye, main aapki baat theek se sun nahi paayi. Kya aap dobara bol sakte hain?",
    "en": "Sorry, I did not catch that. Could you please say it again?",
}
ERROR_REPLY = {
    "hi": "माफ़ कीजिए, अभी तकनीकी समस्या है। मैं आपको हमारे सहयोगी से जोड़ती हूँ।",
    "hi-en": "Maaf kijiye, abhi technical problem hai. Main aapko hamare colleague se connect karti hoon.",
    "en": "Sorry, I am having a technical problem right now. Let me connect you to a human colleague.",
}

# Extra rules for live phone calls (appended to the agent's system prompt in fast mode)
VOICE_NOTE = """

LIVE PHONE CALL (these rules apply on top of everything above):
- Keep every reply to 2 short sentences, 3 at most. Say the direct answer first with its key figure or date.
  Leave out details the caller did not ask for (item prices, coupons, payment history, policy names).
- If no document covers the question: say you do not have that information and ASK whether they would like
  to be connected to a human colleague. decision="answer". Do not create a ticket unless they say yes or ask
  for a human.
- At most one ticket per call: if one already exists, give its ID instead of creating another."""

# Small talk answered instantly without the LLM (only when the WHOLE utterance is a greeting or a thank-you)
_GREET = re.compile(r"^(hi+|hello+|hey+|hii+|namaste|namaskar|good (morning|afternoon|evening)|hello there|"
                    r"hi there|नमस्ते|नमस्कार|हेलो|हाय)( (ji|there|zentromart))?$")
_THANKS = re.compile(r"^(ok(ay)? )?(thanks?|thank you( so much| very much)?|shukriya|dhanyavaa?d|dhanyawaa?d|"
                     r"धन्यवाद|शुक्रिया)( (ji|a lot))?$")
_HINDI_LATIN = re.compile(r"\b(namaste|namaskar|shukriya|dhanyava+d|dhanyawa+d|ji)\b")
SMALL_TALK = {
    "greet": {"en": "Hello! I am ZentroMart's AI assistant. How can I help you today?",
              "hi-en": "Namaste! Main ZentroMart ki AI assistant hoon. Main aapki kya madad kar sakti hoon?",
              "hi": "नमस्ते! मैं ZentroMart की AI सहायक हूँ। मैं आपकी क्या मदद कर सकती हूँ?"},
    "thanks": {"en": "You are welcome! Is there anything else I can help you with?",
               "hi-en": "Aapka swagat hai! Kya main aur kuch madad kar sakti hoon?",
               "hi": "आपका स्वागत है! क्या मैं और कुछ मदद कर सकती हूँ?"},
}
FILLERS = {"en": ["One moment, let me check that for you.", "Sure, give me a second."],
           "hi-en": ["Ek second, main check karti hoon.", "Ji, abhi dekhti hoon."],
           "hi": ["एक पल, मैं देखती हूँ।", "जी, अभी देखती हूँ।"]}


def small_talk(transcript):
    """('greet' | 'thanks', language) if the whole utterance is small talk, else None."""
    t = re.sub(r"[^\w\s\u0900-\u097F]", " ", (transcript or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    kind = "greet" if _GREET.match(t) else "thanks" if _THANKS.match(t) else None
    if not kind:
        return None
    lang = script_language(t)
    return kind, ("hi-en" if lang == "en" and _HINDI_LATIN.search(t) else lang)

# ------------------------------------------------------------------ language helpers


def script_language(text):
    """'hi' for Devanagari, 'hi-en' for Hindi in Latin script, 'en' otherwise."""
    if re.search(r"[ऀ-ॿ]", text or ""):
        return "hi"
    words = re.findall(r"[a-z]+", (text or "").lower())
    hits = sum(w in HINGLISH_WORDS for w in words)
    return "hi-en" if hits >= 2 or (words and hits / len(words) >= 0.2) else "en"


TTS_LANGUAGE = {"hi": "hi-IN", "hi-en": "auto", "en": "en-IN"}

# ------------------------------------------------------------------ text for the ear (STY-3 + brief)


def for_speech(text):
    """Rewrite a reply so TTS reads it clearly: no markdown, IDs in groups, rupees spoken, short ticket refs."""
    lang = script_language(text)
    t = re.sub(r"[*_#`|>]+", " ", text or "")                              # markdown
    t = re.sub(r"https?://\S+", "", t)
    def _ticket(m):                                                         # long ticket IDs: last 4 characters
        tail = " ".join(m.group(1)[-4:].upper())
        near = m.string[max(0, m.start() - 30):m.start()].lower()
        if "ticket" in near or "टिकट" in near:                             # "ticket (ID is) TKT..." already said
            return ("जिसके आख़िरी अक्षर " if lang == "hi" else "ending in ") + tail
        return ("टिकट जिसके आख़िरी अक्षर " if lang == "hi" else "ticket ending in ") + tail
    t = re.sub(r"\bTKT([0-9a-fA-F]{6,})\b", _ticket, t)
    t = re.sub(r"(?<![\d,.])(\d{8,16})(?![\d,.])",                          # order IDs / RRNs in groups of 4
               lambda m: " ".join(m.group(1)[i:i + 4] for i in range(0, len(m.group(1)), 4)), t)
    rupees = {"hi": " रुपये", "hi-en": " rupaye", "en": " rupees"}[lang]
    t = re.sub(r"₹\s?([\d,]+(?:\.\d+)?)", lambda m: m.group(1) + rupees, t)
    t = re.sub(r"\bRs\.?\s?([\d,]+)", lambda m: m.group(1) + rupees, t)
    t = t.replace(" ", " ").replace("‑", "-")
    return re.sub(r"\s+", " ", t).strip()

# ------------------------------------------------------------------ audio helpers (stdlib only)


def _wav_source(audio):
    """A path or WAV bytes -> something wave / the Gnani SDK can open."""
    if isinstance(audio, (bytes, bytearray)):
        buf = io.BytesIO(audio)
        buf.name = "turn.wav"
        return buf
    return str(audio)


def wav_seconds(path):
    try:
        with wave.open(_wav_source(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None                                       # not a WAV (mp3/m4a...): duration unknown


def split_wav(path, max_seconds):
    """Split a WAV into in-memory WAV chunks of at most max_seconds."""
    chunks = []
    with wave.open(_wav_source(path), "rb") as w:
        params, rate = w.getparams(), w.getframerate()
        step = int(max_seconds * rate)
        while True:
            frames = w.readframes(step)
            if not frames:
                break
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setparams(params)
                out.writeframes(frames)
            buf.seek(0)
            buf.name = "chunk.wav"
            chunks.append(buf)
    return chunks


def _split_wav_bytes(b):
    """(fmt chunk, PCM data) from WAV bytes. Handles streamed WAVs whose header has an unknown/huge size
    (Gnani TTS sometimes writes 0xFFFFFFFF), which Python's wave module cannot read."""
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    i, fmt = 12, None
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        start = i + 8
        if cid == b"fmt ":
            fmt = b[start:start + size]
        elif cid == b"data":
            end = len(b) if size in (0, 0xFFFFFFFF) or start + size > len(b) else start + size
            return fmt, b[start:end]
        i = start + size + (size & 1)
    raise ValueError("no data chunk")


def _make_wav(fmt, pcm):
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(pcm)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm)


def fix_wav(b):
    """Rewrite a WAV with a correct header (safe to save and to play anywhere)."""
    fmt, pcm = _split_wav_bytes(b)
    return _make_wav(fmt, pcm)


def join_wavs(parts):
    """Concatenate WAV byte strings with identical audio format (also works with streamed WAV headers)."""
    pieces = [_split_wav_bytes(p) for p in parts]
    return _make_wav(pieces[0][0], b"".join(pcm for _, pcm in pieces))


def sentence_chunks(text, max_chars=400):
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?।])\s+", text):
        if cur and len(cur) + len(s) > max_chars:
            parts.append(cur.strip()); cur = ""
        cur += s + " "
    if cur.strip():
        parts.append(cur.strip())
    return parts or [text]



class Speech:
    def __init__(self, voice=TTS_VOICE):
        self.stt = GnaniSTTClient()
        self.tts = GnaniTTSClient()
        self.voice = voice
        self.prisma_ok = STT_MODEL == "prisma-v3.0"

    def _prisma(self, audio, previous_text=None):
        r = self.stt.transcribe(audio, language_code="indic-auto", model="prisma-v3.0",
                                language_candidates=["hi-IN", "en-IN"], format="transcribe",
                                keywords=STT_KEYWORDS, topic=STT_TOPIC, previous_text=previous_text or None)
        return r.get("transcript", "") or "", r.get("detected_language_code")

    def transcribe(self, path, previous_text=None):
        """path: an audio file path, or WAV bytes from the microphone.
        Returns (transcript, detected_language_code, stt_model)."""
        if self.prisma_ok:
            try:
                secs = wav_seconds(path)
                if secs and secs > PRISMA_MAX_SECONDS:                # long WAV: send in pieces, keep context
                    texts, lang, prev = [], None, previous_text
                    for piece in split_wav(path, PRISMA_MAX_SECONDS):
                        t, lang = self._prisma(piece, prev)
                        texts.append(t); prev = t
                    return " ".join(texts).strip(), lang, "prisma-v3.0"
                text, lang = self._prisma(_wav_source(path), previous_text)
                return text.strip(), lang, "prisma-v3.0"
            except Exception as e:
                status = getattr(e, "status_code", None)
                permanent = isinstance(status, int) and 400 <= status < 500 and status != 429
                print(f"  [STT] prisma-v3.0 failed ({type(e).__name__} {status or ''}); using default model "
                      + ("from now on" if permanent else "for this turn"))
                if permanent:                                     # e.g. model not enabled for this account
                    self.prisma_ok = False
        r = self.stt.transcribe(_wav_source(path), language_code=STT_FALLBACK_LANGUAGE, format="transcribe")
        return (r.get("transcript", "") or "").strip(), STT_FALLBACK_LANGUAGE, "default"

    def speak(self, text):
        """Text -> (WAV bytes, tts_language). Nothing is written to disk."""
        lang = TTS_LANGUAGE[script_language(text)]
        cfg = AudioConfig(sample_rate=TTS_SAMPLE_RATE, container="wav", encoding="linear_pcm")
        parts = [self.tts.synthesize(p, voice=self.voice, model=TTS_MODEL, language=lang, audio_config=cfg)
                 for p in sentence_chunks(text)]
        return join_wavs(parts), lang

    def speak_stream(self, text, max_chars=160):
        """Yield WAV bytes sentence group by sentence group, in order. All groups are synthesised in parallel,
        so the caller can start playing the first one while the others are still being generated."""
        lang = TTS_LANGUAGE[script_language(text)]
        cfg = AudioConfig(sample_rate=TTS_SAMPLE_RATE, container="wav", encoding="linear_pcm")
        parts = sentence_chunks(text, max_chars=max_chars)
        with ThreadPoolExecutor(max_workers=min(4, len(parts))) as pool:
            futures = [pool.submit(self.tts.synthesize, p, voice=self.voice, model=TTS_MODEL, language=lang,
                                   audio_config=cfg) for p in parts]
            for f in futures:
                yield fix_wav(f.result())

# ------------------------------------------------------------------ voice agent


_CHUNKS = {}


def chunk_texts(ids):
    if not _CHUNKS:
        for c in chunk_kb():
            _CHUNKS.setdefault(c["id"], {"id": c["id"], "title": c["title"], "source": c["file"], "text": c["text"]})
    return [_CHUNKS[i] for i in ids if i in _CHUNKS]


FILLER_SOUNDS = {"um", "umm", "uh", "uhh", "hmm", "hm", "mm", "ah", "aah", "er", "erm", "eh", "huh"}


def is_unclear(transcript):
    """Nothing usable was heard: empty, or only filler sounds. One real word ("yes", "haan", "okay") is NOT
    unclear: it can be an answer to the agent's question."""
    words = [w.lower() for w in re.findall(r"\w+", transcript or "")]
    return not words or all(w in FILLER_SOUNDS for w in words)


class VoiceAgent:
    """One phone call: keeps the Step 3 agent's memory across turns."""

    def __init__(self, voice=TTS_VOICE, fast=False):
        """fast=True (live calls): instant small talk, KB searched before the first LLM call, 8 sections."""
        self.fast = fast
        self.agent = (Agent(prefetch=True, search_k=8, style_note=VOICE_NOTE, one_ticket=True) if fast
                      else Agent())
        self.speech = Speech(voice)
        self.last_lang = "en"
        self.last_transcript = None
        self.last_reply_wav = None          # WAV bytes of the last spoken reply (live calls play this)

    def respond_text(self, transcript):
        """The brain: transcript -> agent result (same Agent as Step 3)."""
        st = small_talk(transcript) if self.fast else None          # checked first: "Hi" is one word but clear
        if not st and is_unclear(transcript):
            return {"decision": "clarify", "reply": RETRY_PROMPT[self.last_lang], "used_sections": [],
                    "retrieved": [], "tool_calls": [], "unclear_audio": True}
        if st:                                                    # instant reply, kept in the agent's memory
            kind, self.last_lang = st
            reply = SMALL_TALK[kind][self.last_lang]
            self.agent.turn_no += 1                               # so the agent does not re-introduce itself
            self.agent.history += [{"role": "user", "content": transcript}, {"role": "assistant", "content": reply}]
            return {"decision": "answer", "reply": reply, "used_sections": [], "retrieved": [], "tool_calls": [],
                    "small_talk": kind}
        self.last_lang = script_language(transcript)
        out = self.agent.turn(transcript)
        if not out.get("llm_ok", True) or not out.get("reply"):
            out = {**out, "decision": "escalate", "reply": ERROR_REPLY[self.last_lang]}
        return out

    def hear(self, audio):
        """Audio (path or WAV bytes) -> dict(transcript, stt_language, stt_model, error, stt_s)."""
        t0, rec = time.time(), {"error": None}
        try:
            transcript, lang, model = self.speech.transcribe(audio, self.last_transcript)
        except Exception as e:
            transcript, lang, model = "", None, None
            rec["error"] = f"STT {type(e).__name__}: {str(e)[:200]}"
        self.last_transcript = transcript or self.last_transcript
        rec.update(transcript=transcript, stt_language=lang, stt_model=model, stt_s=round(time.time() - t0, 1))
        return rec

    def think(self, transcript):
        """Transcript -> agent result plus agent_s (time spent in the LLM / tools)."""
        t0 = time.time()
        out = self.respond_text(transcript)
        return {**out, "agent_s": round(time.time() - t0, 1)}

    def turn(self, audio_path, reply_wav=None, closing=None):
        """One full voice turn = hear() + think() + speak. audio_path: file path or WAV bytes.
        reply_wav: where to save the reply audio (run.py), or None to keep it in self.last_reply_wav (live calls).
        closing: optional (regex, goodbye_text). If the caller's words match the regex (e.g. "bye"), the agent
        says goodbye_text and the result has call_ended=True, without calling the LLM."""
        t0 = time.time()
        rec = {"audio": "<microphone>" if isinstance(audio_path, (bytes, bytearray)) else str(audio_path),
               "reply_audio": None, **self.hear(audio_path)}
        if closing and closing[0].search(rec["transcript"] or ""):
            out = {"decision": "answer", "reply": closing[1], "used_sections": [], "retrieved": [],
                   "tool_calls": [], "call_ended": True}
        else:
            out = self.think(rec["transcript"])
        spoken = for_speech(out.get("reply", ""))
        self.last_reply_wav = None
        try:
            self.last_reply_wav, rec["tts_language"] = self.speech.speak(spoken)
            if reply_wav:
                Path(reply_wav).parent.mkdir(parents=True, exist_ok=True)
                Path(reply_wav).write_bytes(self.last_reply_wav)
                rec["reply_audio"] = str(reply_wav)
        except Exception as e:
            rec["error"] = (rec["error"] + " | " if rec["error"] else "") + f"TTS {type(e).__name__}: {str(e)[:200]}"
        used = list(out.get("used_sections") or [])
        kb_ids = [i for i in out.get("retrieved", []) if not i.startswith("products.csv")]
        rec.update(call_ended=bool(out.get("call_ended")),
                   reply=out.get("reply", ""), spoken_text=spoken, decision=out.get("decision"),
                   unclear_audio=bool(out.get("unclear_audio")), used_sections=used,
                   retrieved=list(out.get("retrieved") or []), chunks=chunk_texts(used or kb_ids[:5]),
                   tool_calls=[{k: v for k, v in c.items()
                                if k in ("name", "query", "order_id", "verified", "ticket_id", "reason")}
                               for c in out.get("tool_calls", [])],
                   seconds=round(time.time() - t0, 1))
        return rec

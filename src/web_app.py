"""Web interface for the ZentroMart voice agent: talk in the browser, hear the reply, say "bye" to end.

    python run.py            # from the repository root: opens http://127.0.0.1:7860

Record with the mic button; when you stop, the agent transcribes, answers and plays its reply.
Saying bye / goodbye / alvida / बाय / अलविदा ends the call. "New call" starts a fresh conversation.

Each call is saved to outputs/live_calls/<timestamp>/ :
    turn01.customer.wav, turn01.reply.wav, turn01.json, call.jsonl
"""
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
SRC = HERE if (HERE / "voice.py").exists() else HERE / "src"
sys.path.insert(0, str(SRC))

from voice import VoiceAgent, TTS_VOICE  # noqa: E402
from config import OUT_DIR  # noqa: E402

CALLS_DIR = OUT_DIR / "live_calls"
BYE = re.compile(r"\b(bye|goodbye|good bye|alvida|tata)\b|बाय|अलविदा", re.IGNORECASE)


GREETING = "Namaste! I am ZentroMart's AI support assistant. How can I help you today?"


def new_call():
    """Start a call: fresh agent memory, then the agent speaks its greeting."""
    call_dir = CALLS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    call_dir.mkdir(parents=True, exist_ok=True)
    va = VoiceAgent(TTS_VOICE, fast=True)                     # live-call mode: short replies, faster search
    va.agent.turn_no += 1                                     # greeting already says it is an AI assistant
    va.agent.history.append({"role": "assistant", "content": GREETING})
    state = {"agent": va, "dir": call_dir, "turn": 0, "ended": False}
    greeting_wav, status = call_dir / "turn00.greeting.wav", f"Call started. Saving to {call_dir}"
    try:
        wav, _ = va.speech.speak(GREETING)
        greeting_wav.write_bytes(wav)
    except Exception as e:                                    # show the real TTS error, keep the call usable
        greeting_wav, status = None, status + f" | TTS ERROR: {type(e).__name__}: {str(e)[:200]}"
    history = [{"role": "assistant", "content": GREETING}]
    return state, history, str(greeting_wav) if greeting_wav else None, status


def on_audio(audio_path, state, history):
    if not audio_path:
        return state, history, None, gr.update(), "No audio recorded."
    if state is None or state["ended"]:
        state, history, _, _ = new_call()

    state["turn"] += 1
    n, call_dir = state["turn"], state["dir"]
    customer_wav = call_dir / f"turn{n:02d}.customer{Path(audio_path).suffix or '.wav'}"
    shutil.copy(audio_path, customer_wav)
    reply_wav = call_dir / f"turn{n:02d}.reply.wav"

    rec = state["agent"].turn(customer_wav, reply_wav)
    rec["file"] = customer_wav.name
    (call_dir / f"turn{n:02d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(call_dir / "call.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    history = history + [
        {"role": "user", "content": rec["transcript"] or "(could not hear)"},
        {"role": "assistant", "content": f"{rec['reply']}  [{rec['decision']}]"},
    ]
    status = f"Turn {n} saved in {call_dir}"
    if rec["error"]:
        status += f" | ERROR: {rec['error']}"
    if BYE.search(rec["transcript"] or ""):
        state["ended"] = True
        status += " | Call ended. Press 'New call' to start again."

    return state, history, str(reply_wav) if reply_wav.exists() else None, None, status


with gr.Blocks(title="ZentroMart Voice Support") as demo:
    gr.Markdown("## ZentroMart voice support\nRecord your question with the mic. Say **bye** to end the call.")
    state = gr.State(None)
    chat = gr.Chatbot(height=380, label="Conversation")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Your question")
    reply = gr.Audio(label="Agent reply", autoplay=True, interactive=False)
    status = gr.Markdown()
    new_btn = gr.Button("New call")

    mic.stop_recording(on_audio, [mic, state, chat], [state, chat, reply, mic, status])
    new_btn.click(new_call, None, [state, chat, reply, status])
    demo.load(new_call, None, [state, chat, reply, status])


def launch(port=7860, share=False):
    demo.launch(server_port=port, share=share, inbrowser=True)


if __name__ == "__main__":
    launch()
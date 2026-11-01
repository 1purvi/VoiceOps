"""Web interface for the ZentroMart voice agent, started by run.py.

    python run.py            # from the repository root: opens http://127.0.0.1:7860

How a call works
  1. When the page opens, a call starts and the agent says its greeting out loud.
     (If the browser blocks sound until you click, a "Start call" button appears; one click plays it.)
  2. Press record, ask your question, press stop. The agent answers in text and speech.
  3. Say "bye" (or goodbye / alvida / बाय / अलविदा): the agent says goodbye and the call ends.
     Or press "End call" at any time. "Start new call" begins a fresh conversation.

Each call is saved to outputs/live_calls/<timestamp>/ :
    turn00.greeting.wav, turn01.customer.wav, turn01.reply.wav, turn01.json, call.jsonl
"""
import base64
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
FAREWELL = "Thank you for calling ZentroMart. Goodbye!"

_greeting_wav = {}                       # voice -> WAV bytes, synthesised once per server run


def greeting_audio(va):
    if TTS_VOICE not in _greeting_wav:
        _greeting_wav[TTS_VOICE], _ = va.speech.speak(GREETING)
    return _greeting_wav[TTS_VOICE]


def greeting_html(wav_bytes, call_id):
    """Greeting player. The page script plays it as soon as it appears; if the browser blocks sound,
    it shows the Start call button instead (one click plays it)."""
    if not wav_bytes:
        return ""
    src = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode()
    return (f'<audio id="zm-greet" data-call="{call_id}" src="{src}" preload="auto"></audio>'
            '<div id="zm-start" style="display:none;justify-content:center;padding:12px 0">'
            '<button id="zm-start-btn" class="lg primary" style="padding:12px 28px;font-size:16px;'
            'border-radius:8px;border:none;background:#ea580c;color:white;cursor:pointer">'
            'Start call</button></div>')


def call_ui(active):
    """Show the mic and End call during a call; show Start new call after it ends."""
    return (gr.update(visible=active, value=None),      # mic
            gr.update(visible=active),                  # End call
            gr.update(visible=not active))              # Start new call


def new_call():
    """Start a call: fresh agent memory, then the agent greets the caller."""
    call_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    call_dir = CALLS_DIR / call_id
    call_dir.mkdir(parents=True, exist_ok=True)
    va = VoiceAgent(TTS_VOICE, fast=True)                     # live-call mode: short replies, faster search
    va.agent.turn_no += 1                                     # the greeting already says it is an AI assistant
    va.agent.history.append({"role": "assistant", "content": GREETING})
    state = {"agent": va, "dir": call_dir, "turn": 0, "ended": False}

    status, wav = "**Call in progress.** Press record, ask your question, then press stop.", None
    try:
        wav = greeting_audio(va)
        (call_dir / "turn00.greeting.wav").write_bytes(wav)
    except Exception as e:                                    # show the real TTS error, keep the call usable
        status += f"  \nTTS error: {type(e).__name__}: {str(e)[:200]}"
    history = [{"role": "assistant", "content": GREETING}]
    return (state, history, greeting_html(wav, call_id), None, status, *call_ui(True))


def end_call(state, history):
    """End the call from the button."""
    if state and not state["ended"]:
        state["ended"] = True
        history = history + [{"role": "assistant", "content": "*Call ended.*"}]
    return (state, history, "", None, "**Call ended.** Press *Start new call* to talk again.", *call_ui(False))


def on_audio(audio_path, state, history):
    """One turn: customer audio -> transcript -> agent -> spoken reply. 'Bye' ends the call."""
    if state is None or state["ended"]:
        return (state, history, None, "**Call ended.** Press *Start new call* to talk again.", *call_ui(False))
    if not audio_path:
        return (state, history, None, "No audio recorded. Press record and try again.", *call_ui(True))

    state["turn"] += 1
    n, call_dir = state["turn"], state["dir"]
    customer_wav = call_dir / f"turn{n:02d}.customer{Path(audio_path).suffix or '.wav'}"
    shutil.copy(audio_path, customer_wav)
    reply_wav = call_dir / f"turn{n:02d}.reply.wav"

    rec = state["agent"].turn(customer_wav, reply_wav, closing=(BYE, FAREWELL))
    rec["file"] = customer_wav.name
    (call_dir / f"turn{n:02d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(call_dir / "call.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    history = history + [{"role": "user", "content": rec["transcript"] or "(could not hear you)"},
                         {"role": "assistant", "content": rec["reply"]}]
    reply_audio = str(reply_wav) if reply_wav.exists() else None
    error = f"  \nError: {rec['error']}" if rec.get("error") else ""

    if rec.get("call_ended"):
        state["ended"] = True
        history = history + [{"role": "assistant", "content": "*Call ended.*"}]
        return (state, history, reply_audio,
                "**Call ended.** Press *Start new call* to talk again." + error, *call_ui(False))
    return (state, history, reply_audio, "**Call in progress.** Press record to ask your next question." + error,
            *call_ui(True))


# Plays the greeting as soon as it appears on the page. Browsers may block sound until the first click;
# then the Start call button is shown and one click plays the greeting.
PAGE_JS = """
<script>
(() => {
  const tryGreet = () => {
    const a = document.getElementById('zm-greet');
    if (!a || a.dataset.tried === a.dataset.call) return;
    a.dataset.tried = a.dataset.call;
    a.play().catch(() => {
      const s = document.getElementById('zm-start');
      if (s) s.style.display = 'flex';
    });
  };
  new MutationObserver(tryGreet).observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#zm-start-btn')) return;
    const a = document.getElementById('zm-greet');
    if (a) { a.currentTime = 0; a.play(); }
    document.getElementById('zm-start').style.display = 'none';
  });
})();
</script>
"""
STOP_AUDIO_JS = "() => { document.querySelectorAll('audio').forEach(a => a.pause()); }"

with gr.Blocks(title="ZentroMart Voice Support") as demo:
    gr.Markdown("## ZentroMart voice support\nAsk about orders, returns, refunds and deliveries. "
                "Say **bye** or press **End call** to finish.")
    state = gr.State(None)
    status = gr.Markdown()
    greet = gr.HTML()
    chat = gr.Chatbot(height=380, label="Conversation")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Your question (record, then stop)")
    reply = gr.Audio(label="Agent reply", autoplay=True, interactive=False)
    with gr.Row():
        end_btn = gr.Button("End call", variant="stop")
        new_btn = gr.Button("Start new call", variant="primary", visible=False)

    call_outputs = [state, chat, greet, reply, status, mic, end_btn, new_btn]
    demo.load(new_call, None, call_outputs)
    new_btn.click(new_call, None, call_outputs)
    end_btn.click(None, None, None, js=STOP_AUDIO_JS)                 # silence any audio still playing
    end_btn.click(end_call, [state, chat], call_outputs)
    mic.stop_recording(on_audio, [mic, state, chat], [state, chat, reply, status, mic, end_btn, new_btn])


def free_port(start, tries=20):
    """First free port from start upward (an earlier run.py may still hold 7860)."""
    import socket
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise OSError(f"No free port between {start} and {start + tries - 1}. Stop the other run.py and try again.")


def launch(port=7860, share=False):
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    free = free_port(port)
    if free != port:
        print(f"Port {port} is busy (an earlier run.py may still be running), using {free} instead.")
    print(f"Opening the call at http://127.0.0.1:{free}  (Ctrl+C to stop)")
    demo.launch(server_port=free, share=share, inbrowser=True, head=PAGE_JS,
                allowed_paths=[str(CALLS_DIR)])                  # lets the page play the saved reply audio


if __name__ == "__main__":
    launch()

# VoiceOps — ZentroMart Voice Support Agent

A voice customer-support agent for ZentroMart, a fictional Indian online store. A customer speaks a
question; the agent answers from the company's policy documents, looks up orders, hands over to a human
when needed, and replies with speech in English, Hindi or Hinglish.

```
customer audio → Gnani STT → agent (LLM + document search + tools) → Gnani TTS → spoken reply
```

## Tech stack

| Part | What is used |
| --- | --- |
| LLM | `nvidia/nemotron-3.5-super-vl-120b-a12b` through the NVIDIA API, called with the `openai` Python client (any OpenAI-compatible provider can be set in `.env`) |
| Embeddings | `intfloat/multilingual-e5-base` (sentence-transformers), handles English and Hindi |
| Keyword search | BM25 (`rank-bm25`) |
| Vector store | In-memory `numpy` matrix of embeddings, saved to disk with `pickle` (no external vector database) |
| Data | CSV files read with `pandas` (customers, orders, items, products, pincodes, holidays) |
| Speech | Gnani only: `prisma-v3.0` speech-to-text and `timbre-v2.5` text-to-speech (voice Kaveri), via `gnani-vachana` |
| Interface | Gradio web app; terminal mic calls with `sounddevice` |

## How it works

### 1. LLM and prompt
The LLM gets a system prompt that makes it ZentroMart's AI support assistant, sets today's date
(20 October 2026) and lists its rules. It must reply in JSON with:
`key_facts` (exact figures from the documents), `decision` (`answer`, `clarify`, `escalate` or `refuse`),
`reply` (what the customer hears) and `used_sections` (which document sections it relied on).

### 2. Chunking strategy
- Every policy document in `kb/` is split at its `[SECTION-ID]` headings, so one chunk = one policy section.
- A section longer than 1,500 characters is split again at paragraph breaks.
- Each chunk keeps its metadata: section ID, document title, document type (policy, agent guide, help, FAQ,
  marketing), status (current or superseded), effective date and language.
- Each product row in `products.csv` becomes its own chunk (title, price, return category, seller,
  warranty), so questions about a specific product find it.

### 3. Vector store and retrieval
- Chunks are embedded once with `multilingual-e5-base` and stored as a `numpy` matrix in
  `outputs/vector_index.pkl`; a BM25 keyword index is built beside it.
- For each question, the meaning-based search (embeddings) and the keyword search (BM25) are combined
  with reciprocal rank fusion.
- Older (superseded) policies and lower-priority documents are ranked slightly lower
  (policy > agent guide > help > FAQ > marketing). A product code in the question pulls that product's row.
- Only the top sections are sent to the LLM, and it may answer only from them. If they do not cover the
  question, it says it does not know and offers a human.

### 4. Conversation agent and tools
The agent runs a loop: the LLM decides which tool to call, sees the result, and repeats until it gives
its final answer. Tools:

| Tool | What it does |
| --- | --- |
| `search_kb` | searches the policy documents (the retrieval above) |
| `lookup_order` | looks up an order; shows details only after the caller gives the last 4 phone digits or the delivery pincode |
| `create_ticket` | hands the case to a human with a short summary |
| `date_calc` | counts business days (Mon–Sat, skipping holidays) or calendar days in Python |
| `final_answer` | returns the decision, reply and sections used |

The agent remembers the whole conversation (order ID, verified order, tickets), so follow-up questions
work without repeating details. Safety rules are enforced in code: at most two verification attempts,
the pincode is never shown to the LLM, one ticket per issue, and a handover is confirmed only after the
ticket is actually created.

### 5. Voice
- **Speech-to-text:** Gnani `prisma-v3.0` detects Hindi or English automatically and is given shop words
  (refund, pincode, order ID…) and the previous sentence as hints.
- **Text-to-speech:** Gnani `timbre-v2.5`; the speaking language follows the reply's script.
- Replies are rewritten for the ear: order numbers in groups of four, rupee amounts in words, dates with
  the year.
- If nothing clear is heard, the agent asks the caller to repeat. Greetings and thanks are answered
  instantly; in a live call, replies are kept to two or three short sentences.

## What each file does

```
run.py                     start here: runs Steps 1–3, then opens the call app (or runs a folder of audio)
src/
├── config.py              paths, settings and API keys loaded from .env
├── llm.py                 system prompt, LLM call with retries, JSON parsing
├── rag_pipeline.py        chunking, embeddings + BM25 index, hybrid retrieval, date table
├── agent.py               conversation agent: tool loop, memory, verification and ticket rules
├── voice.py               Gnani speech-to-text and text-to-speech around the agent
├── evaluate.py            runs Steps 1–3 and saves the results (called by run.py)
├── web_app.py             Gradio page: record a question, hear the reply, keep the call going
├── batch_run.py           answers every audio file in a folder and saves the results
├── call_handsfree.py      terminal call that starts and stops recording on your voice
├── call_push_to_talk.py   terminal call: press Enter to talk
├── metrics.py             checks answers against the expected facts in the practice files
├── eval_cache.py          runs and caches the practice questions
├── step1_baseline.ipynb   Step 1: the LLM alone, without documents
├── step2_rag.ipynb        Step 2: the LLM with document search
├── step3_conversations.py Step 3: runs the multi-turn practice conversations
└── data_validation.py     checks the dataset files before use
sample_outputs/            example replies and saved practice results
report.pdf                 project report
demo.mp4                   demo video
```

## Run

Everything starts from one command:

```
python run.py
```

It runs the whole project in order:

1. **Step 1:** the LLM alone answers `practice/questions.csv`, and the scores are saved.
2. **Step 2:** the LLM with document search answers the same questions, and the Step 1 vs Step 2 table is saved.
3. **Step 3:** the agent runs the multi-turn `practice/conversations.csv`, and the conversation results are saved.
4. **App:** the browser opens, a call starts and the agent says *"Namaste! I am ZentroMart's AI support
   assistant. How can I help you today?"* Record your question, listen to the reply, and say "bye" or press
   **End call** to finish.

All results are saved in `outputs/`. Every LLM answer is cached, so after the first full run Steps 1–3 only
re-score the saved answers and take a few seconds.

| Command | What it does |
| --- | --- |
| `python run.py` | Steps 1–3, then the app |
| `python run.py --fresh` | Deletes the cached answers and runs Steps 1–3 from scratch (slow), then the app |
| `python run.py --skip_eval` | Opens the app straight away |
| `python run.py --audio_dir path/to/audio_folder` | Answers every audio file in a folder; replies, audio and the document sections used are saved in `sample_outputs/run/` |

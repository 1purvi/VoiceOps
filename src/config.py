import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent           # .../VoiceOps
load_dotenv(ROOT / ".env")

DATASET_DIR = ROOT / "Capstone_Project_Dataset_FTTT_2026"
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = "2026-10-20" 
LLM_TPM = int(os.getenv("LLM_TPM", "0"))                                                      # old style: explicit URL / key / model in .env
    
EMBED_MODEL = os.getenv("EMBED_MODEL")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
TOP_K = int(os.getenv("TOP_K"))

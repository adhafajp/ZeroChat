import warnings

warnings.filterwarnings("ignore")

MODEL_BASE_PATH = "microsoft/phi-2"
ADAPTER_PATH = "adhafajp/phi2-qlora-zero-chat"

DEFAULT_SYSTEM = "You are Zero, a helpfull assistant."
PROMPT_FORMAT = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
MAX_NEW_TOKEN=256
DO_SAMPLE=False
REPETITION_PENALTY=1.1
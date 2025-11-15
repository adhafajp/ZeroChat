import warnings

warnings.filterwarnings("ignore")

MODEL_BASE_PATH = r"E:\AI\model-base\phi-2-2.7B"
ADAPTER_PATH = r"D:\a-it\AI\Fine Tune\alpaca-identitas-squadv2\adapter\result\phi-2-2.7B_qlora_alpaca-51.8k_identity-model-283_squadv2-15k-phase1f"

# MODEL_BASE_PATH = "microsoft/phi-2"
# ADAPTER_PATH = "adhafajp/phi2-qlora-zero-chat"

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
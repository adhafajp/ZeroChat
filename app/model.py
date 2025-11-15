from contextlib import asynccontextmanager
from fastapi import FastAPI
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from .config import MODEL_BASE_PATH, ADAPTER_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Startup Event: Starting Model Loading ---")

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading tokenizer from: {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH, trust_remote_code=True
    )

    print("Configuring ChatML special tokens...")
    special_tokens = {
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(
        f"Added {num_added} additional_special_tokens: {special_tokens['additional_special_tokens']}"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(
        f"Tokenizer check: EOS token is '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})"
    )
    print(
        f"Tokenizer check: PAD token is '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})"
    )
    print(f"Current Tokenizer dictionary size: {len(tokenizer)}")

    print(f"Loading base model from: {MODEL_BASE_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print(f"Adapting the model's token embeddings to the size: {len(tokenizer)}")
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id
    print("Embedding model successfully adjusted.")

    print(f"Applying the QLoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Adapter successfully applied.")
    model.config.use_cache = True
    model.eval()

    app.state.model = model
    app.state.tokenizer = tokenizer

    print("\n\nYour fine-tuned model (Phi-2 + Adapter) is ready to use.!")

    yield

    print("--- Shutdown Event: Cleaning the Model ---")
    del app.state.model
    del app.state.tokenizer
    torch.cuda.empty_cache()
import torch
import asyncio
from threading import Thread
from transformers import (
    TextIteratorStreamer, 
    PreTrainedModel, 
    PreTrainedTokenizer
)
from typing import AsyncIterator

from .config import (
    PROMPT_FORMAT, 
    DEFAULT_SYSTEM, 
    MAX_NEW_TOKEN, 
    REPETITION_PENALTY, 
    DO_SAMPLE
)

STOP_TOKENS = ["<|endoftext|>", "<|im_end|>"]

def _format_prompt(instruction: str) -> str:
    return PROMPT_FORMAT.format(
        system_prompt=DEFAULT_SYSTEM, instruction=instruction
    )

def _get_generation_kwargs(tokenizer, inputs, streamer=None) -> dict:
    return dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKEN,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=DO_SAMPLE,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        pad_token_id=tokenizer.pad_token_id,
    )


def generate_text_non_stream(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    text: str
) -> str:
    print(f"\nStarting generation (non-stream) for: '{text}'")
    device = next(model.parameters()).device
    
    prompt_text = _format_prompt(text)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_token_count = inputs["input_ids"].shape[1]

    generation_kwargs = _get_generation_kwargs(tokenizer, inputs)
    generation_kwargs.pop("streamer", None)

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    generated_tokens = outputs[0][prompt_token_count:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    cut_index = len(generated_text)
    for stop_token in STOP_TOKENS:
        if stop_token in generated_text:
            cut_index = min(cut_index, generated_text.index(stop_token))
    
    final_answer = generated_text[:cut_index].strip()
    print(f"Results (non-stream): {final_answer}")
    return final_answer

async def generate_text_stream(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    text: str
) -> AsyncIterator[str]:
    print(f"\nStarting generation (stream) for: '{text}'")
    device = next(model.parameters()).device
    
    prompt_text = _format_prompt(text)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )

    generation_kwargs = _get_generation_kwargs(tokenizer, inputs, streamer)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    try:
        while True:
            try:
                new_text = await asyncio.to_thread(streamer.__next__)
            except StopIteration:
                print("Streaming finished (streamer finished).")
                break

            stop_token_found = False
            clean_chunk = ""

            if any(tok in new_text for tok in STOP_TOKENS):
                stop_token_found = True
                for tok in STOP_TOKENS:
                    if tok in new_text:
                        new_text = new_text.split(tok)[0]
                        break
                clean_chunk = new_text
            else:
                clean_chunk = new_text

            if clean_chunk:
                yield clean_chunk

            if stop_token_found:
                print("Streaming completed (stop token found).")
                break
    
    except Exception as e:
        print(f"Error in stream generator: {e}")
        yield f"[ERROR] {e}"
    finally:
        if thread.is_alive():
            thread.join(timeout=1.0)
        print("The stream generator is closed.")
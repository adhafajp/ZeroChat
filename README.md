# 💬 ZeroChat — Full-Stack AI Assistant (Phi-2 2.7B + QLoRA)

This project showcases a complete **AI engineering workflow** — from **fine-tuning** to **experiment tracking** and **production-grade deployment** — built around a specialized assistant model called **"Zero"**, fine-tuned from [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2) using **QLoRA**.

The total committed compute time for fine-tuning exceeded **9 days** across multiple iterative phases, demonstrating a strong focus on *experimentation and resource optimization*.

---

## 🧩 Project Overview

**ZeroChat** demonstrates how to turn a large language model into a deployable AI assistant through a structured MLOps process, including:

- ✅ **Fine-tuning** using QLoRA (efficient low-VRAM adaptation)  
- ✅ **Experiment tracking** with Weights & Biases (W&B)  
- ✅ **FastAPI backend** for async streaming inference  
- ✅ **Modern web-based frontend** (HTML + TailwindCSS + JS)  
- ✅ **GitHub-hosted source** for open portfolio use  

The model adapter is hosted on Hugging Face:  
🔗 [**adhafajp/phi2-qlora-zero-chat**](https://huggingface.co/adhafajp/phi2-qlora-zero-chat)

---

## 🧠 Model Summary

| Component | Description |
|------------|-------------|
| **Base Model** | [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2) |
| **Fine-Tuning Method** | QLoRA (Quantized LoRA Fine-Tuning) |
| **Language** | English |
| **Precision** | 4-bit (NF4) |
| **Frameworks** | `transformers`, `peft`, `bitsandbytes`, `fastapi` |
| **Tracking** | Weights & Biases (W&B) |

### Dataset Composition
The model was fine-tuned on a curated combination of:
- 🧩 `yahma/alpaca-cleaned` → instruction-following  
- 📚 `rajpurkar/squad_v2` → reading comprehension and QA  
- 🧠 **`custom_persona` (283 samples)** → personal assistant identity. **Aggressively oversampled (×9)** to ensure the model retained its custom identity against the larger datasets.  

---

## 🧮 Training Phases

The fine tuning consist of multiple stage experiment

Stage 1:

| Phase | Summary | Runtime |
|--------|----------|----------|
| **1A** | Initial fine-tune (canceled due to overfitting) | 11h 50m |
| **1B** | Full 2-epoch fine-tune on Alpaca + SQuADv2 + persona | 5d 11h 50m |
| **1C** | Small re-train (underfit) | 19h |
| **1D / 1D-A / 1E** | Refinement attempts with packing & oversampling | ~3d total |
| **1F** | Final adapter re-train from **1B** (expanded persona dataset, balanced oversampling) | 1d 5h |

Stage 2:

After gathering all the insights from the initial experiments (1A-1F), fine-tuning was restarted completely from scratch. By applying all the lessons learned, this new training process achieved better and more balanced performance in just 1s 21h.
The adapter released in this repository is the result of this final, optimized training.

| Phase | Summary | Runtime |
|--------|----------|----------|
| **1** | Fine-tune again from scratch by applying all the insights from previous experiments. | 1d 21h |

> **Key Insight:** The final model is highly effective at **context extraction (RAG)** for its identity, proving that RAG is a more stable method for storing factual/identity information than relying solely on fine-tuning.

📊 **W&B Log (Phase 1F):** [wandb.ai/VoidNova/.../runs/bpju3d09](https://wandb.ai/VoidNova/phi-2-2.7B_qlora_alpaca-51.8k_identity-model-232_squadv2-15k/runs/bpju3d09?nw=nwuseradhafajp)
📊 **W&B Log (Final):** [wandb.ai/VoidNova/.../runs/rx5fih5v](https://wandb.ai/VoidNova/phi-2_qlora_ZeroChat/runs/rx5fih5v?nw=nwuseradhafajp)

---

## ⚙️ System Architecture

The project consists of two major parts:

### 1. **FastAPI Backend**
- Handles streaming responses from the fine-tuned Phi-2 + LoRA adapter.  
- Optimized for low-latency inference with async endpoints.  
- Simple environment-based configuration.

### 2. **Frontend (HTML + TailwindCSS + JS)**
- Lightweight browser chat UI using Server-Sent Events (SSE).  
- Built for responsiveness and minimal resource usage.  
- Compatible with local and cloud backends.

---

## 🧩 Local Deployment Example

Clone the repository and set up the environment:

```bash
git clone https://github.com/adhafajp/ZeroChat.git
cd ZeroChat
pip install -r requirements.txt

python run.py
```
Access the frontend locally at:
http://localhost:8000

---
🧠 Example Inference (Standalone Script)

You can also load the adapter directly with Python:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

adapter_path = "adhafajp/phi2-qlora-zero-chat"
base_model_path = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    quantization_config=bnb_config, 
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = """<|im_start|>system
You are Zero, a helpful assistant.<|im_end|>
<|im_start|>user
Explain what QLoRA is in simple terms.<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
---

## 📄 License

This project and the adapter are released under the **MIT License**.  
Copyright (c) 2025 **Adhafa JP**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

---

## 🙏 Acknowledgements

- **Base Model:** [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2)  
  Licensed under the **MIT License** © Microsoft (2023)

- **Core Libraries:**  
  [Transformers](https://github.com/huggingface/transformers),  
  [PEFT](https://github.com/huggingface/peft),  
  [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes),  
  [PyTorch](https://pytorch.org/), and [FastAPI](https://fastapi.tiangolo.com/)

- **Experiment Tracking:** [Weights & Biases (W&B)](https://wandb.ai) for model training and evaluation logs.

- **Author:** [@adhafajp](https://github.com/adhafajp)

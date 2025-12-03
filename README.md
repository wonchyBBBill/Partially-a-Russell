# Partially-a-Russell

This is a light work that try to recreate Russell's point about Hegel by simple fine-tuing.
```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
#download the llm to local
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./Qwen2-7B-Instruct
```

## There a two series of codes ---- two fine-tuned models. Two models are fine-tuned with Different datasets. BUT, both datasets were built on Russel_against_Hegel.docx. 
## The file Russel_against_Hegel.docx is the chapter on Hegel of Russell's A History of Western Philosophy. 
### 1. The Primary Model: Model fine-tuned with long, intense dialogue.
Run this model by:
```shell
python Run_the_Model.py
# This code will directly run the fine-tuned model. 
```

It's easy to do this fine-tuning step by step. 
```shell
# To create the jsonl file for fine-tuning:
python Summon_the_Dialogue.py # This code will produce russel_vs_hegel.jsonl with Russel_against_Hegel.docx
# Then fine-tune the model with LoRA, run:
python Fine_Tune.py # This code will produce the LoRA folder lora_russell_on_hegel_final.
# Finally you can try the fine-tuned model by:
python Run_the_Model.py
```

### 2. The Second Model

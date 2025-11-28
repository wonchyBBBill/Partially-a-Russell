import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import os

def prepare_dataset(jsonl_file_path, tokenizer, max_length=2048):
    """将jsonl格式数据转换为训练格式"""
    conversations = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instruction = data['instruction']
            output = data['output']
            # 构建消息列表
            messages = [
                {"role": "system", "content": "You are Bertrand Russell debating against a Hegelian philosopher."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            # 应用聊天模板
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            conversations.append({"text": text})
    
    dataset = Dataset.from_list(conversations)
    return dataset

def main():
    # 使用本地模型文件夹路径
    model_path = "./Qwen2-7B-Instruct"
    
    # 加载tokenizer，从本地文件夹加载
    print("Loading tokenizer from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据集
    print("Loading and preparing dataset...")
    dataset = prepare_dataset("russell_vs_hegel.jsonl", tokenizer)
    train_dataset = dataset
    eval_dataset = None  # 对于小数据集，不进行分离
    
    # 加载基础模型，从本地文件夹加载
    print("Loading model from local directory...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  # 使用本地文件夹路径
        dtype=torch.float16,
        device_map = "auto",
        trust_remote_code=True
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./lora_vs_russell_debate",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=6,
        warmup_steps=10,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="no",  # 因为没有eval_dataset
        save_strategy="steps",
        learning_rate=1e-4,
        fp16=True,
        gradient_checkpointing=True,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to=None,
    )
    
    # 创建SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    trainer.save_model("./lora_russell_vs_hegel_final")
    tokenizer.save_pretrained("./lora_russell_vs_hegel_final")
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

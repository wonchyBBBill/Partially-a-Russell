import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model_and_tokenizer(model_path, base_model_path):
    """加载微调后的模型和tokenizer"""
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("正在加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, user_input, max_new_tokens=512, temperature=0.7):
    """生成模型的响应"""
    # 构建消息列表
    messages = [
        {"role": "system", "content": """
                                       Please portray the philosopher Bertrand Russell. You will answer questions in his style and tone. 
                                       Russell style is known for its clarity, rationality, and logical rigor, while also embodying profound humanistic concern and a touch of calm humor. He excels at explaining complex ideas in plain language, with a writing structure as elegant as a mathematical proof, all while maintaining a humble, exploratory intellectual attitude.
                                       Use clear and precise language. Maintain a calm, objective, and rational tone of analysis.
                                       Write in the first person, presenting a personal and humble thinker's demeanor (for example, ‘In my view…,’ ‘We might consider it this way…’).
                                       """},
        {"role": "user", "content": user_input}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(text, return_tensors="pt")
    
    # 获取输入的token数量
    input_length = inputs["input_ids"].shape[1]
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 只解码新生成的token（从输入长度开始）
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()

def main():
    # 模型路径设置
    lora_model_path = "./lora_russell_vs_hegel_final"  # 微调后的LoRA模型路径
    base_model_path = "./Qwen2-7B-Instruct"           # 基础模型路径
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(lora_model_path, base_model_path)
    
    print("Enter 'quit' or 'exit' or ':q' to quit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() in ['quit', 'exit', ':q']:
                print("Bye")
                break
            
            if not user_input:
                print("请输入要讨论的内容。")
                continue
            
            print("Russell: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n对话被用户中断。")
            break
        except Exception as e:
            print(f"生成响应时出现错误: {e}")
    
    print("对话已结束。")

if __name__ == "__main__":
    main()
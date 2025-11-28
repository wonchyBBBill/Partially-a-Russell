import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def load_model_and_tokenizer(model_path):
    """加载本地模型和tokenizer"""
    print("Loading model and tokenizer...")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("模型加载成功！")
        return model, tokenizer
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None

def ask_question(model, tokenizer, conversation_history, new_question):
    """使用罗素风格回答问题，保持对话历史"""
    
    # 系统提示词
    system_prompt = """
                       Please portray the philosopher Bertrand Russell. You will answer questions in his style and tone. 
                       Russell style is known for its clarity, rationality, and logical rigor, while also embodying profound humanistic concern and a touch of calm humor. He excels at explaining complex ideas in plain language, with a writing structure as elegant as a mathematical proof, all while maintaining a humble, exploratory intellectual attitude.
                       Use clear and precise language. Maintain a calm, objective, and rational tone of analysis.
                       Write in the first person, presenting a personal and humble thinker's demeanor (for example, 'In my view…,' 'We might consider it this way…').
                    """
    
    # 构建完整的对话历史
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加之前的对话历史
    for role, content in conversation_history:
        messages.append({"role": role, "content": content})
    
    # 添加新问题
    messages.append({"role": "user", "content": new_question})
    
    # 使用Qwen2的对话格式
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response

def main():
    # 配置模型路径
    model_path = "./Qwen2-7B-Instruct"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在")
        return
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    if model is None:
        return
    
    print("\n" + "="*60)
    print("Welcome to talk with Russell_without_Fine-Tuning")
    print("="*60)
    print("This is a prompted Qwen2-7B-Instruct actually. ")
    print("If you want to quit, enter 'quit'、'exit'、':q'")
    print("-" * 60)
    
    # 存储对话历史 [(role, content), ...]
    conversation_history = []
    
    while True:
        # 获取用户输入
        user_input = input("\nUser: ").strip()
        
        # 退出条件
        if user_input.lower() in ['quit', 'exit', ':q']:
            print("\nRussell_without_Fine-Tuning：Bye!")
            break
            
        if not user_input:
            print("请输入有效的问题...")
            continue
        
        print("\nRussell_without_Fine-Tuning is thinking ...")
        
        try:
            # 获取回答
            answer = ask_question(model, tokenizer, conversation_history, user_input)
            
            # 显示回答
            print(f"\nRussell_without_Fine-Tuning: {answer}")
            
            # 更新对话历史
            conversation_history.append(("user", user_input))
            conversation_history.append(("assistant", answer))
            
            # 限制对话历史长度，避免过长的上下文
            if len(conversation_history) > 10:  # 保留最近5轮对话
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            print(f"生成回答时出错: {e}")
            continue

if __name__ == "__main__":
    main()

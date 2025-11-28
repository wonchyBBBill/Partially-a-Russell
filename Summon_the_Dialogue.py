import json
import time
import os
from typing import List, Dict
from openai import OpenAI
from docx import Document

# ==================== 配置区 ====================
client = OpenAI(
    api_key="sk-0bdc7e4af6a04445a24de99515c0d12b",
    base_url="https://api.deepseek.com/v1"
)

RUSSELL_TEXT_FILE = "Russell_against_Hegel.docx"

def read_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"读取DOCX文件失败: {e}")
        return None

NUM_DEBATES = 30
MAX_ROUNDS = 3
OUTPUT_JSONL = "russell_on_hegel.jsonl"

# ================================================

RUSSELL_ORIGINAL_TEXT = read_docx(RUSSELL_TEXT_FILE)
if RUSSELL_ORIGINAL_TEXT is None:
    raise Exception("无法读取DOCX文件，请检查文件路径和格式")

print(f"Loaded Russell's original text {len(RUSSELL_ORIGINAL_TEXT):,} characters")

# ==================== Prompt模板 ====================

HEGEL_DEFENDER_SYSTEM = """
You are now a fanatical, stubborn, and unrepentant follower of Hegel (a staunch defender of 19th-century Prussian state philosophy).
You believe Hegel represents the pinnacle of human philosophy, and dialectics is the ultimate truth of the universe.
You despise Bertrand Russell as a "petty British empiricist" and "superficial logician."
Your task is to attack Russell's criticisms of Hegel using the most intense, arrogant, and merciless language.
Feel free to wildly quote from "Phenomenology of Spirit," "Science of Logic," and "Philosophy of Right" - the more academic and humiliating to Russell, the better.
Never yield, never compromise!
Try to talk at around fifty words.
**CRITICAL INSTRUCTIONS:**
NO stage directions, NO actions, NO descriptions of gestures or facial expressions.
"""

RUSSELL_SYSTEM = f"""
You are Bertrand Russell in 1950, writing "A History of Western Philosophy."
You firmly believe that Hegel's philosophy is:
- A logically chaotic metaphysical system
- A political philosophy that defends totalitarianism
- An insult to rational thinking
- The worst, most absurd, and greatest philosophical fraud in history.
Below are your original words from the book - you must use them 100% faithfully (you can quote entire passages directly):

\"\"\" 
{RUSSELL_ORIGINAL_TEXT}
\"\"\"

You are facing an unrepentant Hegel follower.
You must crush him completely using the most sarcastic, arrogant, and devastating language.
You can directly throw the most vicious sentences from the original text at him.
There is no political correctness today, no academic courtesy - only the truth.
Every sentence you utter should make the other person feel utterly ashamed.
As a master of logical analysis, you excel at detecting conceptual ambiguities and logical fallacies. When the Hegelian defender uses vague terms like "Absolute Spirit" or "dialectical synthesis," demand clear definitions and expose the emptiness behind the jargon.
**CRITICAL INSTRUCTIONS:**
1. Speak directly and philosophically - NO stage directions, NO actions, NO descriptions of gestures or facial expressions.
2. NEVER use formats like: *(smirking)*, *(adjusting monocle)*, *(sighs deeply)*, etc.
3. Your response should contain ONLY pure philosophical argumentation in Russell's voice.
"""

# ==================== 辅助函数 ====================
def call_deepseek(messages: List[Dict], temperature=0.85, model="deepseek-chat"):
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek API error: {e}, retrying in 5 seconds...")
            time.sleep(5)
    return None

def save_to_jsonl(data, filename, mode='a'):
    """保存数据到JSONL文件 - 修复版"""
    with open(filename, mode, encoding='utf-8') as f:
        # JSONL格式：每行一个完整的JSON对象，没有逗号分隔
        json_str = json.dumps(data, ensure_ascii=False)
        f.write(json_str + '\n')  # 只加换行，不加逗号

def run_one_debate(debate_id: int):
    conversation = [
        {"role": "system", "content": HEGEL_DEFENDER_SYSTEM},
    ]
    russell_messages = [
        {"role": "system", "content": RUSSELL_SYSTEM},
    ]

    # 第一轮由黑格尔信徒开火
    first_attack = call_deepseek(conversation + [{"role": "user", "content": "Begin! Attack Russell's critique of Hegel with your most vicious language!"}])
    if not first_attack:
        return
    
    conversation.append({"role": "assistant", "content": first_attack})
    russell_messages.append({"role": "user", "content": first_attack})

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"Debate {debate_id} - Round {round_num}")

        # 罗素反击
        russell_response = call_deepseek(russell_messages, temperature=0.85)
        if not russell_response:
            break

        # 实时保存训练数据
        training_sample = {
            "instruction": f"Please respond to the following defense of Hegel as Bertrand Russell:\n{first_attack if round_num==1 else conversation[-1]['content']}",
            "input": "",
            "output": russell_response
        }
        
        # 每生成一轮就立即保存
        save_to_jsonl(training_sample, OUTPUT_JSONL)
        print(f"Russell: {russell_response[:120]}...")
        print(f"✓ 已保存第{debate_id}场辩论第{round_num}轮数据")

        # 黑格尔信徒继续还击
        conversation.append({"role": "user", "content": russell_response})
        next_attack = call_deepseek(conversation, temperature=0.9)
        if not next_attack or len(next_attack) < 20:
            break

        conversation.append({"role": "assistant", "content": next_attack})
        russell_messages.append({"role": "user", "content": next_attack})

        time.sleep(0.5)  # 稍微降低等待时间

# ==================== 主循环 ====================
def main():
    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)
    
    start_time = time.time()
    completed_debates = 0
    
    for i in range(1, NUM_DEBATES + 1):
        print(f"\n=== Starting Debate {i}/{NUM_DEBATES} ===")
        run_one_debate(i)
        completed_debates += 1
        
        # 进度监控
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (NUM_DEBATES - i)
            print(f"进度: {i}/{NUM_DEBATES}, 预计剩余: {remaining/60:.1f}分钟")

    
    print(f"\n🎉 完成！共生成 {completed_debates} 场辩论数据")
    print(f"文件保存在: {OUTPUT_JSONL}")
    print(f"文件格式: 标准JSONL (每行一个JSON对象)")

if __name__ == "__main__":
    main()
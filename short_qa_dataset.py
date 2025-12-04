import json
from docx import Document
from openai import OpenAI

client = OpenAI(
    api_key="sk-0bdc7e4af6a04445a24de99515c0d12b",
    base_url="https://api.deepseek.com/v1"
)

RUSSELL_TEXT_FILE = "Russell_against_Hegel.docx"

def read_docx(file_path):
    """读取DOCX文件内容"""
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

def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    """将文本分割成重叠的块"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def generate_qa_pairs(text_chunk):
    """Using the API to Generate Q&A Pairs for Text Blocks"""
    try:
        prompt = f"""Please generate 3-5 high quality Q&A pairs based on the text below. 
                     Each question should be specific to the text and the answers should be accurate and concise.

Text content：
{text_chunk}

Please return the JSONL format directly, one full JSON object per line, not wrapped in an array. The format is as follows：
{{"instruction": "Question 1", "input": "", "output": "Answer 1"}}
{{"instruction": "Question 2", "input": "", "output": "Answer 2"}}
"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional text analysis assistant who specialises in extracting key information from text and generating high quality Q&A pairs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        return result
        
    except Exception as e:
        print(f"生成问答对失败: {e}")
        return ""

def process_document_for_finetuning():
    """主函数：处理文档并生成微调数据（直接生成JSONL）"""
    # 1. 读取文档
    print("正在读取文档...")
    document_text = read_docx(RUSSELL_TEXT_FILE)
    if not document_text:
        print("无法读取文档，程序退出。")
        return
    
    print(f"文档读取成功，长度: {len(document_text)} 字符")
    
    # 2. 分割文本
    print("正在分割文本...")
    text_chunks = split_text_into_chunks(document_text)
    print(f"文本分割为 {len(text_chunks)} 个块")
    
    # 3. 为每个文本块生成问答对并直接写入JSONL文件
    output_filename = "finetuning_data.jsonl"
    total_pairs = 0
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(text_chunks, 1):
            print(f"正在处理第 {i}/{len(text_chunks)} 个文本块...")
            
            jsonl_content = generate_qa_pairs(chunk)
            if jsonl_content:
                # 直接写入AI返回的JSONL内容
                f.write(jsonl_content + '\n')
                
                # 统计生成的数量
                lines = [line for line in jsonl_content.split('\n') if line.strip()]
                chunk_pairs = len(lines)
                total_pairs += chunk_pairs
                print(f"  生成了 {chunk_pairs} 个问答对")
            else:
                print(f"  第 {i} 个文本块生成问答对失败")
            
            # 避免API限制，添加延迟
            import time
            time.sleep(1)
    
    print(f"\n完成！共生成 {total_pairs} 个问答对")
    print(f"数据已保存到: {output_filename}")
    
    # 显示文件预览
    print("\n文件前3行预览:")
    with open(output_filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"第{i+1}行: {line.strip()}")
            else:
                break

if __name__ == "__main__":
    process_document_for_finetuning()

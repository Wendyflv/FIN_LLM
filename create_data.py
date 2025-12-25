import os
# 必须在 import transformers 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_ENDPOINT"] = "https://hf-mirror.com"
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =================配置区域=================
# 论文使用的基座模型是 Llama-3-8B-Instruct [cite: 117]
MODEL_ID = "Qwen/Qwen2.5-0.5B" 
OUTPUT_FILE = "findpo_train_data.jsonl"

# 标签映射：标准三分类
LABEL_MAP = {
    0: "Negative",
    1: "Neutral", 
    2: "Positive"
}

# NWGI 数据集的特殊映射 (5转3) 
# 假设原始标签为 0:Strong Neg, 1:Mild Neg, 2:Neu, 3:Mild Pos, 4:Strong Pos
NWGI_MAP = {
    0: "Negative", 1: "Negative",
    2: "Neutral",
    3: "Positive", 4: "Positive"
}

# =================模型加载=================
def load_reference_model():
    """
    加载参考模型 (Reference Model) 用于生成 Rejected 样本。
    如果没有显存，可以将 device_map 改为 "cpu" 或使用量化版本。
    """
    print(f"正在加载模型: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 方案A：不使用 device_map，手动设置设备
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    return tokenizer, model

# =================推理函数=================
def get_model_prediction(model, tokenizer, instruction, input_text):
    """
    让模型基于 Input 生成预测，用于判断它是对是错。
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}
Input: {input_text}
Sentiment:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            temperature=0.1, # 低温度以获得确定性结果
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip()
    
    # 简单解析：如果模型输出包含关键词，则判定为该类别
    if "Positive" in response: return "Positive"
    if "Negative" in response: return "Negative"
    if "Neutral" in response: return "Neutral"
    return "Neutral" # 默认兜底，或者你可以选择丢弃这条数据

# =================核心构造逻辑 (论文核心)=================
def construct_dpo_pair(model, tokenizer, instruction, input_text, ground_truth_label):
    """
    根据论文 Section 4.1.1 实现 Chosen/Rejected 逻辑
    """
    # 1. 构造 Chosen (胜出项)：直接使用 Ground Truth [cite: 141]
    chosen_response = ground_truth_label
    
    # 2. 获取模型预测 [cite: 142]
    model_pred = get_model_prediction(model, tokenizer, instruction, input_text)
    
    # 3. 构造 Rejected (落败项)
    all_labels = ["Positive", "Negative", "Neutral"]
    
    if model_pred == ground_truth_label:
        # 情况A：模型预测正确 [cite: 143, 144, 145]
        # 论文逻辑：随机采样一个错误的标签作为 Rejected
        remaining_labels = [l for l in all_labels if l != ground_truth_label]
        rejected_response = random.choice(remaining_labels)
    else:
        # 情况B：模型预测错误 
        # 论文逻辑：直接使用模型生成的错误预测作为 Rejected
        # 目的：Guide the model away from its own mistakes (远离自己的错误)
        rejected_response = model_pred

    # 返回 DPO 格式数据
    return {
        "instruction": instruction,
        "input": input_text,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

# =================主流程=================
def main():
    # 1. 初始化
    tokenizer, model = load_reference_model()
    dpo_data = []
    
    # 通用指令
    instruction = "Analyze the sentiment of the following financial text. Output one of: Positive, Negative, Neutral."

    # 2. 加载数据集 (这里以 FPB 为例，你可以把 TFNS 和 NWGI 加进来)
    print("正在处理 Financial PhraseBank (FPB)...")
    try:
        # 注意：你需要根据实际情况调整 dataset path
        ds_fpb = load_dataset("atrost/financial_phrasebank", split="train",trust_remote_code=True)
        
        for item in tqdm(ds_fpb):
            text = item['sentence']
            label_idx = item['label']
            ground_truth = LABEL_MAP[label_idx]
            
            dpo_entry = construct_dpo_pair(model, tokenizer, instruction, text, ground_truth)
            dpo_data.append(dpo_entry)
            
    except Exception as e:
        print(f"加载 FPB 失败: {e}")

    # 3. 加载 NWGI (GPT-labeled) - 演示 5转3 逻辑 
    # print("正在处理 NWGI...")
    # ds_nwgi = load_dataset("oliverwang15/news_with_gpt_instructions", split="train")
    # for item in ds_nwgi:
    #     # 需要根据实际字段调整
    #     label_idx = item['label'] # 假设这里是 0-4
    #     ground_truth = NWGI_MAP[label_idx] # 执行合并映射
    #     ...

    # 4. 保存为 jsonl
    print(f"正在保存 {len(dpo_data)} 条数据到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print("完成！")

if __name__ == "__main__":
    main()
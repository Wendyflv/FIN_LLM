import json
import os
import torch
import csv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 1. 配置裁判 (Qwen-Max) =================
# 注意：这里保留 Qwen-Max 仅用于"阶段三：自动化评分"，不再用于生成问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

llm = ChatOpenAI(
    model="qwen-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your api key here",  # 替换为你的 API Key
    streaming=False
)

# ================= 2. 阶段一：读取本地数据 =================
def load_test_questions(file_path="eval_data.json"):
    print(f"正在读取本地文件: {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}，请确保该文件在当前目录下。")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        print(f"成功加载 {len(questions)} 条测试数据")
        return questions
    except json.JSONDecodeError as e:
        print(f"JSON 文件格式错误: {e}")
        raise e
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        raise e

# ================= 3. 阶段二：使用本地 DPO 模型进行推理 =================
class LocalModelEvaluator:
    def __init__(self, model_path):
        print(f"正在加载本地 DPO 模型: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.model.eval()
        except Exception as e:
            print(f"加载模型失败，请检查路径 '{model_path}' 是否正确。错误信息: {e}")
            raise e

    def get_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # 这里 max_new_tokens 可以根据需要调整
            outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.3)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def run_inference(self, questions):
        all_results = []
        # 使用 tqdm 显示进度
        for q in tqdm(questions, desc="本地模型推理中"):
            # 定义三种 Prompt 策略
            prompts = {
                "Standard": f"<|im_start|>user\n直接回答：{q['question']}<|im_end|>\n<|im_start|>assistant\n",
                "CoT": f"<|im_start|>user\n请逐步思考并解决以下问题，展示你的推理过程：{q['question']}<|im_end|>\n<|im_start|>assistant\n",
                "ToT": f"<|im_start|>user\n针对以下问题，请提出三个不同的解决思路，评估每个思路的优缺点，并最后得出最合理的结论：{q['question']}<|im_end|>\n<|im_start|>assistant\n"
            }
            
            res = {"id": q['id'], "question": q['question']}
            for mode, p in prompts.items():
                res[mode] = self.get_response(p)
            all_results.append(res)
        return all_results

# ================= 4. 阶段三：Qwen-Max 自动化评分 =================
def judge_results(inference_results):
    final_report = []
    
    judge_system_prompt = """你是一名金融专家评审。请对以下三种 Prompt 策略的回答进行评分（0-10分）。
    评估标准：
    1. 逻辑严密性（Reasoning）
    2. 结论准确性（Accuracy）
    3. 专业性（Professionalism）
    
    请严格按此 JSON 格式返回，不要包含其他文字：
    {"Standard": 8, "CoT": 9, "ToT": 7, "Reason": "简短评价理由"}"""

    for res in tqdm(inference_results, desc="Qwen-Max 评分中"):
        user_content = f"问题: {res['question']}\n\n"
        user_content += f"A (Standard 回答): {res['Standard']}\n\n"
        user_content += f"B (CoT 回答): {res['CoT']}\n\n"
        user_content += f"C (ToT 回答): {res['ToT']}"
        
        try:
            response = llm.invoke([
                ("system", judge_system_prompt),
                ("user", user_content)
            ])
            # 清理可能存在的 markdown 代码块标记
            content_str = response.content.replace("```json", "").replace("```", "").strip()
            scores = json.loads(content_str)
            
            res.update({
                "Std_Score": scores.get("Standard", 0),
                "CoT_Score": scores.get("CoT", 0),
                "ToT_Score": scores.get("ToT", 0),
                "Judge_Comment": scores.get("Reason", "无评价")
            })
            final_report.append(res)
        except Exception as e:
            print(f"评分失败 ID {res['id']}: {e}")
            # 即使评分失败，也保留原始数据，避免数据丢失
            res.update({
                "Std_Score": -1, "CoT_Score": -1, "ToT_Score": -1, "Judge_Comment": f"评分出错: {str(e)}"
            })
            final_report.append(res)
            
    return final_report

# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    # 配置你的本地模型路径
    LOCAL_MODEL_PATH = "./qwen2-0.5b-dpo-aligned"
    DATA_FILE = "eval_data.json"

    # 1. 读取本地数据
    questions = load_test_questions(DATA_FILE)
    
    # 2. 本地 DPO 模型推理
    if os.path.exists(LOCAL_MODEL_PATH):
        local_eval = LocalModelEvaluator(LOCAL_MODEL_PATH)
        inference_results = local_eval.run_inference(questions)
        
        # 3. 裁判评分 (仍然使用 GPT-4 API)
        final_report = judge_results(inference_results)
        
        # 4. 保存为 CSV
        output_file = "full_alignment_eval_report.csv"
        try:
            with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
                if final_report:
                    writer = csv.DictWriter(f, fieldnames=final_report[0].keys())
                    writer.writeheader()
                    writer.writerows(final_report)
            print(f"\n[系统评估任务完成] 报告已生成：{output_file}")
        except Exception as e:
            print(f"保存 CSV 失败: {e}")
    else:
        print(f"错误：本地模型路径 {LOCAL_MODEL_PATH} 不存在，请检查路径配置。")
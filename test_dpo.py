import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =================配置路径=================
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
# 指向你刚才合并 SFT 后的那个物理目录（如果删了，就重新合并加载）
# 为了稳妥，我们直接从原始基座加载，然后手动合并两层
SFT_ADAPTER_DIR = "./qwen2-0.5b-fin-lora"
DPO_ADAPTER_DIR = "./qwen2-0.5b-dpo-aligned"

# =================1. 加载模型逻辑=================
print("正在构建对齐后的完整模型...")

# 加载原始基座
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# A. 加载 SFT 权重并合并 (必须先做这一步，因为 DPO 是基于 SFT 状态对齐的)
print("正在应用 SFT 金融知识层...")
model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)
model = model.merge_and_unload()

# B. 加载 DPO 权重 (这一层可以合并，也可以直接挂载推理)
print("正在应用 DPO 偏好对齐层...")
model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR)
model.eval()

# =================2. 测试推理函数=================
def ask_finance(prompt, input_text=""):
    # 构造与训练时完全一致的 Prompt 格式
    if input_text:
        full_prompt = f"<|im_start|>user\n{prompt}\nInput: {input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1, # 稍微加一点惩罚防止小模型复读
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 只解码生成的部分
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer

# =================3. 实测案例=================
print("\n" + "="*30)
print("金融模型实测开始")
print("="*30)

test_cases = [
    {"q": "什么是股票的市盈率（P/E Ratio）？", "i": ""},
    {"q": "分析这段文本的情感", "i": "公司本季度财报显示利润大幅增长20%，远超分析师预期。"},
    {"q": "如果通货膨胀上升，对债券价格有什么影响？", "i": ""}
]

for case in test_cases:
    print(f"\n【问题】: {case['q']}")
    if case['i']: print(f"【上下文】: {case['i']}")
    response = ask_finance(case['q'], case['i'])
    print(f"【回答】: \033[1;32m{response}\033[0m") # 绿色显示回答
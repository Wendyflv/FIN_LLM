import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2-0.5B"
adapter_path = "./qwen2-0.5b-fin-lora" # 上一步保存的路径

# 1. 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 2. 加载 LoRA 适配器 (合并权重)
model = PeftModel.from_pretrained(base_model, adapter_path)
# 如果你想永久合并权重保存：
# model = model.merge_and_unload() 

# 3. 测试推理
prompt = "什么是股票市场的市盈率？"
formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs, 
    max_new_tokens=200, 
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
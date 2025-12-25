import torch
import matplotlib.pyplot as plt  # <--- 新增：导入绘图库
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_ENDPOINT"] = "https://hf-mirror.com"
# ----------------配置参数----------------
MODEL_ID = "Qwen/Qwen2-0.5B"  
DATASET_ID = "eggbiscuit/DISC-FIN-SFT"
OUTPUT_DIR = "./qwen2-0.5b-fin-lora"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------1. 加载 Tokenizer----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ----------------2. 加载数据集----------------
print(f"Loading dataset: {DATASET_ID}...")
from datasets import Dataset
# 1. 使用 streaming=True 模式加载，这会绕过严格的 Arrow 格式检查
ds_stream = load_dataset(DATASET_ID, split="train", streaming=True)

# 2. 手动遍历并只提取我们需要的字段，过滤掉导致报错的坏数据（如 history 等）
data_list = []
for item in ds_stream:
    data_list.append({
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"]
    })

# 3. 将清洗后的数据转回标准的 Dataset 对象
dataset = Dataset.from_list(data_list)

# 打印一下数据样例以确认格式
print("Data sample:", dataset[0])

# ----------------3. 数据格式化函数----------------
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        instruction = example['instruction'][i]
        input_text = example['input'][i]
        output = example['output'][i]

        if input_text:
            instruction = f"{instruction}\n{input_text}"

        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        output_texts.append(text)
    return output_texts

# ----------------4. 加载模型----------------
print(f"Loading model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
model.gradient_checkpointing_enable()

# ----------------5. 配置 LoRA----------------
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,           
    lora_alpha=32,  
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# ----------------6. 训练参数----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,      # 每10步记录一次日志，这个频率决定了图的点的密度
    save_steps=100,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    group_by_length=True,
    report_to="tensorboard"       # 这里我们手动画图，所以保持 none
)

# ----------------7. 初始化 Trainer----------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=formatting_prompts_func,
    packing=False,
)

# ----------------8. 开始训练----------------
print("Starting training...")
trainer.train()

# ----------------9. 保存模型----------------
print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ----------------10. (新增) 绘制并保存 Loss 图----------------
print("Plotting loss curve...")

# 从 trainer 状态中获取日志历史
log_history = trainer.state.log_history

steps = []
losses = []

# 遍历日志提取 step 和 loss
for log in log_history:
    if "loss" in log and "step" in log:
        steps.append(log["step"])
        losses.append(log["loss"])

if steps:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Loss", color='blue', alpha=0.6)
    
    # 可选：绘制平滑曲线（移动平均），让趋势更明显
    if len(losses) > 10:
        try:
            import pandas as pd
            smooth_loss = pd.Series(losses).rolling(window=5).mean()
            plt.plot(steps, smooth_loss, label="Smoothed Loss (MA-5)", color='red', linewidth=2)
        except ImportError:
            pass # 如果没有安装 pandas 则跳过平滑曲线

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片到输出目录
    loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved to: {loss_plot_path}")
    
    # 如果是在 Jupyter Notebook 中，可以使用 plt.show()
    # plt.show() 
else:
    print("No loss data found in logs.")

print("Training and plotting complete!")
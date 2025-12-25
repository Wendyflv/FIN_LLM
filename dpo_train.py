import os
import torch
import gc
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOTrainer, DPOConfig

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# =================配置参数=================
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"       
SFT_ADAPTER_DIR = "./qwen2-0.5b-fin-lora" 
DATA_FILE = "findpo_train_data.jsonl"     
OUTPUT_DIR = "./qwen2-0.5b-dpo-aligned"   
TEMP_SFT_MERGED = "./temp_sft_merged"    

# =================1. 数据预处理=================
def process_dpo_data(example):
    prompt = f"<|im_start|>user\n{example['instruction']}\nInput: {example['input']}<|im_end|>\n<|im_start|>assistant\n" if example['input'] else f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    return {
        "prompt": prompt,
        "chosen": f"{example['chosen']}<|im_end|>",
        "rejected": f"{example['rejected']}<|im_end|>"
    }

# =================2. 物理合并 SFT=================
print("正在物理合并 SFT 权重...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
model_sft = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR)
merged_model = model_sft.merge_and_unload()
merged_model.save_pretrained(TEMP_SFT_MERGED)

del base_model, model_sft, merged_model
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(TEMP_SFT_MERGED, torch_dtype=torch.bfloat16, device_map="auto")

# =================3. 挂载 DPO LoRA=================
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()

# =================4. 准备数据=================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(process_dpo_data)

# =================5. DPO 训练配置=================
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,
    num_train_epochs=1,
    logging_steps=5, # 较小的日志步数可以获得更平滑的曲线
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
    remove_unused_columns=False,
    precompute_ref_log_probs=False,
    report_to="none" # 禁用默认报告，手动绘图
)

# =================6. 初始化 Trainer=================
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# =================7. 开始训练=================
print("开始对齐训练...")
train_result = trainer.train()

# =================8. 绘图监控指标=================
def plot_dpo_log(log_history, save_path):
    steps = []
    loss = []
    acc = []
    margin = []
    
    for entry in log_history:
        if "loss" in entry and "rewards/accuracies" in entry:
            steps.append(entry["step"])
            loss.append(entry["loss"])
            acc.append(entry.get("rewards/accuracies", 0))
            margin.append(entry.get("rewards/margins", 0))

    if not steps:
        print("未发现有效日志，跳过绘图。")
        return

    plt.figure(figsize=(12, 8))

    # 子图1: Loss
    plt.subplot(2, 2, 1)
    plt.plot(steps, loss, color='blue', label='Training Loss')
    plt.title('DPO Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)

    # 子图2: Accuracy (最关键)
    plt.subplot(2, 2, 2)
    plt.plot(steps, acc, color='green', label='Rewards Accuracy')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Baseline(0.5)')
    plt.title('Preference Accuracy (Should > 0.5)')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)

    # 子图3: Margin
    plt.subplot(2, 2, 3)
    plt.plot(steps, margin, color='purple', label='Rewards Margin')
    plt.title('Reward Margin (Should Increase)')
    plt.xlabel('Step')
    plt.ylabel('Margin')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练指标图已保存至: {save_path}")

# 执行绘图
plot_dpo_log(trainer.state.log_history, os.path.join(OUTPUT_DIR, "dpo_metrics.png"))

# 保存模型
trainer.save_model(OUTPUT_DIR)
print(f"训练完成！")
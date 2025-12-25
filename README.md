# FinAlign: 金融领域小语言模型对齐项目

本仓库包含了针对 **Qwen2.5-0.5B** 模型进行金融领域适配的完整全流程方案，涵盖了从监督微调 (SFT) 到直接偏好优化 (DPO) 的两阶段训练，以及基于提示工程 (CoT/ToT) 的系统化评估框架。

## 📂 文件说明

### 1. 训练与数据构造脚本

- `lora_train.py`: 监督微调 (SFT) 脚本。使用 LoRA 技术对全线性层进行微调，将 `eggbiscuit/DISC-FIN-SFT` 数据集的金融知识注入基座模型。
- `create_data.py`: 偏好数据集构造脚本。参考 FinDPO 逻辑，利用 SFT 后的模型进行推理，通过“远离自身错误”的逻辑构造 Chosen/Rejected 样本对。
- `dpo_train.py`: 直接偏好优化 (DPO) 脚本。加载 SFT 后的权重，利用自建的偏好数据集进行行为对齐训练，并自动生成训练指标图（Loss/Accuracy/Margin）。

### 2. 推理与评估工具

- `test_lora.py`: 用于测试 SFT 阶段后的模型推理效果。
- `test_dpo.py`: 用于加载对齐后的完整模型（SFT 合并层 + DPO 适配器），测试其在金融问答和情感分析中的表现。
- `cot_eval.py`: 系统评估框架。支持 **Standard**、**Chain-of-Thought (CoT)** 和 **Tree-of-Thought (ToT)** 三种提示策略的推理，并调用高级模型（如 Qwen-Max/GPT-4）作为裁判进行自动化打分。

## 🚀 快速开始

### 1. 环境准备

确保您的 Python 环境安装了必要的库：

```bash
pip install torch transformers datasets peft trl accelerate matplotlib tqdm langchain_openai
```

### 2. 第一阶段：监督微调 (SFT)

运行以下脚本开始知识注入：

```bash
python lora_train.py
```

训练完成后，权重将保存在 `./qwen2-0.5b-fin-lora` 目录下。

### 3. 第二阶段：构造 DPO 偏好数据

基于 SFT 模型生成偏好对：

```bash
python create_data.py
```

此脚本会生成`findpo_train_data.jsonl` 文件。

### 4. 第三阶段：直接偏好优化 (DPO)

进行偏好对齐训练：

```bash
python dpo_train.py
```

对齐后的模型将保存在`./qwen2-0.5b-dpo-aligned`。

### 5. 系统评估

运行自动化评估流程：

```bash
python cot_eval.py
```

评估结果将保存为`full_alignment_eval_report.csv`，包含每种策略的得分及裁判评语。

## 📊 实验指标

本项目在训练过程中会自动绘制并保存以下监控指标：

- **SFT Loss 曲线**: 保存于 `./qwen2-0.5b-fin-lora/loss_curve.png`。
- **DPO 综合指标**: 包含对齐准确率（Accuracy）和奖励差值（Margin），保存于 `./qwen2-0.5b-dpo-aligned/dpo_metrics.png`。



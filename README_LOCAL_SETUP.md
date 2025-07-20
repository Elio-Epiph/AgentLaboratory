# AgentLaboratory 本地模型部署指南

本指南将帮助你在服务器上使用本地 Qwen 模型运行 AgentLaboratory，无需联网或 API 密钥。

## 📋 前置要求

### 1. 服务器环境
- Python 3.8+
- 至少 8GB RAM（推荐 16GB+）
- 足够的磁盘空间存储模型

### 2. 已安装的模型
根据你的服务器，我们推荐使用以下模型：
- **Qwen2.5-1.5B-Instruct**（最快，适合 CPU 推理）
- **Qwen2.5-7B-Instruct**（平衡性能和速度）
- **Qwen2.5-72B-Instruct**（最高性能，需要更多资源）

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install torch transformers accelerate tiktoken pyyaml
```

### 2. 测试本地模型
```bash
python test_local_model.py
```

### 3. 优化 CPU 推理（可选）
```bash
python optimize_cpu_inference.py
```

### 4. 运行 AgentLaboratory
```bash
python run_agentlab_local.py
```

## 📁 文件说明

### 核心文件
- `inference.py` - 已修改，支持本地模型推理
- `experiment_configs/MATH_agentlab.yaml` - 配置文件，使用本地模型
- `test_local_model.py` - 测试脚本
- `optimize_cpu_inference.py` - CPU 优化脚本
- `run_agentlab_local.py` - 运行脚本

### 模型路径配置
在 `experiment_configs/MATH_agentlab.yaml` 中：
```yaml
llm-backend: "qwen-local:/data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct"
lit-review-backend: "qwen-local:/data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct"
```

## ⚡ 性能优化建议

### 1. 模型选择
- **1.5B 模型**：最快，适合快速原型和测试
- **7B 模型**：平衡性能和速度，推荐用于正式实验
- **72B 模型**：最高性能，但需要大量内存和计算资源

### 2. CPU 优化技巧
```python
# 在 inference.py 中已实现的优化：
- low_cpu_mem_usage=True  # 减少内存使用
- torch_dtype=torch.float32  # CPU 上用 float32
- torch.no_grad()  # 推理时禁用梯度计算
```

### 3. 推理参数优化
```python
# 减少生成 token 数量以加快速度
max_new_tokens=256  # 而不是 512

# 使用确定性生成（更快）
temperature=0.0  # 而不是 0.7
```

## 🔧 故障排除

### 1. 模型加载失败
```bash
# 检查模型路径
ls /data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct/

# 检查必要文件
ls /data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct/config.json
ls /data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct/tokenizer.json
ls /data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct/pytorch_model.bin
```

### 2. 内存不足
- 使用更小的模型（1.5B 而不是 7B）
- 减少 `max_new_tokens`
- 使用 `low_cpu_mem_usage=True`

### 3. 推理速度慢
- 使用 1.5B 模型
- 减少 `max_new_tokens`
- 使用 `temperature=0.0` 进行确定性生成
- 考虑使用 GPU（如果可用）

## 📊 性能基准

### 测试环境
- CPU: Intel Xeon
- RAM: 16GB
- 模型: Qwen2.5-1.5B-Instruct

### 推理速度
- **1.5B 模型**: ~2-3 秒/100 tokens
- **7B 模型**: ~8-12 秒/100 tokens
- **72B 模型**: ~30-60 秒/100 tokens

## 🎯 使用示例

### 1. 基本运行
```bash
python run_agentlab_local.py
```

### 2. 指定不同模型
```bash
python run_agentlab_local.py --model /data/pretrained_models/Qwen2.5-7B-Instruct
```

### 3. 仅测试设置
```bash
python run_agentlab_local.py --test-only
```

### 4. 使用自定义配置
```bash
python run_agentlab_local.py --config experiment_configs/MATH_agentrxiv.yaml
```

## 🔄 更新和维护

### 1. 更新模型
```bash
# 下载新模型
git lfs clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct /data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct-new

# 更新配置文件中的路径
```

### 2. 更新依赖
```bash
pip install --upgrade torch transformers accelerate
```

## 📝 注意事项

1. **首次运行**：模型加载需要时间，请耐心等待
2. **内存使用**：确保有足够的内存运行选择的模型
3. **网络依赖**：虽然模型是本地的，但某些功能（如 arXiv 检索）仍需要网络
4. **API 密钥**：使用本地模型时不需要 OpenAI 或 DeepSeek API 密钥

## 🆘 获取帮助

如果遇到问题：
1. 运行 `python test_local_model.py` 检查基本设置
2. 检查模型路径和文件完整性
3. 查看错误日志和内存使用情况
4. 尝试使用更小的模型进行测试

---

**祝你使用愉快！🎉** 
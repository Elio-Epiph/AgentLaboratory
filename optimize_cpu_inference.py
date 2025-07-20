#!/usr/bin/env python3
"""
CPU 推理优化脚本
用于提高本地模型在 CPU 上的推理速度
"""

import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def optimize_for_cpu_inference(model_path):
    """
    优化模型用于 CPU 推理
    """
    print(f"Optimizing model for CPU inference: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return None
    
    try:
        # 1. 加载 tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. 加载模型并优化
        print("Loading and optimizing model...")
        
        # CPU 优化选项
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU 上用 float32
            device_map="cpu",
            low_cpu_mem_usage=True,     # 减少内存使用
            trust_remote_code=True
        )
        
        # 3. 设置为评估模式
        model.eval()
        
        # 4. 尝试进一步优化（如果可用）
        try:
            # 使用 torch.compile 优化（PyTorch 2.0+）
            if hasattr(torch, 'compile'):
                print("Applying torch.compile optimization...")
                model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️  torch.compile not available: {e}")
        
        print("✅ Model optimized for CPU inference!")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error optimizing model: {e}")
        return None

def benchmark_inference(tokenizer, model, prompt, num_runs=5):
    """
    测试推理速度
    """
    print(f"\nBenchmarking inference speed ({num_runs} runs)...")
    
    # 准备输入
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        inference_time = end_time - start_time
        times.append(inference_time)
        
        print(f"Run {i+1}: {inference_time:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage inference time: {avg_time:.2f}s")
    print(f"Tokens per second: {100/avg_time:.1f}")
    
    return avg_time

def get_memory_usage():
    """
    获取内存使用情况
    """
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    return memory_mb

def main():
    """
    主函数
    """
    print("=" * 60)
    print("CPU Inference Optimization for Local Models")
    print("=" * 60)
    
    # 模型路径
    model_path = "/data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct"
    
    # 检查 PyTorch 版本
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 优化模型
    result = optimize_for_cpu_inference(model_path)
    if result is None:
        return
    
    tokenizer, model = result
    
    # 测试推理速度
    test_prompt = "Explain what is machine learning in one sentence."
    benchmark_inference(tokenizer, model, test_prompt)
    
    # 检查内存使用
    get_memory_usage()
    
    print("\n🎉 CPU optimization completed!")
    print("\nTips for faster CPU inference:")
    print("1. Use smaller models (1.5B vs 7B)")
    print("2. Reduce max_new_tokens")
    print("3. Use lower temperature for deterministic output")
    print("4. Consider using int8 quantization if memory allows")

if __name__ == "__main__":
    main() 
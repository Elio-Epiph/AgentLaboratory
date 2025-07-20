#!/usr/bin/env python3
"""
测试本地 Qwen 模型是否能正常工作
"""

import sys
import os
sys.path.append('.')

from inference import query_model

def test_local_qwen():
    """测试本地 Qwen 模型"""
    print("Testing local Qwen model...")
    
    # 测试用的模型路径
    model_path = "/data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct"
    model_str = f"qwen-local:{model_path}"
    
    # 测试 prompt
    system_prompt = "You are a helpful AI assistant."
    prompt = "Hello, please introduce yourself briefly."
    
    try:
        print(f"Using model: {model_str}")
        print(f"System prompt: {system_prompt}")
        print(f"User prompt: {prompt}")
        print("-" * 50)
        
        # 调用模型
        response = query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            temp=0.7,
            print_cost=True
        )
        
        print("Response:")
        print(response)
        print("-" * 50)
        print("✅ Local Qwen model test successful!")
        
    except Exception as e:
        print(f"❌ Error testing local model: {e}")
        return False
    
    return True

def test_agentlab_config():
    """测试 AgentLaboratory 配置"""
    print("\nTesting AgentLaboratory configuration...")
    
    try:
        import yaml
        with open('experiment_configs/MATH_agentlab.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        llm_backend = config.get('llm-backend', '')
        print(f"LLM Backend: {llm_backend}")
        
        if llm_backend.startswith('qwen-local:'):
            print("✅ Configuration looks good for local Qwen!")
        else:
            print("⚠️  Configuration might need adjustment")
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Local Qwen Model Test for AgentLaboratory")
    print("=" * 60)
    
    # 测试本地模型
    success = test_local_qwen()
    
    # 测试配置
    test_agentlab_config()
    
    if success:
        print("\n🎉 All tests passed! You can now run AgentLaboratory with local Qwen.")
        print("\nTo run AgentLaboratory:")
        print("python ai_lab_repo.py --config experiment_configs/MATH_agentlab.yaml")
    else:
        print("\n❌ Tests failed. Please check the error messages above.") 
#!/usr/bin/env python3
"""
AgentLaboratory 本地运行脚本
使用本地 Qwen 模型运行 AgentLaboratory
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'torch', 'transformers', 'accelerate', 'yaml', 'tiktoken'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_model_path(model_path):
    """检查模型路径是否存在"""
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    # 检查必要的文件
    required_files = ['config.json', 'tokenizer.json', 'pytorch_model.bin']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing model files: {', '.join(missing_files)}")
        print("The model might be incomplete or corrupted")
        return False
    
    print(f"✅ Model found at: {model_path}")
    return True

def run_agentlab(config_file, model_path=None):
    """运行 AgentLaboratory"""
    
    # 构建命令
    cmd = [
        sys.executable, 'ai_lab_repo.py',
        '--config', config_file
    ]
    
    if model_path:
        cmd.extend(['--llm-backend', f'qwen-local:{model_path}'])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # 运行 AgentLaboratory
        result = subprocess.run(cmd, check=True)
        print("✅ AgentLaboratory completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ AgentLaboratory failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  AgentLaboratory interrupted by user")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run AgentLaboratory with local Qwen model')
    parser.add_argument('--config', default='experiment_configs/MATH_agentlab.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', default='/data/kaohesheng/qiujinbo/Qwen2.5-1.5B-Instruct',
                       help='Local model path')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the setup, don\'t run AgentLaboratory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AgentLaboratory Local Runner")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查模型
    if not check_model_path(args.model):
        return
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    print(f"✅ Config file found: {args.config}")
    
    if args.test_only:
        print("\n🧪 Test mode - checking setup only")
        print("✅ Setup looks good! You can run AgentLaboratory now.")
        return
    
    # 运行 AgentLaboratory
    print(f"\n🚀 Starting AgentLaboratory with local Qwen model...")
    success = run_agentlab(args.config, args.model)
    
    if success:
        print("\n🎉 AgentLaboratory completed successfully!")
    else:
        print("\n❌ AgentLaboratory failed. Check the error messages above.")

if __name__ == "__main__":
    main() 
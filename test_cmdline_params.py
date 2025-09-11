#!/usr/bin/env python3
"""
Test command line parameter constraint support
"""

import sys
import subprocess

def test_cmdline_args():
    """Test different command line argument combinations"""
    
    test_commands = [
        {
            'name': '默认参数 (无params_constraints)',
            'cmd': ['python', '-c', '''
from args import arg_parser
import sys
args = arg_parser.parse_args([])
print(f"FLOPs constraints: {args.flops_constraints}")
print(f"Params constraints: {args.params_constraints}")
''']
        },
        {
            'name': '设置params_constraints',
            'cmd': ['python', '-c', '''
from args import arg_parser
import sys
args = arg_parser.parse_args([
    "--flops_constraints", "100", "120", "150", "200",
    "--params_constraints", "0.5", "0.8", "1.2", "1.5"
])
print(f"FLOPs constraints: {args.flops_constraints}")
print(f"Params constraints: {args.params_constraints}")
''']
        },
        {
            'name': '只设置FLOPs (兼容性测试)',
            'cmd': ['python', '-c', '''
from args import arg_parser
import sys
args = arg_parser.parse_args([
    "--flops_constraints", "80", "100", "130"
])
print(f"FLOPs constraints: {args.flops_constraints}")
print(f"Params constraints: {args.params_constraints}")
''']
        }
    ]
    
    print("=== 命令行参数约束测试 ===\n")
    
    for i, test in enumerate(test_commands):
        print(f"测试 {i+1}: {test['name']}")
        print("-" * 50)
        
        try:
            result = subprocess.run(
                test['cmd'],
                capture_output=True,
                text=True,
                cwd='.'
            )
            
            if result.returncode == 0:
                print("✓ 成功")
                print("输出:")
                print(result.stdout.strip())
            else:
                print("✗ 失败")
                print("错误:")
                print(result.stderr.strip())
                
        except Exception as e:
            print(f"✗ 异常: {e}")
        
        print()

def show_help():
    """Show help message for new parameter"""
    print("=== 新增参数说明 ===\n")
    
    try:
        result = subprocess.run(
            ['python', '-c', '''
from args import arg_parser
arg_parser.print_help()
'''],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        if result.returncode == 0:
            # 只显示与params_constraints相关的部分
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'params_constraints' in line:
                    print("找到新参数:")
                    print(line)
                    if i + 1 < len(lines):
                        print(lines[i+1])  # 显示help信息
                    break
        else:
            print("无法获取帮助信息")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_cmdline_args()
    print("="*60)
    show_help()
#!/usr/bin/env python3
# cleanup_check.py
# 检查是否还有旧的API引用

import os
import re
import sys

def check_old_api_references():
    """检查是否还有旧的API引用"""
    
    # 要检查的文件
    files_to_check = [
        "constants.py",
        "ros_interface.py", 
        "state_manager.py",
        "robot_controller.py",
        "run_loco_policy.py",
        "test_motion_switcher.py"
    ]
    
    # 旧的API关键词
    old_api_keywords = [
        "sport_state",
        "robot_state",
        "SPORT_STATE_API_ID",
        "publish_sport_state",
        "_sport_state_change",
        "/api/robot_state/request"
    ]
    
    found_old_references = []
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for keyword in old_api_keywords:
                    if keyword in line:
                        found_old_references.append({
                            'file': filename,
                            'line': line_num,
                            'keyword': keyword,
                            'content': line.strip()
                        })
    
    return found_old_references

def check_new_api_usage():
    """检查新API的使用情况"""
    
    files_to_check = [
        "constants.py",
        "ros_interface.py", 
        "state_manager.py",
        "robot_controller.py",
        "run_loco_policy.py",
        "test_motion_switcher.py"
    ]
    
    # 新的API关键词
    new_api_keywords = [
        "motion_switcher",
        "MOTION_SWITCHER_API_ID",
        "publish_motion_switcher",
        "/api/motion_switcher/request"
    ]
    
    found_new_references = []
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for keyword in new_api_keywords:
                    if keyword in line:
                        found_new_references.append({
                            'file': filename,
                            'line': line_num,
                            'keyword': keyword,
                            'content': line.strip()
                        })
    
    return found_new_references

def main():
    """主函数"""
    print("=== 清理检查报告 ===\n")
    
    # 检查旧的API引用
    print("1. 检查旧的API引用...")
    old_refs = check_old_api_references()
    
    if old_refs:
        print(f"❌ 发现 {len(old_refs)} 个旧的API引用:")
        for ref in old_refs:
            print(f"   {ref['file']}:{ref['line']} - {ref['keyword']}")
            print(f"   {ref['content']}")
        print()
    else:
        print("✅ 没有发现旧的API引用\n")
    
    # 检查新的API使用
    print("2. 检查新的API使用...")
    new_refs = check_new_api_usage()
    
    if new_refs:
        print(f"✅ 发现 {len(new_refs)} 个新的API使用:")
        for ref in new_refs:
            print(f"   {ref['file']}:{ref['line']} - {ref['keyword']}")
        print()
    else:
        print("❌ 没有发现新的API使用\n")
    
    # 总结
    print("=== 总结 ===")
    if not old_refs and new_refs:
        print("✅ 清理完成！所有旧的API引用已移除，新API已正确使用。")
        return 0
    elif old_refs:
        print("❌ 清理不完整，仍有旧的API引用需要移除。")
        return 1
    else:
        print("⚠️  没有发现API使用，请检查代码。")
        return 2

if __name__ == "__main__":
    sys.exit(main()) 
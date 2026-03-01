import os
import re
from pathlib import Path
from typing import Tuple, Dict, List, Any

def count_lines_in_file(file_path: Path) -> Tuple[int, int, int]:
    """统计单个Python文件的总行数、代码行数和注释行数"""
    total_lines = 0
    code_lines = 0
    comment_lines = 0
    in_multiline_comment = False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                line = line.strip()

                # 处理多行注释
                if not in_multiline_comment:
                    # 检查是否为单行注释
                    if line.startswith('#'):
                        comment_lines += 1
                        continue
                    
                    # 检查是否为多行注释的开始
                    if line.startswith("'''") or line.startswith('"""'):
                        in_multiline_comment = True
                        comment_lines += 1
                        
                        # 检查是否同时是多行注释的结束
                        if line.count("'''") == 2 or line.count('"""') == 2:
                            in_multiline_comment = False
                        continue
                    
                    # 非空行视为代码行
                    if line:
                        code_lines += 1
                else:
                    comment_lines += 1
                    # 检查是否为多行注释的结束
                    if line.endswith("'''") or line.endswith('"""'):
                        in_multiline_comment = False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return total_lines, code_lines, comment_lines

def scan_directory(root_dir: Path = Path('.')) -> Dict[str, Any]:
    """扫描目录及其子目录中的所有Python文件并统计行数"""
    results = {
        'total_files': 0,
        'total_lines': 0,
        'code_lines': 0,
        'comment_lines': 0,
        'files': []  # type: List[Dict[str, Any]]
    }

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(root_dir)
                
                total, code, comments = count_lines_in_file(file_path)
                results['total_files'] += 1
                results['total_lines'] += total
                results['code_lines'] += code
                results['comment_lines'] += comments
                results['files'].append({
                    'path': str(rel_path),
                    'total_lines': total,
                    'code_lines': code,
                    'comment_lines': comments
                })
    
    # 按代码行数排序文件
    results['files'].sort(key=lambda x: x['code_lines'], reverse=True)
    return results

def print_statistics(results: Dict[str, Any]) -> None:
    """打印统计结果"""
    print("\n===== Python代码行数统计 =====")
    print(f"总文件数: {results['total_files']}")
    print(f"总行数: {results['total_lines']}")
    print(f"代码行数: {results['code_lines']}")
    print(f"注释行数: {results['comment_lines']}")
    print(f"注释比例: {results['comment_lines']/results['total_lines']*100:.2f}%")
    
    print("\n前10大文件:")
    for file in results['files'][:10]:
        print(f"{file['path']}: {file['code_lines']} 行代码")

if __name__ == "__main__":
    current_dir = Path('.')
    stats = scan_directory(current_dir)
    print_statistics(stats)
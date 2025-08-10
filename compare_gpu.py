import numpy as np
import re
import sys

def parse_matrix_from_log(log_content, name, rows, cols):
    """
    从日志文件内容中解析指定的矩阵。

    Args:
        log_content (str): 日志文件的全部内容。
        name (str): 要解析的矩阵名称 (例如, 'Mat_A')。
        rows (int): 矩阵的行数。
        cols (int): 矩阵的列数。

    Returns:
        numpy.ndarray: 解析出的矩阵。
    """
    # 定义用于查找矩阵头部的正则表达式
    if name == 'Mat_C':
        header_pattern = re.compile(r"^\s*" + re.escape(name) + r":")
    else:
        header_pattern = re.compile(r"^\s*" + re.escape(name) + r"\(K-major:\d+x\d+\):")

    lines = log_content.split('\n')
    start_index = -1
    for i, line in enumerate(lines):
        if header_pattern.search(line):
            start_index = i + 1
            break
    
    if start_index == -1:
        raise ValueError(f"错误：在日志文件中未找到矩阵 {name}。")

    # 提取属于该矩阵的所有数据行
    matrix_lines = []
    
    # 修正后的终结符正则表达式：
    # 匹配行首任意空格后的 Mat_ | Copy SMem | run mma | end mma
    end_pattern = re.compile(r"^\s*(Mat_[A-Z]|Copy SMem|run mma|end mma)")
    
    for i in range(start_index, len(lines)):
        line = lines[i]
        # 如果找到终结符，或该行为空，则停止解析当前矩阵
        if end_pattern.search(line) or not line.strip():
            break
        
        # 清理行数据：移除行号、源码引用标记和方括号
        line_content = re.sub(r'^\s*\d+\s*:\s*', '', line)
        line_content = re.sub(r'\\', '', line_content)
        line_content = line_content.replace('[', '').replace(']', '')
        matrix_lines.append(line_content.strip())

    # 将所有数字连接成一个长字符串，然后分割
    full_data_str = " ".join(matrix_lines)
    numbers = []
    if full_data_str:
        # 按空格分割字符串，过滤掉可能产生的空字符串
        num_strings = [s for s in full_data_str.split(' ') if s]
        numbers = [float(n) for n in num_strings]
    
    # 检查元素数量是否匹配
    expected_elements = rows * cols
    if len(numbers) != expected_elements:
        print(f"警告：矩阵 {name} 解析出的元素数量为 {len(numbers)}，但期望为 {expected_elements}。", file=sys.stderr)
        # 补齐或截断以尝试继续
        if len(numbers) > expected_elements:
            numbers = numbers[:expected_elements]
        else:
            numbers.extend([0.0] * (expected_elements - len(numbers)))

    # 将一维列表重塑为正确的矩阵维度
    return np.array(numbers).reshape((rows, cols))

def extract_mnk_from_log(log_content):
    """
    从日志第一行提取形如 'shape:M N K' 的 M, N, K 三个整数
    """
    match = re.search(r'shape\s*:\s*(\d+)\s+(\d+)\s+(\d+)', log_content)
    if not match:
        raise ValueError("日志中未找到形如 'shape:M N K' 的形状描述")
    M = int(match.group(1))
    N = int(match.group(2))
    K = int(match.group(3))
    return M, N, K

def main():
    """
    主函数，执行整个流程。
    """
    try:
        # 读取日志文件
        with open('test.log', 'r', encoding='utf-8') as f:
            log_content = f.read()

        M, N, K = extract_mnk_from_log(log_content)

        # --- 解析矩阵 ---
        mat_a = parse_matrix_from_log(log_content, 'Mat_A', M, K)
        mat_b = parse_matrix_from_log(log_content, 'Mat_B', N, K)
        mat_c_from_log = parse_matrix_from_log(log_content, 'Mat_C', M, N)

        # --- 执行矩阵乘法 ---
        calculated_mat_c = np.matmul(mat_a, mat_b.T)

        # --- 对比结果 ---
        are_equal = np.allclose(calculated_mat_c, mat_c_from_log, atol=1e-5)

        # --- 输出结论 ---
        print("矩阵解析与计算完成。")
        print("-" * 30)
        print(f"Mat_A 的维度: {mat_a.shape}")
        print(f"Mat_B 的维度: {mat_b.shape}")
        print(f"日志中 Mat_C 的维度: {mat_c_from_log.shape}")
        print(f"计算出的 Mat_C 维度: {calculated_mat_c.shape}")
        print("-" * 30)

        print("\n日志中的 Mat_C (前5x5片段):")
        print(mat_c_from_log[:8, :8])
        
        print("\n计算出的 Mat_C (前5x5片段):")
        print(calculated_mat_c[:8, :8])
        print("-" * 30)

        if are_equal:
            print("\n✅ 结论：结果一致。")
            print("脚本计算出的 Mat_C 与日志文件中记录的 Mat_C 完全相同。")
        else:
            print("\n❌ 结论：结果不一致。")
            print("脚本计算出的 Mat_C 与日志文件中记录的 Mat_C 不同。")
            difference = np.abs(calculated_mat_c - mat_c_from_log)
            print(f"最大差值: {np.max(difference)}")

    except FileNotFoundError:
        print("错误：找不到 'out.log' 文件。请确保该文件与脚本在同一目录下。")
    except Exception as e:
        print(f"在执行过程中发生错误: {e}")

if __name__ == '__main__':
    main()
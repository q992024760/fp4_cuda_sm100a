import numpy as np
import re
import sys
import torch


def mxfp4_e4m3_matmul(block_size=32, A_quant=None, B_quant=None, A_block_scales=None, B_block_scales=None, calculate_dtype=torch.float64):
    # 将输入的数组都转为torch格式
    A_quant = torch.from_numpy(A_quant) if isinstance(A_quant, np.ndarray) else A_quant
    B_quant = torch.from_numpy(B_quant) if isinstance(B_quant, np.ndarray) else B_quant
    A_block_scales = torch.from_numpy(A_block_scales) if isinstance(A_block_scales, np.ndarray) else A_block_scales
    B_block_scales = torch.from_numpy(B_block_scales) if isinstance(B_block_scales, np.ndarray) else B_block_scales
    # 将scale重新解释为torch的fp8_e4m3类型
    # 使用view重新解释二进制数据为float8_e4m3fn
    A_block_scales = A_block_scales.view(torch.float8_e4m3fn)
    B_block_scales = B_block_scales.view(torch.float8_e4m3fn)
    # import pdb; pdb.set_trace()

    m, k = A_quant.shape
    n, k2 = B_quant.shape
    assert k == k2, "The number of columns of A and B must be equal"
    # 保证block size兼容
    padded_k = (k + block_size - 1) // block_size * block_size
    n_blocks = padded_k // block_size

    # 初始化结果矩阵
    C = torch.zeros((m, n), dtype=torch.float32)

    # 执行矩阵乘法
    for i in range(m):
        for j in range(n):
            total = torch.tensor(0.0, dtype=calculate_dtype)

            # 处理每个block
            for block_idx in range(n_blocks):
                start = block_idx * block_size
                end = min((block_idx + 1) * block_size, k)
                actual_block_size = end - start

                if actual_block_size == 0:
                    continue

                # 取A和B的block
                A_block = A_quant[i, start:end].to(calculate_dtype)
                B_block = B_quant[j, start:end].to(calculate_dtype)
                A_scale = A_block_scales[i, block_idx].to(calculate_dtype)
                B_scale = B_block_scales[j, block_idx].to(calculate_dtype)

                # 计算点积
                dot = torch.sum(A_block * B_block * A_scale * B_scale)
                total += dot
            C[i, j] = total

    return C.numpy()

def mxfp4_e8m0_matmul(block_size=32, A_quant=None, B_quant=None, A_block_scales=None, B_block_scales=None, calculate_dtype=np.float32):
    m, k = A_quant.shape
    n, k2 = B_quant.shape
    assert k == k2, "The number of columns of A and B must be equal"
    # Ensure block size compatibility
    padded_k = (k + block_size - 1) // block_size * block_size
    n_blocks = padded_k // block_size
    
    # Initialize result matrix (始终float32)
    C = np.zeros((m, n), dtype=np.float32)
    
    # Perform matrix multiplication
    for i in range(m):
        for j in range(n):
            total = calculate_dtype(0.0)
            
            # Process each block
            for block_idx in range(n_blocks):
                start = block_idx * block_size
                end = min((block_idx + 1) * block_size, k)
                actual_block_size = end - start
                
                if actual_block_size == 0:
                    continue
                
                # Get blocks of A and B
                A_block = A_quant[i, start:end].astype(calculate_dtype)
                B_block = B_quant[j, start:end].astype(calculate_dtype)
                A_scale = calculate_dtype(A_block_scales[i, block_idx])
                B_scale = calculate_dtype(B_block_scales[j, block_idx])
                # import pdb; pdb.set_trace()
                A_scale = (calculate_dtype(2.0) ** (A_scale - 127)).astype(calculate_dtype)
                B_scale = (calculate_dtype(2.0) ** (B_scale - 127)).astype(calculate_dtype)
                A_scale = np.nan if A_scale == 0 else A_scale
                B_scale = np.nan if B_scale == 0 else B_scale
                
                # Compute dot product
                dot = np.sum(A_block * B_block * A_scale * B_scale, dtype=calculate_dtype)
                total += dot
            C[i, j] = np.float32(total)
    
    return C

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
    elif name in ['scale_A', 'scale_B']:
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

def parse_scale_data(data):
    def parse_hex_to_int8(hex_str):
        """将十六进制字符串转换为int8整数"""
        num = int(hex_str, 16)
        return num
    
    # 初始化存储结构
    scale_A = []
    scale_B = []
    current_scale = None
    
    # 正则表达式匹配数据行中的所有十六进制值
    hex_pattern = r'0x[0-9A-Fa-f]{2}'
    
    for line in data.split('\n'):
        line = line.strip()
        
        # 检测scale标识
        if line.startswith('scale A'):
            current_scale = 'A'
            continue
        elif line.startswith('scale B'):
            current_scale = 'B'
            continue
            
        # 处理数据行 - 匹配所有十六进制值
        if line and line[0].isdigit() and '[' in line:
            hex_values = re.findall(hex_pattern, line)
            if not hex_values:
                continue
                
            # 将十六进制值转换为int8
            int8_values = [parse_hex_to_int8(hex_val) for hex_val in hex_values]
            
            # 根据当前处理的scale添加到相应列表
            if current_scale == 'A':
                scale_A.append(int8_values)
            elif current_scale == 'B':
                scale_B.append(int8_values)

    if scale_A:
        n = len(scale_A[0])
        scale_A = np.array(scale_A).reshape(-1, n)
    else:
        scale_A = np.array([])

    if scale_B:
        n = len(scale_B[0])
        scale_B = np.array(scale_B).reshape(-1, n)
    else:
        scale_B = np.array([])
    return scale_A.astype(np.uint8), scale_B.astype(np.uint8)

def main():
    """
    主函数，执行整个流程。
    """

    # 读取日志文件
    with open('test.log', 'r', encoding='utf-8') as f:
        log_content = f.read()

    # --- 解析矩阵 ---
    mat_a = parse_matrix_from_log(log_content, 'Mat_A', 2048, 256)
    mat_b = parse_matrix_from_log(log_content, 'Mat_B', 32, 256)
    mat_c_from_log = parse_matrix_from_log(log_content, 'Mat_C', 2048, 32)
    a_scale, b_scale = parse_scale_data(log_content)
    # --- 执行矩阵乘法 ---
    # calculated_mat_c = mxfp4_e8m0_matmul(16, mat_a, mat_b, a_scale, b_scale)
    # calculated_mat_c = mxfp4_e8m0_matmul(32, mat_a, mat_b, a_scale, b_scale)
    calculated_mat_c = mxfp4_e4m3_matmul(16, mat_a, mat_b, a_scale, b_scale)
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
        c_log_skipped_inf = np.where(np.isinf(mat_c_from_log), 0, mat_c_from_log)
        c_calu_skipped_inf = np.where(np.isinf(calculated_mat_c), 0, calculated_mat_c)
        difference = np.mean(np.abs(c_calu_skipped_inf - c_log_skipped_inf))
        print(f"平均差值: {difference}")
    # 检查mat_c_from_log与calculated_mat_c在相同索引下inf不一致的情况
        log_inf = np.isinf(mat_c_from_log)
        calu_inf = np.isinf(calculated_mat_c)
        mismatch = np.where(log_inf != calu_inf)
        if mismatch[0].size > 0:
            for idx in zip(*mismatch):
                print(f"inf index: {idx}, mat_c_from_log value: {mat_c_from_log[idx]}, calculated_mat_c value: {calculated_mat_c[idx]}")
                break

if __name__ == '__main__':
    main()
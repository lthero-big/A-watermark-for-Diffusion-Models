def compare_bit_accuracy(file1, file2):
    # 打开并读取两个文件
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # 使用逗号分割数据，得到列表
        data1 = f1.read().strip().split(',')
        data2 = f2.read().strip().split(',')
    
    # 确保比较的长度一致，根据较短的列表截断较长的列表
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    
    # 统计匹配的位数
    matching_bits = sum(b1 == b2 for b1, b2 in zip(data1, data2))
    
    # 计算位比特正确率
    accuracy = matching_bits / min_length
    
    return accuracy

# 示例使用
file1 = 'inserted.txt'
file2 = 'output.txt'
accuracy = compare_bit_accuracy(file1, file2)
print(f'位比特正确率: {accuracy:.2%}')

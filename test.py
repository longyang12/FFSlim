import bisect

# 示例数据
pre_sums = [(5, 0, 4), (10, 5, 9), (15, 10, 14), (20, 15, 19)]

def find_index(pre_sums, x):
    # 提取所有的“最大编号”创建一个虚拟数组（或者直接针对元组查找）
    # 我们查找第一个“最大编号 >= x”的元素
    # 由于元组比较是逐个字段进行的，我们可以利用这一点
    
    # 方式 A：手动二分（性能最强，无额外空间）
    low = 0
    high = len(pre_sums) - 1
    
    while low <= high:
        mid = (low + high) // 2
        node_sum, node_min, node_max = pre_sums[mid]
        
        if node_min <= x <= node_max:
            return mid
        elif x < node_min:
            high = mid - 1
        else:
            low = mid + 1
    return -1

# 测试
x = 12
idx = find_index(pre_sums, x)
print(f"编号 {x} 在下标: {idx}, 对应元组: {pre_sums[idx]}")
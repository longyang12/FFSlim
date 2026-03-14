# def calculate_initial_capacity(sample_count, avg_size, total_memory):
#     """计算初始缓存容量
#     :param sample_count: 数据集对象总数
#     :param avg_size: 平均对象大小（字节）
#     :param total_memory: 可用内存总量（字节）
#     """
#     # 基础容量 = 内存总量 / 平均对象大小
#     base_capacity = total_memory // avg_size
        
#     # 根据数据集规模动态调整[3](@ref)
#     x1 = 0.5  # 收益最大的拐点位置，根据经验
#     x2 = 0.5  # 最多50%内存用于缓存，根据经验
    
#     return min(sample_count * x1, int(base_capacity * x2))

# def adjust_capacity(hit_rate, capacity, avg_size, total_memory):
#     """动态调整缓存容量"""
#     current_usage = capacity * avg_size / total_memory
#     l1 = 0.3
#     h1 = 0.8
#     h2 = 0.7
#     if hit_rate < l1 and current_usage < h2:  # 低命中率+低缓存分配
#         new_capacity = min(int(total_memory/avg_size), int(capacity * 1.1))  # 扩容10%
#     elif hit_rate > h1 and current_usage > h2:  # 高命中率+高缓存分配
#         new_capacity = int(capacity * 0.9)      # 缩容10%
#     elif hit_rate < l1 and current_usage > h2:  # 低命中率+高缓存分配
#         new_capacity = 0  # 缓存无效，禁用缓存
#     else:  # 稳定状态  高命中率+低缓存分配
#         return
        
#     return new_capacity



import numpy as np
import matplotlib.pyplot as plt

def plot_kb_distribution(filename):
    """
    读取txt文件中的数字（单位为字节），转换为KB，自动划分区间并绘制分布直方图
    
    :param filename: txt文件路径
    """
    try:
        # 1. 读取文件内容
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        if not content:
            print("文件为空，无法处理。")
            return
        
        # 2. 提取数字并转换为数值列表（字节单位）
        byte_numbers = []
        for num_str in content.split():
            try:
                byte_numbers.append(float(num_str))
            except ValueError:
                print(f"警告：跳过无效数据 '{num_str}'")
                continue
        
        if not byte_numbers:
            print("未找到有效数字。")
            return
        
        # 3. 将字节转换为KB (1 KB = 1024 bytes)
        if size != "text_size":
            kb_numbers = np.array(byte_numbers) / 1024
        else:
            kb_numbers = np.array(byte_numbers)

        # 4. 使用NumPy自动计算区间和频数 (使用 'auto' 算法选择最优区间)
        counts, bin_edges = np.histogram(kb_numbers, bins='auto')
        
        # 5. 计算区间中点（作为x轴标签位置）
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # 生成区间范围标签
        bin_ranges = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
        
        # 6. 绘制条形图
        plt.figure(figsize=(12, 7))
        plt.bar(bin_centers, counts, width=np.diff(bin_edges)*0.8, 
                      edgecolor='black', linewidth=0.7, alpha=0.7, color='skyblue')
        
        # 8. 添加图表标签和标题
        if size != "text_size":
            plt.xlabel('Data Size Range (KB)', fontsize=12, labelpad=10)
        else:
            plt.xlabel('Data Size Range (B)', fontsize=12, labelpad=10)

        plt.ylabel('Frequency', fontsize=12, labelpad=10)
        plt.title('Histogram of data size distribution with automatic binning.', fontsize=14, pad=20)
        
        # 10. 添加网格线便于阅读
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
        # 显示图表
        plt.savefig(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_{size}_distributed.png")
        plt.show()
        
        # 打印一些统计信息
        print(f"原始数据范围: {min(byte_numbers):.2f} - {max(byte_numbers):.2f} 字节")
        if size != "text_size":
            print(f"转换后范围: {min(kb_numbers):.2f} - {max(kb_numbers):.2f} KB")
        print(f"自动划分的区间数量: {len(counts)}")
        print("\n各区间详情:")
        for i, (bin_range, count) in enumerate(zip(bin_ranges, counts)):
            print(f"区间 {i+1}: {bin_range} KB → {count} 个数据点")
            
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'")
    except Exception as e:
        print(f"处理文件时出错: {e}")


# 使用示例
if __name__ == "__main__":
    for dataset in ["flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "cc3m", "MSRVTT", "Clotho", "Clotho_E"]:
        for size in ["media_size", "text_size", "sample_size"]:
            plot_kb_distribution(f'/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_{size}_nums.txt')  # 替换为您的txt文件路径
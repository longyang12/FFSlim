import matplotlib.pyplot as plt
from collections import Counter

def plot_number_frequency(filename):
    """
    读取txt文件中的数字，统计频率并绘制条形图
    :param filename: txt文件路径
    """
    try:
        # 读取文件内容
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read().strip()  # 读取整行并去除首尾空白字符[6](@ref)
        
        # 检查文件是否为空
        if not content:
            print("文件为空，无法处理。")
            return
        
        # 分割字符串并转换为数字列表[2,6](@ref)
        numbers = []
        for num_str in content.split():  # 默认以空格分割[2](@ref)
            try:
                numbers.append(int(num_str))  # 转换为整数
            except ValueError:
                print(f"警告：跳过无效数据 '{num_str}'")
        
        if not numbers:
            print("未找到有效数字。")
            return
        
        # 统计数字频率[7](@ref)
        freq_counter = Counter(numbers)
        
        # 准备绘图数据（按数字排序以便图表更清晰）
        sorted_numbers = sorted(freq_counter.keys())
        frequencies = [freq_counter[num] for num in sorted_numbers]
        
        # 绘制条形图[7](@ref)
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_numbers, frequencies, color='skyblue', edgecolor='black')
        
        # 添加标签和标题
        plt.xlabel('media-text ratio', fontsize=12)
        plt.ylabel('counts', fontsize=12)
        plt.title('media-text ratio distributed', fontsize=14)
        
        # 添加网格线以便阅读
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 显示图表
        plt.savefig(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_ratio_distributed.png")
        plt.show()
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'，请检查文件路径。")
    except IOError:
        print(f"错误：读取文件 '{filename}' 时发生I/O错误。")
    except Exception as e:
        print(f"发生未知错误：{e}")

# 使用示例
if __name__ == "__main__":
    for dataset in ["flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "cc3m", "MSRVTT", "Clotho", "Clotho_E"]:
        plot_number_frequency(f'/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_ratio_nums.txt')  # 替换为您的txt文件路径
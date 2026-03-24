import re

def analyze_detailed_forgetting(file_path):
    total_count = 0
    total_dynamic_forget = 0  # 增加
    total_static_forget = 0   # 不变
    
    # 正则表达式匹配 key 和 score_shifts 数组
    pattern = re.compile(r"key:\d+,score_shifts:\[(.*?)\]")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                match = pattern.search(line)
                if match:
                    # 转换字符串为整数列表
                    shifts = [int(x.strip()) for x in match.group(1).split(',')]
                    
                    # 累计总次数
                    total_count += len(shifts)
                    
                    # 遍历数组进行对比 (从第二个元素开始)
                    for i in range(1, len(shifts)):
                        current = shifts[i]
                        previous = shifts[i-1]
                        
                        if current > previous:
                            total_dynamic_forget += 1
                        elif current == previous:
                            total_static_forget += 1
                            
        # 计算比例
        total_forgetting = total_dynamic_forget + total_static_forget
        dynamic_rate = (total_dynamic_forget / total_count) if total_count > 0 else 0
        static_rate = (total_static_forget / total_count) if total_count > 0 else 0
        overall_forget_rate = (total_forgetting / total_count) if total_count > 0 else 0

        # 输出格式化报告
        print("="*30)
        print("  详细遗忘数据汇总分析")
        print("="*30)
        print(f"1. 总体样本数据：")
        print(f"   - 总处理数据点次数: {total_count}")
        print("-" * 30)
        print(f"2. 遗忘分类统计：")
        print(f"   - 动态低价值遗忘 (增加): {total_dynamic_forget} 次")
        print(f"   - 静态低价值遗忘 (不变): {total_static_forget} 次")
        print(f"   - 累计总遗忘次数: {total_forgetting} 次")
        print("-" * 30)
        print(f"3. 遗忘率分析：")
        print(f"   - 动态遗忘率: {dynamic_rate:.4%}")
        print(f"   - 静态遗忘率: {static_rate:.4%}")
        print(f"   - 总体遗忘率: {overall_forget_rate:.4%}")
        print("="*30)

    except FileNotFoundError:
        print(f"错误：无法找到文件 '{file_path}'")
    except Exception as e:
        print(f"分析过程中出错: {e}")


if __name__ == "__main__":
    # FFSlim dgim_iacma
    # datasets=("cc3m", "flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "MSRVTT", "Clotho", "Clotho_E")
    # counts=("102751", "12405", "12713", "47183", "33109", "28856", "4000", "1575", "7876")
    # for dataset, count in zip(datasets, counts):
    #     analyze_detailed_forgetting(f'/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/{dataset}_FFSlim_{count}_dgim_iacma_object_importance.log')

    # # TDP dgim_iacma
    # datasets=("cc3m", "flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "MSRVTT", "Clotho", "Clotho_E")
    # counts=("102751", "12405", "12713", "47183", "33109", "28856", "4000", "1575", "7876")
    # for dataset, count in zip(datasets, counts):
    #     analyze_detailed_forgetting(f'/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/{dataset}_TDP_{count}_iacma_object_importance.log')
    

    # # flickr30k_FFSlim dgim_iacma
    # for num in [3101, 6203, 9304, 12405, 15507, 18608, 21709, 24810, 27912, 31013]:
    #     analyze_detailed_forgetting(f'/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/flickr30k_FFSlim_{num}_dgim_iacma_object_importance.log')
    
    # # gqa_FFSlim dgim_iacma
    # for num in [7214, 14428, 21642, 28856, 36070, 43283, 50497, 57711, 64925, 72139]:
    #     analyze_detailed_forgetting(f'/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/gqa_FFSlim_{num}_dgim_iacma_object_importance.log')
    
    # flickr30k_TDP iacma
    for num in [3101, 6203, 9304, 12405, 15507, 18608, 21709, 24810, 27912, 31013]:
        analyze_detailed_forgetting(f'/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/flickr30k_TDP_{num}_iacma_object_importance.log')
    
    # # gqa_TDP iacma
    # for num in [7214, 14428, 21642, 28856, 36070, 43283, 50497, 57711, 64925, 72139]:
    #     analyze_detailed_forgetting(f'/home/llm/experiment/dataset_Analysis_Modeling/object_importance_log/gqa_TDP_{num}_iacma_object_importance.log')

    
from ffrecord import FileReader
import pickle
import os
import statistics
import pandas as pd


output_dir = "/home/llm/experiment/dataset_Analysis_Modeling/analyze/output"
results = []


for dataset in ["cc3m","flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "MSRVTT", "Clotho", "Clotho_E"]:
    file_path = f"/mnt/datasets/{dataset}_FFSlim/mscoco_caption.ffr"

    sample_num = 0
    media_num = 0
    avg_sample_size = 0
    avg_media_size = 0
    avg_text_size = 0
    avg_quantity_ratio = 0
    avg_size_ratio = 0
    std_sample_size = 0
    std_media_size = 0
    std_text_size = 0
    std_quantity_ratio = 0
    std_size_ratio = 0

    sample_size_nums = []
    media_size_nums = []
    text_size_nums = []
    ratio_nums = []

    # 如果文件已经存在直接从文件中读取数据，否则进行计算并保存到文件中
    if os.path.exists(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_ratio_nums.txt"):
        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_ratio_nums.txt","r") as f:
            ratio_nums = list(map(int, f.read().split()))

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_media_size_nums.txt","r") as f:
            media_size_nums = list(map(int, f.read().split()))

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_text_size_nums.txt","r") as f:
            text_size_nums = list(map(int, f.read().split()))

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_sample_size_nums.txt","r") as f:
            sample_size_nums = list(map(int, f.read().split()))
    else:
        reader = FileReader(file_path)
        for i in range(reader.n):
            datasets = reader.read([i])
            for sample in datasets:
                samples1 = pickle.loads(sample)
                ratio_nums.append(len(samples1["captions"]))    
                media_size = len(samples1["image"])   
                media_size_nums.append(media_size)
                for caption in samples1["captions"]:
                    text_size = len(caption)
                    text_size_nums.append(text_size)
                    sample_size_nums.append(media_size+text_size)

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_ratio_nums.txt","w") as f:
            for num in ratio_nums:
                    f.write(f"{num} ")

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_media_size_nums.txt","w") as f:
            for num in media_size_nums:
                    f.write(f"{num} ")

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_text_size_nums.txt","w") as f:
            for num in text_size_nums:
                f.write(f"{num} ")

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_sample_size_nums.txt","w") as f:
            for num in sample_size_nums:
                    f.write(f"{num} ")

    sample_num = len(sample_size_nums)
    media_num = len(media_size_nums)
    avg_sample_size = sum(sample_size_nums)/len(sample_size_nums)
    avg_media_size = sum(media_size_nums)/len(media_size_nums)
    avg_text_size = sum(text_size_nums)/len(text_size_nums)
    avg_quantity_ratio = sum(ratio_nums)/len(ratio_nums)
    avg_size_ratio = avg_media_size/avg_text_size

    std_sample_size = statistics.pstdev(sample_size_nums)
    std_media_size = statistics.pstdev(media_size_nums)
    std_text_size = statistics.pstdev(text_size_nums)
    std_quantity_ratio = statistics.pstdev(ratio_nums)
    size_ratio_nums = [
        (sample_size_nums[i] - text_size_nums[i]) / text_size_nums[i]
        for i in range(len(sample_size_nums)) if text_size_nums[i] > 0
    ]
    std_size_ratio = statistics.pstdev(size_ratio_nums) if size_ratio_nums else 0

    # 输出可以直接复制到excel的格式，均值延续原格式，标准差保留2位小数
    if dataset == "cc3m":
        print(
            f"dataset_FFSlim：\tsample_num\tmedia_num\tavg_sample_size\tstd_sample_size\tavg_media_size\tstd_media_size\tavg_text_size\tstd_text_size\tavg_quantity_ratio\tstd_quantity_ratio\tavg_size_ratio\tstd_size_ratio"
        )
    print(
        f"{dataset}_FFSlim：\t{sample_num}\t{media_num}\t{int(avg_sample_size)}\t{std_sample_size:.2f}\t{int(avg_media_size)}\t{std_media_size:.2f}\t{int(avg_text_size)}\t{std_text_size:.2f}\t{avg_quantity_ratio:.1f}\t{std_quantity_ratio:.2f}\t{int(avg_size_ratio)}\t{std_size_ratio:.2f}"
    )

    results.append(
        {
            "dataset": f"{dataset}_FFSlim",
            "sample_num": sample_num,
            "media_num": media_num,
            "avg_sample_size": int(avg_sample_size),
            "std_sample_size": round(std_sample_size, 2),
            "avg_media_size": int(avg_media_size),
            "std_media_size": round(std_media_size, 2),
            "avg_text_size": int(avg_text_size),
            "std_text_size": round(std_text_size, 2),
            "avg_quantity_ratio": round(avg_quantity_ratio, 1),
            "std_quantity_ratio": round(std_quantity_ratio, 2),
            "avg_size_ratio": int(avg_size_ratio),
            "std_size_ratio": round(std_size_ratio, 2),
        }
    )

df = pd.DataFrame(results)
excel_path = os.path.join(output_dir, "FFSlim_statistics.xlsx")
df.to_excel(excel_path, index=False)
print(f"Excel已输出：{excel_path}")
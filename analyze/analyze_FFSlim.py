from ffrecord import FileReader
import pickle
import os


for dataset in ["flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "cc3m", "MSRVTT", "Clotho", "Clotho_E"]:
    file_path = f"/mnt/datasets/{dataset}_FFSlim/mscoco_caption.ffr"

    sample_num = 0
    media_num = 0
    avg_sample_size = 0
    avg_media_size = 0
    avg_text_size = 0
    avg_quantity_ratio = 0
    avg_size_ratio = 0

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

    # 输出可以直接复制到excel的格式,avg_quantity_ratio保留1位小数、其他保留整数
    if dataset == "flickr30k":
        print(f"dataset_FFSlim：\tsample_num\tmedia_num\tavg_sample_size\tavg_media_size\tavg_text_size\tavg_quantity_ratio\tavg_size_ratio")
    print(f"{dataset}_FFSlim：\t{sample_num}\t{media_num}\t{int(avg_sample_size)}\t{int(avg_media_size)}\t{int(avg_text_size)}\t{avg_quantity_ratio:.1f}\t{int(avg_size_ratio)}")
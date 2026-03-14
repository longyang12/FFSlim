import pandas as pd
import os
        
for dataset in ["flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "cc3m", "MSRVTT", "Clotho", "Clotho_E"]:
    file_path = f"/mnt/datasets/{dataset}_TDP/{dataset}_TDP.csv"

    
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
    if os.path.exists(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_ratio_nums.txt"):
        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_ratio_nums.txt","r") as f:
            ratio_nums = list(map(int, f.read().split()))

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_media_size_nums.txt","r") as f:
            media_size_nums = list(map(int, f.read().split()))

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_text_size_nums.txt","r") as f:
            text_size_nums = list(map(int, f.read().split()))

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_sample_size_nums.txt","r") as f:
            sample_size_nums = list(map(int, f.read().split()))
    else:
        #解析csv文件(image_path,image_caption),将image_path相同的聚合在一起,并计算每个样本的媒体大小、文本大小、样本大小、数量比和大小比，图像路径为相对路径，base_path为"/mnt/datasets/{dataset}_TDP/img/“
        # 注意一个media加一个text算一个样本，聚合只是为了方便算比例
        df = pd.read_csv(file_path)
        grouped = df.groupby('image_path')['image_caption'].apply(list).reset_index()
        for index, row in grouped.iterrows():
            image_path = row['image_path']
            captions = row['image_caption']
            media_size = os.path.getsize(f"/mnt/datasets/{dataset}_TDP/img/{image_path}")
            media_size_nums.append(media_size)
            for caption in captions:
                text_size = len(caption)
                text_size_nums.append(text_size)
                sample_size_nums.append(media_size+text_size)
            ratio_nums.append(len(captions))


        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_ratio_nums.txt","w") as f:
            for num in ratio_nums:
                    f.write(f"{num} ")

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_media_size_nums.txt","w") as f:
            for num in media_size_nums:
                    f.write(f"{num} ")

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_text_size_nums.txt","w") as f:
            for num in text_size_nums:
                f.write(f"{num} ")

        with open(f"/home/llm/experiment/dataset_Analysis_Modeling/analyze/output/{dataset}_TDP_sample_size_nums.txt","w") as f:
            for num in sample_size_nums:
                    f.write(f"{num} ")

    sample_num = len(sample_size_nums)
    media_num = len(media_size_nums)
    avg_sample_size = sum(sample_size_nums)/len(sample_size_nums)
    avg_media_size = sum(media_size_nums)/len(media_size_nums)
    avg_text_size = sum(text_size_nums)/len(text_size_nums)
    avg_quantity_ratio = sum(ratio_nums)/len(ratio_nums)
    avg_size_ratio = avg_media_size/avg_text_size

    # 输出可以直接复制到excel的格式,avg_quantity_ratio保留1位小数、其他保留整数,输出标题
    if dataset == "flickr30k":
        print(f"dataset_TDP：\tsample_num\tmedia_num\tavg_sample_size\tavg_media_size\tavg_text_size\tavg_quantity_ratio\tavg_size_ratio")
    print(f"{dataset}_TDP：\t{sample_num}\t{media_num}\t{int(avg_sample_size)}\t{int(avg_media_size)}\t{int(avg_text_size)}\t{avg_quantity_ratio:.1f}\t{int(avg_size_ratio)}")
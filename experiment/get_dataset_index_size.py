import os
from ffrecord import FileReader


# datasets = ["flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "cc3m"]
datasets = ["Clotho_E"]

for dataset in datasets:
    # TDP
    file_path = "/mnt/datasets/" + dataset + "_TDP/" + dataset +"_TDP.csv"
    file_size = os.path.getsize(file_path)
    print(f"{dataset}_TDP:index_size:{file_size}")

    # FFRecord
    file_path = "/mnt/datasets/" + dataset + "_FFRecord/mscoco_caption.ffr"
    reader = FileReader(file_path)
    print(f"{dataset}_FFRecord:index_size:{reader.n*4}")

    # FFSlim
    if dataset in ["vqa2", "gqa"]:
        file_path = "/mnt/datasets/" + dataset + "_FFSlim/mscoco_caption.ffr"
        reader = FileReader(file_path)
        file_size = os.path.getsize("/mnt/datasets/" + dataset + "_FFSlim/" + dataset +".pkl")
        print(f"{dataset}_FFSlim:index_size:{reader.n*4+file_size}")
    else:
        file_path = "/mnt/datasets/" + dataset + "_FFSlim/mscoco_caption.ffr"
        reader = FileReader(file_path)
        print(f"{dataset}_FFSlim:index_size:{reader.n*4}")


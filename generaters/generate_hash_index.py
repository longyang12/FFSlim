from ffrecord import FileReader
import pickle
import os



for dataset in ["cc3m","flickr30k", "flickr30k_entities", "mscoco", "vqa2", "gqa", "MSRVTT", "Clotho", "Clotho_E"]:
    file_path = f"/mnt/datasets/{dataset}_FFSlim/mscoco_caption.ffr"

    # 如果已经生成过则读取
    if os.path.exists(f"/mnt/datasets/{dataset}_FFSlim/{dataset}-hash.pkl"):
        with open(f'/mnt/datasets/{dataset}_FFSlim/{dataset}-hash.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        hash_index = loaded_data["hash_index"]
        print(f"dataset:{dataset}, hash_index size:{len(hash_index)}")
    else:
        reader = FileReader(file_path)
        samples = [i for i in range(reader.n)]
        batch_size = 64
        n = len(samples)//batch_size
        m = len(samples)%batch_size
        batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(0,n)]
        if m > 0:
            batch_samples_list.append(samples[n*batch_size:])
        print(f"n={n},m={m}")


        hash_index = dict()
        key_index = 0
        Media2Text_index = 0
        for batch in batch_samples_list:
            datasets = reader.read(batch)
            for sample in datasets:
                obj = pickle.loads(sample)
                captions = obj["captions"]
                n = len(captions)
                for i in range(n):
                    hash_index[key_index] = (Media2Text_index,i)
                    key_index += 1
                Media2Text_index += 1

        data = {"hash_index":hash_index}
        with open(f'/mnt/datasets/{dataset}_FFSlim/{dataset}-hash.pkl', 'wb') as f:
            pickle.dump(data, f)



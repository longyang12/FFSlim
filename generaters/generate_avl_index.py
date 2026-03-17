from ffrecord import FileReader
import pickle
import os


for dataset in ["vqa2", "gqa"]:
    file_path = f"/mnt/datasets/{dataset}_FFSlim/mscoco_caption.ffr"
    reader = FileReader(file_path)

    # 如果已经生成过则读取
    if os.path.exists(f"/mnt/datasets/{dataset}_FFSlim/{dataset}-avl.pkl"):
        with open(f'/mnt/datasets/{dataset}_FFSlim/{dataset}-avl.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        pre_sums = loaded_data["pre_sums"]
        print(f"dataset={dataset},reader.n={reader.n},len(pre_sums)={len(pre_sums)}")
    else:
        samples = [i for i in range(reader.n)]
        batch_size = 64
        n = len(samples)//batch_size
        m = len(samples)%batch_size
        batch_samples_list = [samples[i*batch_size:i*batch_size+batch_size] for i in range(0,n)]
        if m > 0:
            batch_samples_list.append(samples[n*batch_size:])
        print(f"n={n},m={m}")

        pre_sums = []
        index = 0
        for batch in batch_samples_list:
            datasets = reader.read(batch)
            for sample in datasets:
                obj = pickle.loads(sample)
                captions = obj["captions"]
                n = len(captions)
                if len(pre_sums) == 0:
                    pre_sums.append((0,n-1))
                else:
                    pre_sums.append((pre_sums[-1][1]+1,pre_sums[-1][1]+n)) # (start id,end id) ...
                index += 1

        print(f"dataset={dataset},reader.n={reader.n},len(pre_sums)={len(pre_sums)}")

        data = {"pre_sums":pre_sums}
        with open(f'/mnt/datasets/{dataset}_FFSlim/{dataset}-avl.pkl', 'wb') as f:
            pickle.dump(data, f)




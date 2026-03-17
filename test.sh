# !/bin/bash

# 将这两行变成数组，并且遍历打印他们


echo $1 $2 $3

# datasets=("cc3m" "flickr30k" "flickr30k_entities" "mscoco" "vqa2" "gqa" "MSRVTT" "Clotho" "Clotho_E")
# counts=("102,751" "12,405" "12,713" "47,183" "33,109" "28,856" "4,000" "1,575" "7,876")

# for i in "${!datasets[@]}"; do
#     for cache_algorithm in fgfglru quiver coordl lru iacma iacma_with_memory; do
#         echo " sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py ${datasets[$i]} ${counts[$i]} ${cache_algorithm} adaptive_index  "
#     done
# done

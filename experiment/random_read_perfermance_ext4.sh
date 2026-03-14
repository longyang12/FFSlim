# 使用 tmux
# tmux new-session -s my_session  # 创建新会话
# python long_task.py             # 在会话中运行任务
# # 断开连接后，重连时恢复会话：
# tmux attach-session -t my_session

# 设置文件系统参数
# sudo umount /mnt/datasets
# sudo tune2fs -l /dev/nvme0n1
# sudo tune2fs -O ^dir_index /dev/nvme0n1
# # sudo tune2fs -O dir_index /dev/nvme0n1
# sudo e2fsck -f -D /dev/nvme0n1
# sudo tune2fs -l /dev/nvme0n1
# sudo mount /dev/nvme0n1 /mnt/datasets

# 设置page cache预取大小
sudo blockdev --getra /dev/nvme2n1
sudo blockdev --setra 0 /dev/nvme2n1
# sudo blockdev --setra 2048 /dev/nvme2n1
sudo blockdev --getra /dev/nvme2n1


###############################
######### data format #########
###############################

# for dataset in flickr30k flickr30k_entities mscoco vqa2 gqa cc3m MSRVTT Clotho Clotho_E
# do
#     for num in 0 
#     do
#         echo "当前数值: $num"
#         # 在此处添加处理逻辑
#         echo "lru"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py $dataset $num lru
#         while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#             sleep 1  # 每秒检查一次
#         done
#         echo "cgroup 内所有进程已结束"
#     done
# done

for dataset in flickr30k flickr30k_entities mscoco vqa2 gqa cc3m MSRVTT Clotho Clotho_E
do
    for num in 0 
    do
        echo "当前数值: $num"
        # 在此处添加处理逻辑
        echo "lru"
        sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py $dataset $num lru
        # while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
        #     sleep 1  # 每秒检查一次
        # done
        # echo "cgroup 内所有进程已结束"
    done
done


# # data format
# # FFRecord
# for num in 0 3101 6203 9304 12406 15507 18608 21710 24811 27913 31014
# for num in 0 
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_ffrecord_loader.py flickr30k $num lru
#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"
# done

# for num in 0 3178 6357 9535 12713 15892 19070 22248 25426 28605 31783
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_ffrecord_loader.py flickr30k_entities $num lru
# done
 
# for num in 0 11797 23594 35392 47189 58986 70783 82580 94378 106175 117972
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_ffrecord_loader.py mscoco $num lru
# done

# for num in 0 8218 16436 24653 32871 41089 49307 57525 65742 73960 82178
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_ffrecord_loader.py vqa2 $num lru
# done

# for num in 0 7254 14508 21762 29016 36270 43523 50777 58031 65285 72539
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_ffrecord_loader.py gqa $num lru
# done

# # Files
# for num in 0 3101 6203 9304 12406 15507 18608 21710 24811 27913 31014
# for num in 0
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_files_loader.py flickr30k $num lru
# done

# for num in 0 3178 6357 9535 12713 15892 19070 22248 25426 28605 31783
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_files_loader.py flickr30k_entities $num lru
# done
 
# for num in 0 11797 23594 35392 47189 58986 70783 82580 94378 106175 117972
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_files_loader.py mscoco $num lru
# done

# for num in 0 8218 16436 24653 32871 41089 49307 57525 65742 73960 82178
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_files_loader.py vqa2 $num lru
# done

# for num in 0 7254 14508 21762 29016 36270 43523 50777 58031 65285 72539
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_files_loader.py gqa $num lru
# done

# # TDP
# for num in 0 3101 6203 9304 12406 15507 18608 21710 24811 27913 31014
# for num in 0
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py flickr30k $num lru
# done

# for num in 0 3178 6357 9535 12713 15892 19070 22248 25426 28605 31783
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py flickr30k_entities $num lru
# done
 
 
# for num in 0 11797 23594 35392 47189 58986 70783 82580 94378 106175 117972
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py mscoco $num lru
# done

# for num in 0 8218 16436 24653 32871 41089 49307 57525 65742 73960 82178
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py vqa2 $num lru
# done

# for num in 0 7254 14508 21762 29016 36270 43523 50777 58031 65285 72539
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py gqa $num lru
# done


# # FFSlim
# for num in 0 3101 6203 9304 12406 15507 18608 21710 24811 27913 31014
# for num in 0
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py flickr30k $num lru
#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"
# done

# for num in 0 3178 6357 9535 12713 15892 19070 22248 25426 28605 31783
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py flickr30k_entities $num lru
# done
 
 
# for num in 0 11797 23594 35392 47189 58986 70783 82580 94378 106175 117972
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py mscoco $num lru
# done

# for num in 0 8218 16436 24653 32871 41089 49307 57525 65742 73960 82178
# for num in 0
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py vqa2 $num lru
#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"    
# done

# for num in 0 7254 14508 21762 29016 36270 43523 50777 58031 65285 72539
# for num in 0
# do
#     echo "当前数值: $num"
#     # 在此处添加处理逻辑
#     echo "lru"
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py gqa $num lru
#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"
# done


# Cache
# TDP flickr30k
# lru lfu quiver coordl iacma iacma_with_memory
# for cache_algorithm in coordl iacma_with_memory
# do
#     for num in 3101 6203 9304 12406 15507 18608 21710 24811 27913 31014
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py flickr30k $num $cache_algorithm
#     done
# done

# # TDP flickr30k_entities
# for cache_algorithm in coordl iacma_with_memory
# do
#     for num in 3178 6357 9535 12713 15892 19070 22248 25426 28605 31783
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py flickr30k_entities $num $cache_algorithm
#     done
# done

# # TDP mscoco
# for cache_algorithm in lru lfu quiver coordl iacma iacma_with_memory
# do
#     for num in 11797 23594 35392 47189 58986 70783 82580 94378 106175 117972
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py mscoco $num $cache_algorithm
#     done
# done

# # TDP vqa2
# for cache_algorithm in lru lfu quiver coordl iacma iacma_with_memory
# do
#     for num in 8218 16436 24653 32871 41089 49307 57525 65742 73960 82178
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py vqa2 $num $cache_algorithm
#     done
# done

# TDP gqa
# for cache_algorithm in lru lfu quiver coordl iacma iacma_with_memory
# for cache_algorithm in iacma iacma_with_memory
# do
#     for num in 7254 14508 21762 29016 36270 43523 50777 58031 65285 72539
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py gqa $num $cache_algorithm
#     done
# done

# # TDP cc3m
# for cache_algorithm in iacma iacma_with_memory
# do
#     for num in 25733 51466 21762 77198 102931 128664 154397 205862 231595 257328
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py cc3m $num $cache_algorithm
#     done
# done

# TDP MSRVTT
# for cache_algorithm in lru lfu quiver coordl iacma iacma_with_memory
for cache_algorithm in lfu quiver coordl
do
    for num in 4000
    do
        echo "当前算法: $cache_algorithm"
        echo "当前缓存大小: $num"
        sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py MSRVTT $num $cache_algorithm
        # while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
        #     sleep 1  # 每秒检查一次
        # done
        # echo "cgroup 内所有进程已结束"
    done
done

# # TDP Clotho
# for cache_algorithm in lru lfu quiver coordl iacma iacma_with_memory
# do
#     for num in 1969 3938 5907 7876 9845 11814 13783 15752 17721 19690
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py Clotho $num $cache_algorithm
#         while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#             sleep 1  # 每秒检查一次
#         done
#         echo "cgroup 内所有进程已结束"
#     done
# done

# TDP Clotho_E
# for cache_algorithm in lru lfu quiver coordl iacma iacma_with_memory
# do
#     for num in 1969 3938 5907 7876 9845 11814 13783 15752 17721 19690
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader.py Clotho_E $num $cache_algorithm
#         while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#             sleep 1  # 每秒检查一次
#         done
#         echo "cgroup 内所有进程已结束"
#     done
# done

# FFSlim flickr30k
# fgfglru fgfglfu quiver coordl lru lfu iacma iacma_with_memory
# for cache_algorithm in fgfglfu
# do
#     # for num in 3101 6203 9304 12406 15507 18608 21710 24811 27913 31014
#     for num in 12406
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py flickr30k $num $cache_algorithm
#         while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#             sleep 1  # 每秒检查一次
#         done
#         echo "cgroup 内所有进程已结束"
#     done
# done

# # FFSlim flickr30k_entities
# for cache_algorithm in quiver coordl iacma iacma_with_memory
# do
#     for num in 3178 6357 9535 12713 15892 19070 22248 25426 28605 31783
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py flickr30k_entities $num $cache_algorithm
#     done
# done

# # FFSlim mscoco
# for cache_algorithm in quiver coordl iacma iacma_with_memory
# do
#     for num in 11797 23594 35392 47189 58986 70783 82580 94378 106175 117972
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py mscoco $num $cache_algorithm
#     done
# done

# # FFSlim vqa2
# for cache_algorithm in quiver coordl iacma iacma_with_memory
# do
#     for num in 8218 16436 24653 32871 41089 49307 57525 65742 73960 82178
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py vqa2 $num $cache_algorithm
#     done
# done

# FFSlim gqa
# for cache_algorithm in iacma iacma_with_memory
# do
#     for num in 7254 14508 21762 29016 36270 43523 50777 58031 65285 72539
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py gqa $num $cache_algorithm
#         while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#             sleep 1  # 每秒检查一次
#         done
#         echo "cgroup 内所有进程已结束"
#     done
# done

# # FFSlim cc3m
# for cache_algorithm in quiver coordl iacma iacma_with_memory
# do
#     for num in 25733 51466 21762 77198 102931 128664 154397 205862 231595 257328
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py cc3m $num $cache_algorithm
#     done
# done

# FFSlim MSRVTT
# for cache_algorithm in lru lfu iacma iacma_with_memory quiver coordl fgfglru fgfglfu
# for cache_algorithm in fgfglfu
# do
#     # for num in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
#     for num in 4000
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py MSRVTT $num $cache_algorithm
#         while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#             sleep 1  # 每秒检查一次
#         done
#         echo "cgroup 内所有进程已结束"
#     done
# done

# FFSlim Clotho_E
# for cache_algorithm in lru lfu iacma iacma_with_memory coordl fgfglfu fgfglru quiver
# do
#     for num in 1969 3938 5907 7876 9845 11814 13783 15752 17721 19690
#     do
#         echo "当前算法: $cache_algorithm"
#         echo "当前缓存大小: $num"
#         sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader.py Clotho_E $num $cache_algorithm
#         while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#             sleep 1  # 每秒检查一次
#         done
#         echo "cgroup 内所有进程已结束"
#     done
# done
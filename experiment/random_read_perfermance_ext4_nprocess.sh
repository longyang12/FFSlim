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
sudo blockdev --getra /dev/nvme0n1
sudo blockdev --setra 0 /dev/nvme0n1
sudo blockdev --getra /dev/nvme0n1

# FFSlim
# for num in 2 4 6 8
# do
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader_nproc.py flickr30k 12406 iacma_with_memory $num 0

#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"
# done


# for num in 2 4 6 8
# do
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader_nproc.py flickr30k 12406 fgfglru $num 0

#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"
# done

# for num in 2 4 6 8
# do
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader_nproc.py flickr30k 12406 iacma_with_memory $num 1

#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"
# done

# for num in 2 4 6 8
# do
#     sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_FFSlim_loader_nproc.py flickr30k 12406 fgfglru $num 1

#     while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
#         sleep 1  # 每秒检查一次
#     done
#     echo "cgroup 内所有进程已结束"
# done

#TDP
# test
# sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader_nproc.py flickr30k 12406 iacma_with_memory 2 0

for num in 2 4 6 8
do
    sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader_nproc.py flickr30k 12406 iacma_with_memory $num 0

    while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
        sleep 1  # 每秒检查一次
    done
    echo "cgroup 内所有进程已结束"
done


for num in 2 4 6 8
do
    sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader_nproc.py flickr30k 12406 lru $num 0

    while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
        sleep 1  # 每秒检查一次
    done
    echo "cgroup 内所有进程已结束"
done

for num in 2 4 6 8
do
    sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader_nproc.py flickr30k 12406 iacma_with_memory $num 1

    while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
        sleep 1  # 每秒检查一次
    done
    echo "cgroup 内所有进程已结束"
done

for num in 2 4 6 8
do
    sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/random_TDP_loader_nproc.py flickr30k 12406 lru $num 1

    while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
        sleep 1  # 每秒检查一次
    done
    echo "cgroup 内所有进程已结束"
done
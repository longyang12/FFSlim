# Files
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_files_loader.py flickr30k 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_files_loader.py flickr30k_entities 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_files_loader.py mscoco 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_files_loader.py vqa2 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_files_loader.py gqa 0 lru

# TDP
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_TDP_loader.py flickr30k 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_TDP_loader.py flickr30k_entities 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_TDP_loader.py mscoco 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_TDP_loader.py vqa2 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_TDP_loader.py gqa 0 lru

# FFRecord
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_ffrecord_loader.py flickr30k 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_ffrecord_loader.py flickr30k_entities 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_ffrecord_loader.py mscoco 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_ffrecord_loader.py vqa2 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_ffrecord_loader.py gqa 0 lru

# FFSlim
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_FFSlim_loader.py flickr30k 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_FFSlim_loader.py flickr30k_entities 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_FFSlim_loader.py mscoco 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_FFSlim_loader.py vqa2 0 lru
sudo ./run_with_cgroup.sh python3 /home/llm/experiment/dataset_Analysis_Modeling/loaders/seq_FFSlim_loader.py gqa 0 lru
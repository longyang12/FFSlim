
# Clotho_E
dataset=Clotho_E
# python /home/llm/experiment/dataset_Analysis_Modeling/generaters/generate_Files.py $dataset
python /home/llm/experiment/dataset_Analysis_Modeling/generaters/generate_TDP.py $dataset
# python /home/llm/experiment/dataset_Analysis_Modeling/generaters/generate_FFRecord.py $dataset
# python /home/llm/experiment/dataset_Analysis_Modeling/generaters/generate_FFSlim.py $dataset

# tar -cf /mnt/datasets/Clotho_E.tar /mnt/datasets/Clotho_E*

# scp /mnt/datasets/Clotho_E.tar lenovo@w0:/mnt/datasets/datasets
import multiprocessing
import time
import os
def process_chunk(chunk, pid):
    result = [x * 2 for x in chunk]  # 处理逻辑
    time.sleep(2)
    print(f"Process {pid} processed: {result}")


if __name__ == "__main__":
    # 创建 CGroup
    os.system("sudo cgcreate -g memory:file_reader1")

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_processes = 4
    chunk_size = len(data) // (num_processes-1)
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=process_chunk, args=(chunk, i))
        processes.append(p)
        p.start()

        with open("/sys/fs/cgroup/file_reader1/cgroup.procs", "a+") as f:
            f.write(str(p.pid))

        os.system("echo ===========================================")
        os.system("cat /sys/fs/cgroup/file_reader1/cgroup.procs")
        os.system("echo ===========================================")

    for p in processes:
        p.join()

    os.system("sudo cgdelete -g memory:file_reader1")
    
#!/bin/bash

# 配置部分
CGROUP_PATH="/sys/fs/cgroup/file_reader"
DISK="nvme2n1"
LOG_FILE="./monitor_log/monitor_log_$(date +%Y%m%d_%H%M%S).csv"

# 检查路径
if [ ! -d "$CGROUP_PATH" ]; then echo "Error: cgroup path not found"; exit 1; fi

# 打印表头：增加 Read(MB/s), Write(MB/s), IOPS(r+w/s)
HEADER="Time,P_CPU(%),P_RSS(MB),P_VIRT(MB),S_Used(MB),S_Cache(MB),Total_Load(MB),Read(MB/s),Write(MB/s),IOPS,SSD_Util(%)"
echo $HEADER | tee $LOG_FILE

while [ $(wc -c < "/sys/fs/cgroup/file_reader/cgroup.procs") -gt 0 ]; do
    # 获取 cgroup 内所有有效 PIDs
    PIDS=$(tr '\n' ',' < "$CGROUP_PATH/cgroup.procs" | sed 's/,$//')
    
    # 1. 进程组级 CPU 和 内存聚合
    P_STATS=$(ps -p ${PIDS//,/ -p } -o %cpu,rss,vsize --no-headers 2>/dev/null | awk '{cpu+=$1; rss+=$2; virt+=$3} END {print cpu, rss/1024, virt/1024}')
    read P_CPU P_RSS P_VIRT <<< "$P_STATS"

    # 2. 系统级内存 (兼容中英文)
    S_MEM_RAW=$(free -m)
    S_USED=$(echo "$S_MEM_RAW" | awk '/Mem:|内存：/{print $3}')
    S_CACHE=$(echo "$S_MEM_RAW" | awk '/Mem:|内存：/{print $6}')

    # 3. 计算 P_RSS + S_Cache (总装载量)
    TOTAL_LOAD=$(echo "scale=2; ${P_RSS:-0} + ${S_CACHE:-0}" | bc)

    # 4. SSD 物理详细数据 (iostat -dx 采样)
    # 提取第14行之后的输出，取最后一次采样的平均值
    IO_RAW=$(iostat -dx $DISK 1 2 | grep "$DISK" | tail -n 1)
    
    # 根据 iostat 版本不同，列位置可能略有差异，通常如下：
    # r/s (读取IOPS): $2, w/s (写入IOPS): $8
    # rKB/s (读取带宽): $3, wKB/s (写入带宽): $9
    # %util (利用率): $NF (最后一列)
    R_IOPS=$(echo $IO_RAW | awk '{print $2}')
    W_IOPS=$(echo $IO_RAW | awk '{print $8}')
    TOTAL_IOPS=$(echo "scale=1; $R_IOPS + $W_IOPS" | bc)
    
    R_MB=$(echo $IO_RAW | awk '{print $3/1024}')
    W_MB=$(echo $IO_RAW | awk '{print $9/1024}')
    SSD_UTIL=$(echo $IO_RAW | awk '{print $NF}')

    # 5. 整合输出
    TIMESTAMP=$(date +"%H:%M:%S")
    printf "%s, %.1f, %.2f, %.2f, %d, %d, %.2f, %.2f, %.2f, %.1f, %.1f\n" \
        "$TIMESTAMP" "${P_CPU:-0}" "${P_RSS:-0}" "${P_VIRT:-0}" "${S_USED:-0}" "${S_CACHE:-0}" \
        "$TOTAL_LOAD" "${R_MB:-0}" "${W_MB:-0}" "${TOTAL_IOPS:-0}" "${SSD_UTIL:-0}" \
        | tee -a $LOG_FILE

    # 注意：iostat 采样已消耗 1 秒
done

echo "监控结束。"
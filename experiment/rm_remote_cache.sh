#ge Cache清理脚本
# 使用示例：./clear_remote_cache.sh <用户名> <服务器IP> <SSH端口(可选)>

# 检查参数输入
if [ $# -lt 2 ]; then
    echo -e "\033[31m错误：参数不足！\033[0m"
    echo "用法：$0 <用户名> <服务器IP> [SSH端口]"
    echo "示例：$0 admin 192.168.1.100"
    exit 1
fi

USER="$1"
HOST="$2"
PORT="${3:-22}"  # 默认SSH端口22
TIMEOUT=3000      # 超时时间（秒）

ssh $USER@$HOST "free -h"

# 清除缓存的核心函数
clear_remote_cache() {
    echo -e "\033[34m➤ 正在连接 $USER@$HOST:$PORT ...\033[0m"
    
    # 执行远程命令（关键步骤）
    ssh -p $PORT -o ConnectTimeout=10 $USER@$HOST \
        "sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'" 2>&1
    
    return $?
}

# 主执行流程
start_time=$(date +%s)
clear_remote_cache &
pid=$!

# 等待动画
echo -ne "\033[33m⏳ 正在清理Page Cache "
while kill -0 $pid 2>/dev/null; do
    echo -n "."
    sleep 1
    # 超时检查
    if [ $(($(date +%s) - start_time)) -gt $TIMEOUT ]; then
        echo -e "\n\033[31m✖ 错误：操作超时（${TIMEOUT}秒）！\033[0m"
        kill -9 $pid 2>/dev/null
        exit 2
    fi
done
wait $pid
status=$?

# 结果反馈
if [ $status -eq 0 ]; then
    echo -e "\n\033[32m✔ Page Cache清理完成！耗时: $(( $(date +%s) - start_time ))秒\033[0m"
else
    echo -e "\n\033[31m✖ 清理失败！错误码: $status\033[0m"
    exit $status
fi

ssh $USER@$HOST "free -h"

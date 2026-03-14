#!/bin/bash

# 监控cgroup内存统计信息
for i in {1..200}
do
echo "***************************************************"
cat /sys/fs/cgroup/file_reader/cgroup.procs
cat /sys/fs/cgroup/file_reader/memory.current
awk '$1 == "anon" || $1 == "kernel" || $1 == "slab" || $1 == "pagetables" {sum += $2} END {print sum}' /sys/fs/cgroup/file_reader/memory.stat
cat /sys/fs/cgroup/file_reader/memory.stat | grep file
cat /sys/fs/cgroup/file_reader/memory.stat | grep anon
cat /sys/fs/cgroup/file_reader/memory.stat | grep slab
cat /sys/fs/cgroup/file_reader/memory.stat | grep pgscan
cat /sys/fs/cgroup/file_reader/memory.stat | grep pgsteal
cat /sys/fs/cgroup/file_reader/memory.stat | grep pgfault
# cat /sys/fs/cgroup/file_reader/memory.stat
echo "***************************************************" 
sleep 1
done
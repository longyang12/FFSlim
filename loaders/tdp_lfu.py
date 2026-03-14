from collections import defaultdict, OrderedDict


class LFUNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.freq = 1

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.node_map = {}  # 哈希表细粒度存储 key:int,value:LFUNode
        self.freq_map = defaultdict(OrderedDict)  # {freq: OrderedDict{key: node}}

    def _update_freq(self, node):
        # 移除旧频率层中的节点
        freq = node.freq
        del self.freq_map[freq][node.key]
        if not self.freq_map[freq]:
            del self.freq_map[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        
        # 更新频率并加入新层
        node.freq += 1
        new_freq = node.freq
        self.freq_map[new_freq][node.key] = node

    def get(self, key: str) -> bytes:
        # 缓存索引
        if key not in self.node_map:
            return -1
        node = self.node_map[key]
        self._update_freq(node)
        return node.value
    
    def put(self, key: str, value: bytes) -> None:
        # 缓存空间不足
        if len(self.node_map) >= self.capacity:
            # 删除缓存
            oldest_key,_ = self.freq_map[self.min_freq].popitem(last=False)
            del self.node_map[oldest_key]
        # 缓存存储
        new_node = LFUNode(key, value)
        self.node_map[key] = new_node
        self.freq_map[1][key] = new_node
        self.min_freq = 1  # 新节点频率初始为1
from collections import defaultdict, OrderedDict


class LFUNode:
    def __init__(self, key=0, value={"image":0,"captions":[]}):
        self.key = key
        self.value = value
        self.remain = len(value["captions"])
        self.freq = 1

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.index = {}  # 哈希表细粒度索引 key:int,value:(img,caption,obj_index)
        self.node_map = {}  # 哈希表粗粒度存储 key:int,value:{image:bytes,captions:[]}
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

    def get(self, key: int) -> tuple:
        # 细粒度缓存索引
        if key not in self.index:
            return -1
        result = self.index[key]
        # 粗粒度缓存管理
        obj_index = result[2]
        node = self.node_map[obj_index]
        self._update_freq(node)
        return result[:-1]
    
    def put(self, key: int, value: dict) -> None:
        # 缓存空间不足
        if len(self.node_map) >= self.capacity:
            # 删除粗粒度缓存
            oldest_key,lfu_node = self.freq_map[self.min_freq].popitem(last=False)
            del self.node_map[oldest_key]
            # 删除细粒度索引
            for i in range(len(lfu_node.value["captions"])):
                del self.index[lfu_node.key + i]
        # 缓存粗粒度存储
        new_node = LFUNode(key, value)
        self.node_map[key] = new_node
        self.freq_map[1][key] = new_node
        self.min_freq = 1  # 新节点频率初始为1
        # 缓存细粒度索引
        for index,c in enumerate(value["captions"]):
            self.index[key+index] = (value["image"],c,key)
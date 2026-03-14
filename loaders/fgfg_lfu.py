from collections import defaultdict, OrderedDict


class FGFGLFUNode:
    def __init__(self, key=0, value={"image":0,"caption":0}):
        self.key = key
        self.value = value
        self.freq = 1

class FGFGLFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.node_map = {}  # 哈希表细粒度存储 key:int,value:{image:bytes,caption:str}
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
        if key not in self.node_map:
            return -1
        node = self.node_map[key]
        self._update_freq(node)
        return node.value
    
    def put(self, key: int, value: dict) -> None:
        for index,c in enumerate(value["captions"]):
            # 跳过重复缓存
            if key+index in self.node_map:
                continue
            # 缓存空间不足
            if len(self.node_map) >= self.capacity:
                # 删除细粒度缓存
                oldest_key,_ = self.freq_map[self.min_freq].popitem(last=False)
                del self.node_map[oldest_key]
            # 缓存细粒度存储
            temp = {"image":value["image"] + index.to_bytes(length=2),"caption":c}
            new_node = FGFGLFUNode(key+index, temp)
            self.node_map[key+index] = new_node
            self.freq_map[1][key+index] = new_node
            self.min_freq = 1  # 新节点频率初始为1
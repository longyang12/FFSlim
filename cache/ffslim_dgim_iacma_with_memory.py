from collections import defaultdict, OrderedDict


class IACMANode:
    def __init__(self, key=0, value={"image":0,"captions":[]}):
        self.key = key
        self.value = value
        self.score = len(value["captions"]) - 1

class DGIMIACMACacheWithMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_score = 10000
        self.index = {}  # 哈希表细粒度索引 key:int,value:(img,caption,obj_index)
        self.node_map = {}  # 哈希表粗粒度存储 key:int,value:{image:bytes,captions:[]}
        self.score_map = defaultdict(OrderedDict)  # {freq: OrderedDict{key: node}}
        self.removed_score = {}  # key:removed_score

    def _update_score(self, node):
        # 移除旧频率层中的节点
        score = node.score
        del self.score_map[score][node.key]
        if not self.score_map[score]:
            del self.score_map[score]
        
        # 更新频率并加入新层
        node.score -= 1
        new_score = node.score
        self.score_map[new_score][node.key] = node
        if self.min_score > new_score:
            self.min_score = new_score

    def get(self, key: int) -> tuple:
        # 细粒度缓存索引
        if key not in self.index:
            return -1
        result = self.index[key]
        # 粗粒度缓存管理
        obj_index = result[2]
        node = self.node_map[obj_index]
        self._update_score(node)
        return result[:-1]
    
    def put(self, key: int, value: dict) -> None:
        # 当缓存空间充足时，不允许无价值对象进入
        score = len(value["captions"]) - 1
        if score == 0:
            return
        
        # 缓存空间不足
        if len(self.node_map) >= self.capacity:
            # 删除粗粒度缓存
            # 当缓存空间不足时,不允许低价值对象进入
            while len(self.score_map[self.min_score]) == 0:
                self.min_score += 1
            score = len(value["captions"]) - 1
            # 获取记忆中的score
            if key in self.removed_score:
                score = self.removed_score[key] - 1
            if score <= self.min_score:
                return
            oldest_key, iacma_node = self.score_map[self.min_score].popitem(last=False)
            del self.node_map[oldest_key]
            # 暂存删除记忆
            if iacma_node.score != 0:
                self.removed_score[oldest_key] = iacma_node.score
            else:
                if oldest_key in self.removed_score:
                    del self.removed_score[oldest_key]
            # 删除细粒度索引
            for i in range(len(iacma_node.value["captions"])):
                del self.index[iacma_node.key + i]
            
        # 缓存粗粒度存储
        new_node = IACMANode(key, value)
        # 获取记忆中的score
        if key in self.removed_score:
            new_node.score = self.removed_score[key] - 1
        self.node_map[key] = new_node
        self.score_map[new_node.score][key] = new_node
        if self.min_score > new_node.score:
            self.min_score = new_node.score
        
        # 缓存细粒度索引
        for index,c in enumerate(value["captions"]):
            self.index[key+index] = (value["image"],c,key)
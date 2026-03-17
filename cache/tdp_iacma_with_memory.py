from collections import defaultdict, OrderedDict


class IACMANode:
    def __init__(self, key=0, value=0, score=0):
        self.key = key
        self.value = value
        self.score = score

class IACMACacheWithMemory:
    def __init__(self, capacity: int, images: list):
        self.capacity = capacity
        self.min_score = 10000
        self.cache = {}  # 哈希表粗粒度存储 key:str,value:IACMANode
        self.node_score = {} # key:str,value:score
        self.score_map = defaultdict(OrderedDict)  # {freq: OrderedDict{key: node}}
        for i in images:
            key = i.split(".")[0]
            if key in self.node_score:
                self.node_score[key] += 1
            else:
                self.node_score[key] = 1
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

    def get(self, key: str) -> bytes:
        # 缓存索引
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 缓存管理
        self._update_score(node)
        # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #     f.write(f"==get==key:{key},len={len(node.value["captions"])},score={node.score},min_score={self.min_score}\n")
        return node.value
    
    def warm_up_put(self, key: str, value: bytes) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            return
        # 缓存存储
        new_node = IACMANode(key, value, self.node_score[key])
        self.cache[key] = new_node
        self.score_map[new_node.score][key] = new_node
        if self.min_score > new_node.score:
            self.min_score = new_node.score

    def put(self, key: str, value: bytes) -> None:
        # 当缓存空间充足时，不允许无价值对象进入
        score = self.node_score[key] - 1
        if score == 0:
            return
        
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            # 删除缓存
            # 当缓存空间不足时,不允许低价值对象进入
            while len(self.score_map[self.min_score]) == 0:
                self.min_score += 1
            score = self.node_score[key] - 1
            if key in self.removed_score:
                score = self.removed_score[key] - 1
            if score  <= self.min_score:
                return
            oldest_key, iacma_node = self.score_map[self.min_score].popitem(last=False)
            del self.cache[oldest_key]
            # 暂存删除记忆
            if iacma_node.score != 0:
                self.removed_score[oldest_key] = iacma_node.score
            else:
                if oldest_key in self.removed_score:
                    del self.removed_score[oldest_key]
            
        # 缓存存储
        new_node = IACMANode(key, value, self.node_score[key] - 1)
        # 获取记忆中的score
        if key in self.removed_score:
            new_node.score = self.removed_score[key] - 1
        self.cache[key] = new_node
        self.score_map[new_node.score][key] = new_node
        if self.min_score > new_node.score:
            self.min_score = new_node.score
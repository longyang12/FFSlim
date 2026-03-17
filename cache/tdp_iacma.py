from collections import defaultdict, OrderedDict


class IACMANode:
    def __init__(self, key=0, value=0, score=0):
        self.key = key
        self.value = value
        self.score = score

class IACMACache:
    def __init__(self, capacity: int, images: list):
        self.capacity = capacity
        self.min_score = 10000
        self.node_map = {}  # 哈希表粗粒度存储 key:str,value:IACMANode
        self.node_score = {} # key:str,value:score
        self.score_map = defaultdict(OrderedDict)  # {freq: OrderedDict{key: node}}
        for i in images:
            key = i.split(".")[0]
            if key in self.node_score:
                self.node_score[key] += 1
            else:
                self.node_score[key] = 1

    def _update_freq(self, node):
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
        if key not in self.node_map:
            return -1
        node = self.node_map[key]
        # 缓存管理
        self._update_freq(node)
        # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #     f.write(f"==get==key:{key},len={len(node.value["captions"])},score={node.score},min_score={self.min_score}\n")
        return node.value
    
    def put(self, key: str, value: bytes) -> None:
        # 当缓存空间充足时，不允许无价值对象进入
        score = self.node_score[key] - 1
        if score == 0:
            return

        # 缓存空间不足
        if len(self.node_map) >= self.capacity:
            # 删除缓存
            # 当缓存空间不足时,不允许低价值对象进入
            while len(self.score_map[self.min_score]) == 0:
                self.min_score += 1
            if self.node_score[key] - 1 <= self.min_score:
                return
            oldest_key, _ = self.score_map[self.min_score].popitem(last=False)
            # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
            #     f.write(f"==del==key:{iacma_node.key},len={len(iacma_node.value["captions"])},score={iacma_node.score},min_score={self.min_score}\n")
            del self.node_map[oldest_key]
            
        # 缓存存储
        new_node = IACMANode(key, value, self.node_score[key] - 1)
        self.node_map[key] = new_node
        self.score_map[new_node.score][key] = new_node
        if self.min_score > new_node.score:
            self.min_score = new_node.score
        
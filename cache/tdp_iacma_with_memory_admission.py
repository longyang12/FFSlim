from collections import defaultdict, OrderedDict


class IACMANode:
    def __init__(self, key=0, value=0, score=0):
        self.key = key
        self.value = value
        self.score = score

class IACMACacheWithMemoryAdmission:
    def __init__(self, capacity: int, images: list):
        self.capacity = capacity
        self.min_score = 10000
        self.cache = {}  # 哈希表粗粒度存储 key:str,value:IACMANode
        self.node_score = {} # key:str,value:score
        self.score_map = defaultdict(OrderedDict)  # {freq: OrderedDict{key: node}}
        self.node_score_shift = {} # # key: obj_index, value:[(id,score_shift1), (id,score_shift2),...]

        
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

    def get(self, key: str, id: int) -> bytes:
        # 缓存索引
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 缓存管理
        self._update_score(node)

        # 记录对象重要性
        self.node_score_shift[key].append((id, node.score))


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

    def put(self, key: str, value: bytes, id: int) -> None:
        # 当缓存空间充足时，不允许无价值对象进入
        score = self.node_score[key] - 1
        if score == 0:
            return
        
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            while len(self.score_map[self.min_score]) == 0:
                self.min_score += 1

            # 当缓存空间不足时,不允许低价值对象进入
            score = self.node_score[key] - 1
            if key in self.removed_score:
                score = self.removed_score[key] - 1
            if score  <= self.min_score:
                if key in self.removed_score: # 即使没进入缓存，重要性依旧要更新
                    self.removed_score[key] = self.removed_score[key] - 1 
                    if self.removed_score[key] <= 0:
                        del self.removed_score[key]
                return
            
            # 删除缓存
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
                        
        # 记录对象重要性
        if key not in self.node_score_shift:
            self.node_score_shift[key] = []
        self.node_score_shift[key].append((id, new_node.score))

    # 输出对象重要性字典
    def get_object_importance(self):
        importance_dict = {}
        for key, score_shifts in self.node_score_shift.items():
            importance_dict[key] = score_shifts
        return importance_dict
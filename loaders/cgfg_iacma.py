from collections import defaultdict, OrderedDict


class IACMANode:
    def __init__(self, key=0, value={"image":0,"captions":[]}):
        self.key = key
        self.value = value
        self.score = len(value["captions"]) - 1

class IACMACache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_score = 10000
        self.index = {}  # 哈希表细粒度索引 key:int,value:(img,caption,obj_index)
        self.node_map = {}  # 哈希表粗粒度存储 key:int,value:{image:bytes,captions:[]}
        self.score_map = defaultdict(OrderedDict)  # {freq: OrderedDict{key: node}}

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

    def get(self, key: int) -> tuple:
        # 细粒度缓存索引
        if key not in self.index:
            return -1
        result = self.index[key]
        # 粗粒度缓存管理
        obj_index = result[2]
        node = self.node_map[obj_index]
        self._update_freq(node)
        # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #     f.write(f"==get==key:{key},len={len(node.value["captions"])},score={node.score},min_score={self.min_score}\n")
        return result[:-1]
    
    def put(self, key: int, value: dict) -> None:
        # 当缓存空间充足时，不允许无价值对象进入
        score = len(value["captions"]) - 1
        if score == 0:
            return
        
        # 缓存空间不足
        if len(self.node_map) >= self.capacity:
            # 删除粗粒度缓存
            # if self.min_score in self.score_map:
                # print(f"show min_score:{self.min_score}")
            # 当缓存空间不足时,不允许低价值对象进入
            while len(self.score_map[self.min_score]) == 0:
                self.min_score += 1
            score = len(value["captions"]) - 1
            if score <= self.min_score:
                return
                # print(f"add min_score:{self.min_score}")
            # print(f"==put==show score_map:{self.score_map.keys()}")
            # for k,v in self.score_map.items():
            #     print(f"key={k},value_len={len(v)}")
            oldest_key, iacma_node = self.score_map[self.min_score].popitem(last=False)
            # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
            #     f.write(f"==del==key:{iacma_node.key},len={len(iacma_node.value["captions"])},score={iacma_node.score},min_score={self.min_score}\n")
            del self.node_map[oldest_key]
            # 删除细粒度索引
            for i in range(len(iacma_node.value["captions"])):
                del self.index[iacma_node.key + i]
            
        # 缓存粗粒度存储
        new_node = IACMANode(key, value)
        self.node_map[key] = new_node
        self.score_map[new_node.score][key] = new_node
        if self.min_score > new_node.score:
            self.min_score = new_node.score
        
        # 缓存细粒度索引
        # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #     f.write(f"==put==key:{key},len={len(value["captions"])},min_score={self.min_score}\n")
        for index,c in enumerate(value["captions"]):
            self.index[key+index] = (value["image"],c,key)
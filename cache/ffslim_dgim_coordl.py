class DLinkedNode:
    def __init__(self, key=0, value={"image":0,"caption":0}):
        self.key = key
        self.value = value

class DGIMCoorDL:
    def __init__(self, capacity: int):
        self.index = {}  # 哈希表细粒度索引 key:int,value:(img,caption,obj_index)
        self.cache = {}  # 哈希表粗粒度存储 key:int,value:{image:bytes,captions:[]}
        self.capacity = capacity

    def get(self, key: int) -> tuple:
        # 细粒度缓存索引
        if key not in self.index:
            return -1
        result = self.index[key]
        return result[:-1]

    def put(self, key: int, value: dict) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            return 
        # 缓存粗粒度存储
        node = DLinkedNode(key, value)
        self.cache[key] = node
        # 缓存细粒度索引
        # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #     f.write(f"==put==key:{key},len={len(value["captions"])}\n")
        for index,c in enumerate(value["captions"]):
            self.index[key+index] = (value["image"],c,key)


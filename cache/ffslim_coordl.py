class DLinkedNode:
    def __init__(self, key=0, value={"image":0,"caption":0}):
        self.key = key
        self.value = value

class CoorDL:
    def __init__(self, capacity: int):
        self.cache = {}  # 哈希表细粒度存储 key:int,value:{image:bytes,captions:str}
        self.capacity = capacity

    def get(self, key: int) -> tuple:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        return node.value

    def put(self, key: int, value: dict) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            return 
        # 缓存细粒度存储
        for index,c in enumerate(value["captions"]):
            # 跳过重复缓存
            if key+index in self.cache:
                continue
            # 缓存空间不足
            if len(self.cache) >= self.capacity:
                return
            # 细粒度缓存存储
            temp = {"image":value["image"]+index.to_bytes(length=2),"caption":c}
            node = DLinkedNode(key+index, temp)
            self.cache[key+index] = node


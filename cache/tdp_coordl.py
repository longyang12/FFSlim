class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value

class CoorDL:
    def __init__(self, capacity: int):
        self.cache = {}  # 哈希表细粒度存储 key:int,value:DLinkedNode
        self.capacity = capacity

    def get(self, key: str) -> bytes:
        # 细粒度缓存索引
        if key not in self.cache:
            return -1
        node = self.cache[key]
        return node.value

    def warm_up_put(self, key: str, value: bytes) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            return
        # 缓存存储
        node = DLinkedNode(key, value)
        self.cache[key] = node

    def put(self, key: str, value: bytes) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            return
        # 缓存存储
        node = DLinkedNode(key, value)
        self.cache[key] = node


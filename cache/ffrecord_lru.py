class DLinkedNode:
    def __init__(self, key=0, value={"image":0,"caption":0}):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}  # 哈希表细粒度存储 key:int,value:{image:bytes,captions:str}
        self.capacity = capacity
        self.head = DLinkedNode()  # 虚拟头节点
        self.tail = DLinkedNode()  # 虚拟尾节点
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node: DLinkedNode) -> None:
        """将节点插入链表尾部（最近使用）"""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def _remove_node(self, node: DLinkedNode) -> None:
        """移除指定节点"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def get(self, key: int) -> tuple:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove_node(node)
        self._add_node(node)
        return node.value

    def put(self, key: int, value: dict) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            lru_node = self.head.next  # 淘汰链表头部的节点
            self._remove_node(lru_node)
            del self.cache[lru_node.key]
        node = DLinkedNode(key, value)
        self.cache[key] = node
        self._add_node(node)


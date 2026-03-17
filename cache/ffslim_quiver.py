class DLinkedNode:
    def __init__(self, key=0, value={"image": b"", "caption": ""}):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class Quiver:
    def __init__(self, capacity: int):
        self.cache = {}  # 哈希表存储 key -> 节点
        self.capacity = capacity
        # 虚拟头尾节点（简化链表操作）
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node: DLinkedNode) -> None:
        """将新节点插入链表尾部（最新插入）"""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def _remove_node(self, node: DLinkedNode) -> None:
        """移除指定节点（淘汰时调用）"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def get(self, key: int) -> dict:
        if key not in self.cache:
            return -1
        return self.cache[key].value

    def put(self, key: int, value: dict) -> None:
        # 缓存细粒度存储
        for index,c in enumerate(value["captions"]):
            # 跳过重复缓存
            if key+index in self.cache:
                continue
            # 缓存空间不足
            if len(self.cache) >= self.capacity:
                fifo_node = self.head.next
                self._remove_node(fifo_node)
                del self.cache[fifo_node.key]
            # 细粒度缓存存储
            temp = {"image":value["image"]+index.to_bytes(length=2),"caption":c}
            node = DLinkedNode(key+index, temp)
            self.cache[key+index] = node
            self._add_node(node)
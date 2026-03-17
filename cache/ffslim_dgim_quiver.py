class DLinkedNode:
    def __init__(self, key=0, value={"image": b"", "caption": ""}):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class DGIMQuiver:
    def __init__(self, capacity: int):
        self.index = {}  # 哈希表细粒度索引 key:int,value:(img,caption,obj_index)
        self.cache = {}  # 哈希表粗粒度存储 key:int,value:{image:bytes,captions:[]}
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

    def get(self, key: int) -> tuple:
        # 细粒度缓存索引
        if key not in self.index:
            return -1
        result = self.index[key]
        return result[:-1]

    def put(self, key: int, value: dict) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            fifo_node = self.head.next  # 淘汰链表头部的节点
            # 删除细粒度索引
            # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
            #     f.write(f"==del==key:{fifo_node.key},len={len(fifo_node.value["captions"])}\n")
            for i in range(len(fifo_node.value["captions"])):
                # if fifo_node.key + i in self.index:
                #     with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
                #         f.write(f"==in==key:{fifo_node.key},index={i}\n")
                del self.index[fifo_node.key + i]
                # else:
                #     with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
                #         f.write(f"==not in index==key:{fifo_node.key},index={index}\n")
            # 删除粗粒度缓存
            self._remove_node(fifo_node)
            del self.cache[fifo_node.key]

        # 缓存粗粒度存储
        node = DLinkedNode(key, value)
        self.cache[key] = node
        self._add_node(node)
        # 缓存细粒度索引
        # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #     f.write(f"==put==key:{key},len={len(value["captions"])}\n")
        for index,c in enumerate(value["captions"]):
            self.index[key+index] = (value["image"],c,key)
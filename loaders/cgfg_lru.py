class DLinkedNode:
    def __init__(self, key=0, value={"image":0,"captions":[]}):
        self.key = key
        self.value = value
        self.remain = len(value["captions"])
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.index = {}  # 哈希表细粒度索引 key:int,value:(img,caption,obj_index)
        self.cache = {}  # 哈希表粗粒度存储 key:int,value:{image:bytes,captions:[]}
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
        # 细粒度缓存索引
        if key not in self.index:
            return -1
        result = self.index[key]
        # 粗粒度缓存管理
        obj_index = result[2]
        # if obj_index in self.cache:
        node = self.cache[obj_index]
        self._remove_node(node)
        self._add_node(node)  # 移动到链表尾部
        # else:
        #     with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #         f.write(f"==not in cache==key:{key},obj_index={obj_index}\n") 
        return result[:-1]

    def put(self, key: int, value: dict) -> None:
        # 缓存空间不足
        if len(self.cache) >= self.capacity:
            lru_node = self.head.next  # 淘汰链表头部的节点
            # 删除细粒度索引
            # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
            #     f.write(f"==del==key:{lru_node.key},len={len(lru_node.value["captions"])}\n")
            for i in range(len(lru_node.value["captions"])):
                # if lru_node.key + i in self.index:
                #     with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
                #         f.write(f"==in==key:{lru_node.key},index={i}\n")
                del self.index[lru_node.key + i]
                # else:
                #     with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
                #         f.write(f"==not in index==key:{lru_node.key},index={index}\n")
            # 删除粗粒度缓存
            self._remove_node(lru_node)
            del self.cache[lru_node.key]

        # 缓存粗粒度存储
        node = DLinkedNode(key, value)
        self.cache[key] = node
        self._add_node(node)
        # 缓存细粒度索引
        # with open("/home/llm/experiment/dataset_Analysis_Modeling/log.txt","a+") as f:
        #     f.write(f"==put==key:{key},len={len(value["captions"])}\n")
        for index,c in enumerate(value["captions"]):
            self.index[key+index] = (value["image"],c,key)


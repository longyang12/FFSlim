from collections import defaultdict, OrderedDict


class FGFGLFUNode:
    def __init__(self, key=0, value={"image":0,"caption":0}):
        self.key = key
        self.value = value
        self.freq = 1
        self.prev = None        # 前驱节点
        self.next = None        # 后继节点

class DoublyLinkedList:
    """双向链表类，提供头删和尾插接口"""
    def __init__(self):
        # 创建虚拟头尾节点，简化边界条件处理
        self.head = FGFGLFUNode()
        self.tail = FGFGLFUNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def add_to_tail(self, node):
        """尾插法：在链表尾部添加节点"""
        # 将新节点插入到tail的前面
        last = self.tail.prev
        
        last.next = node
        node.prev = last
        
        node.next = self.tail
        self.tail.prev = node
        
        self.size += 1
    
    def remove_from_head(self):
        """头删法：删除链表头部第一个真实节点"""
        if self.size == 0:
            return None
            
        to_remove = self.head.next
        self._remove_node(to_remove)
        return to_remove
    
    def _remove_node(self, node):
        """从链表中移除指定节点"""
        if self.size == 0:
            return
            
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
        
        node.prev = None
        node.next = None
        self.size -= 1
    
    def is_empty(self):
        """判断链表是否为空"""
        return self.size == 0
    
    def __len__(self):
        return self.size


class FGFGLFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.node_map = {}  # 哈希表细粒度存储 key:int,value:{image:bytes,caption:str}
        self.freq_map = defaultdict(DoublyLinkedList)  # {freq: OrderedDict{key: node}}

    def _update_freq(self, node):
        # 移除旧频率层中的节点
        freq = node.freq
        self.freq_map[freq]._remove_node(node)
        if not self.freq_map[freq]:
            del self.freq_map[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        
        # 更新频率并加入新层
        node.freq += 1
        new_freq = node.freq
        self.freq_map[new_freq].add_to_tail(node)

    def get(self, key: int) -> tuple:
        # 细粒度缓存索引
        if key not in self.node_map:
            return -1
        node = self.node_map[key]
        self._update_freq(node)
        return node.value
    
    def put(self, key: int, value: dict) -> None:
        for index,c in enumerate(value["captions"]):
            # 跳过重复缓存
            if key+index in self.node_map:
                continue
            # 缓存空间不足
            if len(self.node_map) >= self.capacity:
                # 删除细粒度缓存
                to_remove = self.freq_map[self.min_freq].remove_from_head()
                del self.node_map[to_remove.key]

            # 缓存细粒度存储
            temp = {"image":value["image"] + index.to_bytes(length=2),"caption":c}
            new_node = FGFGLFUNode(key+index, temp)
            self.node_map[key+index] = new_node
            self.freq_map[1].add_to_tail(new_node)
            self.min_freq = 1  # 新节点频率初始为1
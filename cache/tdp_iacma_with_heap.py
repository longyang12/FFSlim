

class IACMANode:
    def __init__(self, key=0, value=0, score=0):
        self.key = key
        self.value = value
        self.score = score

class MinHeap:
    def __init__(self):
        self.heap = []
        self.idx_map = {}  # 跟踪元素索引（用于更新操作）

    def insert(self, item, score):
        self.heap.append((score, item))
        i = len(self.heap) - 1
        self.idx_map[item] = i
        self._sift_up(i)

    def update(self, item, new_score):
        if item not in self.idx_map:
            return
        i = self.idx_map[item]
        old_score = self.heap[i][0]
        self.heap[i] = (new_score, item)
        # 根据新值调整位置
        if new_score < old_score:
            self._sift_up(i)
        elif new_score > old_score:
            self._sift_down(i)

    def extract_min(self):
        if not self.heap:
            return None
        self._swap(0, len(self.heap)-1)
        min_val = self.heap.pop()
        del self.idx_map[min_val[1]]
        self._sift_down(0)
        return min_val

    def _sift_up(self, i):
        while i > 0:
            parent = (i-1)//2
            if self.heap[i][0] < self.heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while i < n:
            left = 2*i + 1
            right = 2*i + 2
            smallest = i
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def _swap(self, i, j):
        # 更新索引映射
        self.idx_map[self.heap[i][1]] = j
        self.idx_map[self.heap[j][1]] = i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

class IACMACacheWithHeap:
    def __init__(self, capacity: int, images: list):
        self.capacity = capacity
        self.node_map = {}  # 哈希表粗粒度存储 key:str,value:IACMANode
        self.node_score = {} # key:str,value:score
        self.score_heap = MinHeap()  # [(score,item),]
        for i in images:
            key = i.split(".")[0]
            if key in self.node_score:
                self.node_score[key] += 1
            else:
                self.node_score[key] = 1

    def get(self, key: str) -> bytes:
        # 缓存索引
        if key not in self.node_map:
            return -1
        node = self.node_map[key]
        # 缓存管理
        self.score_heap.update(node,node.score-1)
        return node.value
    
    def put(self, key: str, value: bytes) -> None:
        # 缓存空间不足
        if len(self.node_map) >= self.capacity:
            # 删除缓存
            oldest_key = self.score_heap.extract_min()[1].key
            del self.node_map[oldest_key]
            
        # 缓存存储
        new_node = IACMANode(key, value, self.node_score[key])
        self.node_map[key] = new_node
        self.score_heap.insert(new_node,new_node.score)
        
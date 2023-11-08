import math


class MinHeap:
    """ 最小堆数据结构（所有非叶结点都小于其子结点） """

    def __init__(self, capacity=10):
        """ 初始化最小堆 """
        self.min_heap = []  # 最小堆
        """ 用列表存储最小堆的原因: 最小堆是“完全二叉树”，如果将结点按树的层次存储到列表中，
        则可以根据当前结点的索引定位到其父结点、左右两个子结点的位置 """
        self.capacity = capacity  # 最小堆容量（最多可以存的结点个数）
        self.size = 0  # 最小堆结点个数（当前堆中的结点个数）

    """
        公开函数
        1. 获取最小堆根结点 peek
        2. 弹出最小堆根结点 pop
        3. 压入最小堆新结点 push
    """

    def peek(self):  # 读取最小堆的根结点（最小结点）
        if self.size == 0:
            raise Exception("the min heap is empty")
        return self.min_heap[0]

    def pop(self):  # 弹出最小堆的根结点（最小结点）
        if self.size == 0:
            raise Exception("the min heap is empty")
        min_node = self.min_heap[0]
        self.min_heap[0] = self.min_heap[self.size - 1]  # 用最后一个结点填充空出来的根结点
        self.min_heap.pop()
        self.size -= 1
        self._heapify_down()  # 调整最小堆（将根结点放到正确的位置）
        return min_node

    def push(self, new_node):  # 压入最小堆一个新的结点
        self._ensure_extra_capacity()
        self.min_heap.append(new_node)
        self.size += 1
        self._heapify_up()  # 调整最小堆（将最后一个结点放到正确的位置）

    """
        私有函数
        1. 获取索引 _get_left_child_index、_get_right_child_index、_get_parent_index
        2. 有无结点 _has_left_child、_has_right_child、_has_parent
        3. 获取结点 _left_child、_right_child、_parent
        4. 交换结点 _swap
        5. 扩充容量 _ensure_extra_capacity
        6. 调整最小堆 _heapify_down、_heapify_up
    """

    @staticmethod
    def _get_left_child_index(parentIndex):  # 返回左子结点的索引
        return 2 * parentIndex + 1

    @staticmethod
    def _get_right_child_index(parentIndex):  # 返回右子结点的索引
        return 2 * parentIndex + 2

    @staticmethod
    def _get_parent_index(childIndex):  # 返回父结点的索引
        return math.floor((childIndex - 1) / 2)

    def _has_left_child(self, index):  # 是否有左子结点
        return self._get_left_child_index(index) < self.size

    def _has_right_child(self, index):  # 是否有右子结点
        return self._get_right_child_index(index) < self.size

    def _has_parent(self, index):  # 是否有父结点
        return self._get_parent_index(index) >= 0

    def _left_child(self, index):  # 取左结点
        return self.min_heap[self._get_left_child_index(index)]

    def _right_child(self, index):  # 取右结点
        return self.min_heap[self._get_right_child_index(index)]

    def _parent(self, index):  # 取父结点
        return self.min_heap[self._get_parent_index(index)]

    def _swap(self, index1, index2):  # 交换两个结点的值
        temp = self.min_heap[index1]
        self.min_heap[index1] = self.min_heap[index2]
        self.min_heap[index2] = temp

    def _ensure_extra_capacity(self):  # 确保最小堆有多余空间
        if self.size == self.capacity:
            self.capacity *= 2  # 扩充最小堆的容量为原来的 2 倍

    def _heapify_down(self):  # 调整最小堆的根结点
        """ 判断待比较结点是否比其子结点都小，如果不是，则与其最小子结点交换，
            直到待比较结点为叶结点 or 待比较结点比其子结点都笑 """
        index = 0
        while self._has_left_child(index):  # 待比较结点只有在有左子结点的情况下才可能有右子结点
            # 找待比较结点的最小子结点
            smallerIndex = self._get_left_child_index(index)
            if self._has_right_child(index) and self._right_child(index) < self._left_child(index):
                smallerIndex = self._get_right_child_index(index)
            # 待比较结点是否需要与其子结点交换
            if self.min_heap[index] < self.min_heap[smallerIndex]:
                break
            self._swap(index, smallerIndex)  # 交换待比较结点和其最小子结点
            index = smallerIndex  # 更新带比较结点的最新索引

    def _heapify_up(self):  # 调整最小堆的末结点
        """ 待比较结点是否比其父结点大，如果不是，则与其父结点交换，
            直到待比较结点是根结点（没有父结点）or 待比较结点大于其父结点"""
        index = self.size - 1
        while self._has_parent(index) and self._parent(index) > self.min_heap[index]:
            self._swap(self._get_parent_index(index), index)
            index = self._get_parent_index(index)

    """ 重写 __str__ __len__ """

    def __str__(self):
        if not self.min_heap:
            return "(MinHeap)->(heap is empty)"

        info = ""
        for node in self.min_heap:
            info += node.__str__()
        return info

    def __len__(self):
        return len(self.min_heap)

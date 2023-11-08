import os

from MinHeap import MinHeap


class HuffmanNode:
    """ 哈夫曼树的结点 """

    def __init__(self, char, freq):
        self.freq = freq  # 频度
        self.char = char  # 当 char=None，表示当前结点不是哈夫曼树的根结点
        self.left_child = None
        self.right_child = None

    def __str__(self):
        return f"{self.__class__.__name__}->(char={self.char},freq={self.freq})"

    def __eq__(self, other):  # 重写 ==
        return self.freq == other.freq

    def __lt__(self, other):  # 重写 <
        return self.freq < other.freq

    def __gt__(self, other):  # 重写 >
        return self.freq > other.freq

    def __le__(self, other):  # 重写 <=
        return self.freq <= other.freq

    def __ge__(self, other):  # 重写 >=
        return self.freq >= other.freq


class HuffmanTree:
    """ 哈夫曼树 """

    def __init__(self, path):
        self.path = path  # 文件路径
        self.minHeap = MinHeap()  # 初始化一个空的最小堆
        self.huffmanTree = None  # 哈夫曼树
        self.char_freq_mapping = {}  # “字符-频率” 映射关系
        self.char_codes_mapping = {}  # “字符-编码” 映射关系
        self.codes_char_mapping = {}  # “编码-字符” 映射关系

    # 接收文本输入，初始化 “字符-频率” 映射关系
    def _init_char_freq_mapping(self):
        with open(self.path, 'r+', encoding="utf-8") as file:
            text = file.read().strip()

            for char in text:
                if char not in self.char_freq_mapping.keys():
                    self.char_freq_mapping[char] = 0
                self.char_freq_mapping[char] += 1

    # 根据 “字符-频率” 映射关系初始化最小堆
    def _init_min_heap(self):
        if not self.char_freq_mapping:  # “字符-频率” 映射关系不存在，则进行对应的初始化操作
            self._init_char_freq_mapping()

        for key in self.char_freq_mapping.keys():
            node = HuffmanNode(key, self.char_freq_mapping[key])
            self.minHeap.push(node)

    # 初始化哈夫曼树
    def _init_huffman_tree(self):
        if not self.minHeap.min_heap:  # 最小堆为空，则进行对应的初始化操作
            self._init_min_heap()

        min_heap = self.minHeap
        while len(min_heap) > 1:  # 堆中最后一个结点，就是二叉树
            """ 从最小堆中取出最小的两个结点，将其合并为一个结点：将两个结点作为这个结点的子节点，将这个结点插入最小堆 """
            leaf_node1 = min_heap.pop()
            leaf_node2 = min_heap.pop()

            merged_node = HuffmanNode(None, leaf_node1.freq + leaf_node2.freq)
            merged_node.left_child = leaf_node1
            merged_node.right_child = leaf_node2

            min_heap.push(merged_node)

        self.huffmanTree = min_heap.pop()

    # 初始化 “字符-编码” 映射关系和 “编码-字符” 映射关系（递归生成）
    # 调用该方法时，node 只能取 self.huffmanTree
    def _init_char_codes_mapping(self, node, code=""):
        if not node:
            return

        if node.char:  # 叶节点
            self.char_codes_mapping[node.char] = code
            self.codes_char_mapping[code] = node.char
            return

        self._init_char_codes_mapping(node.left_child, code + "0")
        self._init_char_codes_mapping(node.right_child, code + "1")

    # 接收文本输入，基于 “字符-编码” 映射关系，返回对应的 Huffman 编码
    def _encode(self):
        if not self.char_codes_mapping:
            self._init_char_codes_mapping(self.huffmanTree)

        encoded_text = ""
        with open(self.path, 'r+', encoding="utf-8") as file:
            text = file.read().strip()

            for char in text:
                encoded_text += self.char_codes_mapping[char]
        return encoded_text

    # 将二进制字符串（Huffman 编码）转换，返回字节字符串
    def _binary_bytes(self):
        # 将二进制字符串补充为 8 的倍数
        encoded_text = self._encode()
        extra_codes = 8 - len(encoded_text) % 8  # 需要填补的位数
        encoded_text += extra_codes * "0"

        extra_info = f"{extra_codes:08b}"  # 用一个字节表示填补的位数（便于之后将二进制代码还原）
        encoded_text = extra_info + encoded_text  # 二进制字符串：补充位数B + 原本的 Huffman 编码 + 补充的0

        # 将修正好的二进制编码转换为字节数组
        binary_bytes = bytes([int(encoded_text[i:i + 8], 2) for i in range(0, len(encoded_text), 8)])

        return binary_bytes

    # 文件压缩
    def compress(self, path=None):
        if path:  # 可以通过 compress 函数重新定义路径
            self.path = path

        filename, extension_name = os.path.splitext(self.path)
        compressed_file_path = filename + ".bin"

        with open(compressed_file_path, "wb") as file:
            self._init_char_freq_mapping()  # 初始化 字符-频率 映射关系
            self._init_min_heap()  # 初始化最小堆
            self._init_huffman_tree()  # 初始化哈夫曼树
            self._init_char_codes_mapping(self.huffmanTree)  # 初始化 字符<->编码 映射关系
            binary_bytes = self._binary_bytes()  # 获得文本数据
            file.write(binary_bytes)

        original_file_size = os.path.getsize(self.path)
        compressed_file_size = os.path.getsize(compressed_file_path)
        print(f"\n{self.path} compressed success!")
        print(f"original file size: {original_file_size / 1024:.2f} KB")
        print(f"compressed file size: {compressed_file_size / 1024:.2f} KB")
        print(f"compress ratio is {(original_file_size - compressed_file_size) / original_file_size * 100:.2f} %\n")


if __name__ == "__main__":
    huffmanTree = HuffmanTree("example1.txt")
    huffmanTree.compress()
    huffmanTree = HuffmanTree("example2.txt")
    huffmanTree.compress()

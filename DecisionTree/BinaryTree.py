"""
    简要介绍
    1. NodeStatus: 结点状态枚举类，区分结点是叶结点还是非叶节点
    2. FeatureStatus: 结点最优化分属性状态枚举类，区分该属性是连续属性还是离散属性
    3. TreeNode: 结点类型
        - field
            feature_name 当前结点对应的数据集的最优化分属性
            feature_type 最优化分属性的类型(离散or连续)
            threshold 最优化分属性的判定分界值
            entropy 最优化分属性的信息增益
            samples 当前结点对应的数据集的样本数量
            node_type 当前结点的类型(叶结点or非叶节点)
            node_class 当前结点对应的数据集的大多数样本所属的类别
        - methods
            print_info() 当前结点的重要信息
            print() 打印当前结点的重要信息
    4. BinaryTree: 二叉树类型
        - fields
            tree 当前二叉树的数据(包括结点、左子树、右子树)
            node 当前二叉树的结点
            left_child 当前二叉树的左子树
            right_child 当前二叉树的右子树
        - methods
            insert_left() 插入左子树
            insert_right() 插入右子树
            level_order_traversal() 非递归层次遍历二叉树，并返回遍历信息
            print_level_order_traversal() 打印二叉树的层次遍历结果
            visualize() 可视化当前二叉树
    4. tnode: 测试结点类型
        - fields
            data 结点数据
        - methods
            print_info() 当前结点的重要信息
            print() 打印当前结点的重要信息

"""
from enum import Enum
from queue import Queue
from graphviz import Digraph
import time


class NodeStatus(Enum):
    """ 结点类型 """
    LEAF = 0
    NON_LEAF = 1


class FeatureStatus(Enum):
    """ 属性类型 """
    DEFAULT = 2
    CONTINUOUS = 1
    DISCRETE = 0


class TreeNode:
    """ 结点类型 """

    # 创建一个结点实例
    def __init__(self, feature_name=None, feature_type=None, threshold=None, entropy=None, samples=None, node_type=None,
                 node_class=None):
        self.feature_name = feature_name  # 结点(最优化分)属性, 无则 None, 以下结点属性都意为结点最优属性
        self.feature_type = feature_type  # 结点属性的类型(FeatureStatus.CONTINUOUS/FeatureStatus.DISCRETE), 无则 None
        self.threshold = threshold  # 结点属性判定阈值, 无则 None
        """
            对于离散类型属性(FeatureStatus.CONTINUOUS), 意味着如果 data[feature_name] == threshold, 去左子树, 否则去右子树
            对于连续类型属性(FeatureStatus.DISCRETE), 意味着如果 data[feature_name] <= threshold, 去左子树, 否则去右子树
        """
        self.entropy = entropy  # 结点属性的信息增益, 无则 None
        self.samples = samples  # 结点包含的样本数据数量, 无则 None
        self.node_type = node_type  # 结点类型 (NodeStatus.LEAF/NodeStatus.NON_LEAF), 必须有值
        self.node_class = node_class  # 结点类别 (即该结点包含的全部样本，大多数属于什么类别，或者说样本标签是什么), 必须有值
        self.depth = 0  # 结点深度

    # 获取结点信息
    def print_info(self):
        node_class = "Approval" if self.node_class == 1 else "Rejected"

        if self.node_type == NodeStatus.LEAF:
            node_info = "node info:\n" \
                        f"class={node_class}"
        elif self.node_type == NodeStatus.NON_LEAF:
            distinguish_flag = "==" if self.feature_type == FeatureStatus.DISCRETE else "<="
            node_info = "node info:\n" \
                        f"\t{self.feature_name}{distinguish_flag}{self.threshold}\n" \
                        f"\tentropy={self.entropy}\n" \
                        f"\tsamples={self.samples}\n" \
                        f"\tclass={node_class}\n"
        else:
            raise Exception("nodes whose type is not known cannot be printed")

        return node_info

    # 打印结点信息
    def print(self):
        node_info = self.print_info()
        print(node_info)


class BinaryTree:
    """ 二叉树类型: 列表表示 """

    """
        一个二叉树由一个列表表示: 列表第一个元素是树的结点, 列表第二个元素是树的左子树, 列表第三个元素是树的右子树
    """

    # 创建一个二叉树实例
    def __init__(self, node):
        self.tree = [node, None, None]
        self.node = self.tree[0]  # TreeNode 类型
        self.left_child = self.tree[1]  # BinaryTree 类型
        self.right_child = self.tree[2]  # BinaryTree 类型

    # 插入左子树
    def insert_left(self, child):
        if not self.left_child:  # 当前树的左子树为空，则将新的子树作为左子树插入
            self.tree[1] = child
            self.left_child = self.tree[1]
        else:  # 如果当前树的左子树不为空，则报错，表示子树插入位置错误
            raise Exception("the left child of the current tree is not empty, cannot be inserted")

    # 插入右子树
    def insert_right(self, child):
        if not self.right_child:  # 当前树的左子树为空，则将新的子树(也有可能是结点)作为左子树插入
            self.tree[2] = child
            self.right_child = self.tree[2]
        else:  # 如果当前树的左子树不为空，则报错，表示子树插入位置错误
            raise Exception("the right child of the current tree is not empty, cannot be inserted")

    # 非递归层次遍历二叉树并返回遍历列表信息，队列实现
    def level_order_traversal(self):
        queue = Queue()
        queue.put(self)  # 存放树对象(根据树对象的 node 属性可以判断这树是不是叶结点)
        traversal_res = []  # 存放前序遍历结果

        while not queue.empty():
            current_tree = queue.get()  # 取出一个树对象
            traversal_res.append(current_tree.node)
            if current_tree.left_child:
                queue.put(current_tree.left_child)
            if current_tree.right_child:
                queue.put(current_tree.right_child)

        return traversal_res

    # 打印二叉树的层次遍历结果
    def print_level_order_traversal(self):
        traversal_res = self.level_order_traversal()
        for node in traversal_res:
            node.print()

    # 可视化二叉树，通过队列进行非递归的实现
    def visualize(self, directory='test_output', file_name="VisualizedBinaryTree", file_format="png"):
        # 创建有向图，设置相关参数
        graph = Digraph(name=file_name, filename=file_name,
                        directory=directory, format=file_format)  # 创建一个有向图，并且设置文件保存地址及输出格式
        graph.graph_attr['dpi'] = '300'  # 设置输出图片的分辨率
        graph.attr('node', shape='box')  # 统一修改 node 对象的形状为 box

        # 为每一个结点分配一个唯一的 ID
        id_counter = 0

        # 使用字典存储结点和对应的 ID (KEY:结点;VALUE:ID), 从而可以根据结点查询对应的 ID
        node_to_id = {}

        # 创建根结点
        root_node = self.node
        root_id = f"node_{id_counter}"
        node_to_id[root_node] = root_id

        graph.node(root_id, label=root_node.print_info())

        # 根据二叉树的内容创建 node 对象和 edge 对象
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            current_tree = queue.get()  # 取出一个树对象
            parent_id = node_to_id[current_tree.node]
            # 如果这个树对象有子树，则创建两个子节点，并且和父结点相连；否则从队列取出下一个树对象
            # 若这个树对象只有一个子树，则另一个子树生成一个不可见的 node 对象
            # 生成子树 node 对象后，将子树添加进队列，留用下一层的遍历
            if not current_tree.left_child and not current_tree.right_child:  # 当前树对象没有子树
                continue

            # 处理左子树
            if current_tree.left_child:  # 当前树对象有左子树
                id_counter += 1
                left_id = f"node_{id_counter}"
                node_to_id[current_tree.left_child.node] = left_id
                graph.node(left_id, label=current_tree.left_child.node.print_info())  # 创建左子树结点
                graph.edge(parent_id, left_id)  # 连接父结点和左子树结点
                queue.put(current_tree.left_child)
            else:  # 当前树对象没有左子树，则随机生成一个不可见的 node 对象并连接
                id_counter += 1
                left_id = f"node_{id_counter}"
                graph.node(left_id, label="", shape="point")  # 生成一个具有随机值的结点(表示空结点)
                graph.edge(parent_id, left_id, style='invis')  # 将父结点和空结点连接，并且不显示

            # 处理右子树
            if current_tree.right_child:  # 当前树对象有右子树
                id_counter += 1
                right_id = f"node_{id_counter}"
                node_to_id[current_tree.right_child.node] = right_id
                graph.node(right_id, label=current_tree.right_child.node.print_info())
                graph.edge(parent_id, right_id)
                queue.put(current_tree.right_child)
            else:  # 当前树对象没有右子树，则随机生成一个不可见的 node 对象并连接
                id_counter += 1
                right_id = f"node_{id_counter}"
                graph.node(right_id, label="", shape="point")  # 生成一个具有随机值的结点(表示空结点)
                graph.edge(parent_id, right_id, style='invis')  # 将父结点和空结点连接，并且不显示
        graph.render(view=True)


# 测试节点类
class tnode:
    def __init__(self, data):
        self.data = data

    def print(self):
        print(self.data)

    def print_info(self):
        return self.data


if __name__ == '__main__':
    treeA = BinaryTree(tnode("A"))
    treeB = BinaryTree(tnode("B"))
    treeC = BinaryTree(tnode("C"))
    treeD = BinaryTree(tnode("D"))  # leaf
    # treeE = BinaryTree(tnode("E"))  # leaf
    # treeF = BinaryTree(tnode("F"))  # leaf
    # treeM = BinaryTree(tnode("M"))
    treeA.insert_left(treeB)
    treeB.insert_left(treeC)
    treeC.insert_left(treeD)
    treeA.visualize()

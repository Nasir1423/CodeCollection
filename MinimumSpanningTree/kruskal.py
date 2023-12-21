import networkx as nx
import matplotlib.pyplot as plt


class UnionFind:
    def __init__(self, size):
        # 初始化并查集数据结构
        self.parent = [i for i in range(size)]  # 父节点数组，用于跟踪每个集合的代表元素
        self.rank = [0] * size  # 秩数组，用于优化并查集的合并操作

    def find(self, x):
        # 查找操作，确定元素x所属的集合（代表元素）
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩，优化查找操作
        return self.parent[x]

    def union(self, x, y):
        # 合并操作，合并以x和y为代表的两个集合
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # x和y已经在同一个集合中

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += 1

        return True  # 集合成功合并


def kruskal(edges, n):
    # Kruskal算法，寻找图的最小生成树
    edges.sort(key=lambda x: x[2])  # 按成本升序排序边
    uf = UnionFind(n)  # 创建并查集数据结构
    min_cost = 0
    result = []  # 用于存储最小生成树的边

    for edge in edges:
        u, v, cost = edge
        if uf.union(u - 1, v - 1):
            # 如果添加这条边不形成环，则将其加入结果
            min_cost += cost
            result.append(edge)

    return min_cost, result


def build_adjacency_matrix(edges, n):
    # 从边的列表构建邻接矩阵
    adjacency_matrix = [[float('inf')] * n for _ in range(n)]

    for edge in edges:
        u, v, cost = edge
        adjacency_matrix[u - 1][v - 1] = cost
        adjacency_matrix[v - 1][u - 1] = cost

    return adjacency_matrix


def visualize_graph(adjacency_matrix):
    # 使用networkx库绘制图形
    G = nx.Graph()
    n = len(adjacency_matrix)

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] != float('inf'):
                G.add_edge(i + 1, j + 1, weight=adjacency_matrix[i][j])

    pos = nx.spring_layout(G)  # 定义节点的布局
    plt.figure()  # 创建一个新的图形对象
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10)  # 绘制图形
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)  # 绘制边上的权重
    plt.title("Minimum Spanning Tree")
    plt.show()


def main():
    # 输入城镇数目和候选道路数目
    N, M = map(int, input().split())

    # 输入候选道路信息
    edges = [list(map(int, input().split())) for _ in range(M)]

    # 使用Kruskal算法求解“畅通工程”的最小成本
    min_cost, result = kruskal(edges, N)

    if len(result) == N - 1:
        # 输出最小成本
        print(f">>>道路畅通的最小成本为 {min_cost}")

        # 输出“畅通工程”的邻接矩阵
        adjacency_matrix = build_adjacency_matrix(result, N)
        print(">>>最小成本的通路对应的邻接矩阵为")
        for row in adjacency_matrix:
            print(" ".join(map(str, row)))

        # 可视化邻接矩阵对应的图形
        visualize_graph(adjacency_matrix)
    else:
        print("警告：输入数据无法满足道路畅通")


if __name__ == "__main__":
    main()

# 示例输入:
# 6 9
# 1 2 1
# 1 3 2
# 2 3 6
# 2 4 11
# 3 4 9
# 3 5 13
# 4 5 7
# 4 6 3
# 5 6 4

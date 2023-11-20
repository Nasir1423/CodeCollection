"""
    八皇后问题介绍: 8*8 的国际棋盘上安排 8 个皇后，使得没有两个皇后互相攻击。
    1. 互相攻击的情形: 两个皇后处于同一行、同一列或同一对角线
    2. 问题分析
        (1) 因为两个皇后不能放在一行，所以每个皇后都在不同的行上（或者说是每行都有一个皇后）
        (2) 无妨定义一个有八个分量的向量 x={x_1,x_2,...,x_8} 作为一种八皇后的一种可能的布局，每个分量取值 1~8
        (3) 问题转化为在 (2) 的情况下，满足两个皇后不在同一列 or 同一对角线两个条件即可，即
            ① x_i == x_j
            ② x_i - x_j = i - j or x_i - x_j == j - i
        (4) 在 (3) 的情况下，因为每行有 8 个位置，因而共有 8^8 种可能存在的布局，又因为任意两个皇后不能在同一列，则
        搜索空间可以下降到 8! 个布局（即每个布局 x 的 8 个分量对应数字 1~8 的一个排列）。
    3. 算法实现：回溯法递归求解，生成并以深度优先的方式搜索一颗完全四叉有根树
        (1) 第 i 层结点对应皇后在第 i 行的可能放置情况
        (2) "合法": 即一个不互相攻击的 8 皇后布局
            "部分": 即一个不互相攻击的少于 8 个的皇后布局
"""


# 递归函数，用于确定当前布局（layout）每一行（row：0~7）皇后的位置
def place_queen(layout, row):
    for place in range(1, 9):
        layout[row] = place

        # print(f"第{row+1}层: {place}")

        if legal(layout):  # 如果当前布局是一个合法布局，则得到一个解，递归结束并返回 True
            return True
        elif partial(layout, row):  # 如果当前布局是一个部分，则继续进行递归求解
            if place_queen(layout, row + 1):
                return True
        # 如果皇后位置下无法找到一个解，则改变皇后位置
    # 如果所有皇后位置下都无法找到一个解，则递归回溯，改变上一个皇后在其所在行的位置


# 判断当前布局是否合法
def legal(layout):
    if 0 in layout:
        return False
    return partial(layout, len(layout) - 1)


# 判断当前布局（layout 的前 row 行）是否是一个部分
def partial(layout, row):
    part = layout[0:row + 1]

    if len(set(part)) != len(part):  # 判断是否存在两个皇后在同一列的情形
        return False

    for i in range(0, row):  # 判断是否存在两个皇后在对角线的情形
        if layout[i] - layout[i + 1] == 1 or layout[i] - layout[i + 1] == -1:
            return False

    return True


chess_layout = [0 for i in range(8)]  # 初始化布局向量

flag = place_queen(chess_layout, 0)

print(f"八皇后一个合法的布局为 {chess_layout}") if flag else print("此问题无解")

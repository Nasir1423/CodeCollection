"""
    算法实现参考
        [1]课件
        [2]https://www.kancloud.cn/digest/pieces-algorithm/163624
"""


def lcs_length_arr(A, B):
    """
        返回数组 L\n
        解释：
            L[i,j] 即 A[1:i] 和 B[1:j] 的最大子序列长度(假定第一个元素的下标是1)
    """
    n = len(A)  # n+1 对应 table L 的 row
    m = len(B)  # m+1 对应 table L 的 col

    """ 
        功能: 初始化 table L，规格为 (n+1)*(m+1)，全部赋值为 0
            其中 L[0,:]=0, L[:,0]=0 符合规则 if i or j=0, then L[i,j]=0
        可改进之处: L[1:n,1:m] 先不初始化，后续根据递推公式再进行赋值，提高程序运行速度
        意义: L 的每一个值 L[i,j] 表示 A[1:i] 和 B[1:j] 的最大公共子序列的长度
        注意: L[i,j] 中的 i 表示 A 的第 i 个元素(A 元素下标看作从 1 开始)
    """
    L = [[0 for j in range(m + 1)] for i in range(n + 1)]

    """
        根据递归规则，填充 table L[1:n,1:m]
        因为 L[i,j] 对应的是 A[i+1] 和 B[j+1] 因此需要进行下标修正
    """
    for i in range(n):
        for j in range(m):
            if A[i] == B[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i + 1][j], L[i][j + 1])

    return L


def lcs_all(A, B, i, j, L, lcs_set, lcs_str=""):
    """
        返回字符串 A、B 的所有最长子序列\n
        参数：
            A、B：str 类型，传入两个字符串(第一次传参后不再修改)\n
            lcs_set：set 类型，存放所有的最长子序列(多次添加值)\n
            lcs_str：str 类型，存放一个最长子序列是(多次修改)\n
            i, j：表示要从 lcs_length_arr[0:i,0:j] 中进行路径回溯(多次修改)\n
            L：二维列表类型，lcs_length_arr(A, B)
        思路：
            路径回溯
    """
    """
        路径回溯规则
        if L[i][j] = L[i-1]L[j-1] + 1 and A[i-1] == B[j-1](subscript correction)
            then back one space to the upper left
        else, go back one space to the left or up
            if left > up, back to left
            elif left < up, back to up
            else 两边分别回溯(通过递归实现)
    """
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:  # 当前元素属于最长子序列的一部分，记录该元素，并且下一步到最近左上角
            lcs_str = A[i - 1] + lcs_str
            i -= 1
            j -= 1
        else:
            if L[i - 1][j] > L[i][j - 1]:  # 此时应该向上走
                i -= 1
            elif L[i - 1][j] < L[i][j - 1]:  # 此时应该向左走
                j -= 1
            else:  # 此时两个方向应该分别回溯
                lcs_all(A, B, i - 1, j, L, lcs_set, lcs_str)  # 向上回溯
                lcs_all(A, B, i, j - 1, L, lcs_set, lcs_str)  # 向左回溯
                return

    lcs_set.add(lcs_str)
    return


if __name__ == '__main__':
    """ 代码测试 """
    str1 = ['xyxxzxyzxy', 'ABCBDA', 'abcbced', 'abcedfg', 'happyisgood']
    str2 = ['zxzyyzxxyxxz', 'BDCABA', 'acbcbcef', 'gfdecba', 'unhappyisbad']

    i = 1
    for A, B in zip(str1, str2):
        vector = set()
        lcs_length = lcs_length_arr(A, B)[len(A)][len(B)]  # 获取一组测试样例的最大相同子序列的长度
        lcs_all(A, B, len(A), len(B), lcs_length_arr(A, B), vector)  # 获取一组测试样例的所有最大相同子序列
        print(fr"=====测试样例{i}======")
        print("1. 字符串内容：A=", A, "B=", B)
        print("2. LCS of A and B is ", lcs_length)
        print("3. all LCS are ", vector)
        i += 1

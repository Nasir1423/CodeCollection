def candidate(m): # 返回列表的候选"多数"
    j = m
    x = L[m]
    count = 1
    while j<len(L)-1 and count>0:
        j += 1
        if L[j]==x:
            count += 1
        else:
            count -= 1
    if j==len(L)-1:
        return x
    else:
        return candidate(j+1)

""" return x if majority exists or none """
def isValidCandidate(): # 检验候选"多数"是不是"多数"，并且给予不同的返回值
    x = candidate(0)
    count = 0
    for i in range(len(L)):
        if L[i] == x:
            count += 1
    if count>(len(L)/2):
        return x
    else:
        return None
        

# 通过用户输入,得到一个数字列表
L = input("请输入一串数字，以空格分隔，回车结束\n>>> ")
L = L.split()
L = [int(strnum) for strnum in L] # 获得数字列表

x = isValidCandidate()
if x:
    print("列表L存在多数，多数为：",x)
else:
    print("列表L不存在多数")

import os
import datetime

def Addr2Shard(addr, shardNum = 16):
    num = int(addr[-5:],16)
    return num % shardNum
    
    
def new_file(prex, testdir):
    #列出目录下所有的文件
    # print(testdir)
    list = os.listdir(testdir)
    #对文件修改时间进行升序排列
    list.sort(key=lambda fn:os.path.getmtime(testdir+'\\'+fn))
    for i in list[::-1]:
        if(i.startswith(prex)):
            return i
    return "None"
    
    
    
# 拆分整数
def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n
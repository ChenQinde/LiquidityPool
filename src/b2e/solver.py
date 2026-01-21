from src.b2e.MKP_gurobi import slove2Gurobi
from src.b2e.data import read_data
from src.b2e.utils import utils
from src.b2e.b2erounding import B2ERounding
import time
import os
import random
import matplotlib.pyplot as plt
from b2erounding import B2ERounding
from multiprocessing import Process
import json

with open('config/simulation_config.json') as config_file:
    config = json.load(config_file)

txsPath = config['txsPath']

# print(path)
# all_data = []
# with open(txsPath, "r") as f:
    # all_data = f.readlines()[1:]
    
enoughRate = 0.5
    
def get_data(k_number, shardNum, slice, dataPath, capacitys_input):
    txs = read_data.readData(all_data)
    txs = txs[slice[0]:slice[1]]
    ctxs = []
    capacitys = []
    for tx in txs:
        if(tx[1] == "None" or tx[2] == "None"):
            continue
        if(utils.Addr2Shard(tx[1], shardNum) != utils.Addr2Shard(tx[2], shardNum) and tx[3] !=0):
            ctxs.append([tx,utils.Addr2Shard(tx[1], shardNum),utils.Addr2Shard(tx[2], shardNum)])
            
    # thresholdBalances = sum(capacitys_input) * enoughRate
    # ctxValue = 0
    
    new_ctxs = []
    for ctx in ctxs:
        # ctxValue += int(ctx[0][-2])
        if(int(ctx[0][-2]) <= 10):
            continue
        new_ctxs.append(ctx)
        # if(ctxValue >= thresholdBalances):
            # break
            
    for i in range(k_number):
        # capacitys.append(200000000000000000000) # 2e20
        # capacitys.append(2000000000000000000) # 2e18
        capacitys.append(capacitys_input[i]) # 2e18
    read_data.writeFile(k_number, slice, new_ctxs, capacitys, dataPath)    
    return txs, new_ctxs, capacitys

        
def run_example_rounding(examplePath = "./data/example4/", iter_num = 300, alpha = 1):
    '''
    run b2e rounding
    '''
    fileDir = os.listdir(examplePath)
    var_type = "LINEAR"
    
    process_list = []

    # for index, dir in enumerate(fileDir):
    for index  in range(200):
        dir = "item" + str(index)
        dataPath = examplePath + dir + "/data1.txt"
        print(dataPath)
        resultPath = examplePath + dir + "/"#+ "/alpha_" + str(alpha) + "_"
        
        # round_time, n_value = B2ERounding(dataPath, var_type, resultPath, iter_num, alpha = alpha) 
        
        p = Process(target=B2ERounding,args=(dataPath, var_type, resultPath, iter_num,alpha,))
        p.start()
        process_list.append(p)
        
        if(len(process_list) == 25):
            for i in process_list:
                p.join()
            process_list = []
    for i in process_list:
        p.join()
  

# 生成用例  
def generate_example(thedir = "./data/example/", number = 10):
    
    if not os.path.isdir(thedir):
        os.mkdir(thedir)
        
    # 生成 number 数量的测试用例
    for i in range(number):
    
    
        itemDir = thedir + "item" + str(i)
        if not os.path.isdir(itemDir):
            os.mkdir(itemDir)
    


        # broker 数量
        
        # k_number = random.randint(50,100)
        
        
        # 分片数量
        shardNum = 64
        
        
        # 交易源自于原始数据的切片位置
        slice = [i * 50000, (i + 1) * 50000]
        
        
        # 读取后数据存储位置
        dataPath = itemDir + "/data1.txt"
        
        
        # broker 金额设置
        with open("D:\博一\BrokerHub\motivation\Broker.txt", "r") as f:
            capacitys_input = [int(i) for i in f.read().strip()[1:-1].split(",")]
        k_number = len(capacitys_input)
        # capacitys_input = [2000000000000000000 for i in range(k_number)] # 2e18
        
        txs, ctxs, capacitys = get_data(k_number, shardNum, slice, dataPath,capacitys_input)
        
        with open(itemDir+ "/Ctx.csv","w") as f1:
            f1.write("id,Tx sour.shard, Tx dest.shard, Fee, Value, Ratio\n")
            for index,ctx in enumerate(ctxs):
                f1.write(str(index) + "," + str(ctx[1])+ "," + str(ctx[2]) + "," + str(ctx[0][-1]) + "," + str(ctx[0][-2]) + "," + str(ctx[0][-1]/ctx[0][-2]) + "\n")
    
    
    
if __name__ == "__main__":
    # 生成数据
    # generate_example(thedir = "./data/BalanceAdd_5wCtx/", number = 50)
    
    run_example_rounding("./data/diff_item200_witoutHub/")
    
    # 运行一个文件夹下所有用例
    # for i in range(100):
    # run_example_rounding("./data/BalanceAdd_bigSmall0/example" + str(0) + "/")
    
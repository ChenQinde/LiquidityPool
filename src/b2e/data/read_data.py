import datetime
import csv
from decimal import Decimal


def get_runtime():
    now = datetime.datetime.now()
    name = "{0}-{1}-{2} {3}h{4}m{5}s".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    return name


# tx (id, source shard id, dest. shard id, Value, Fee)   
def readData(data):
    txs = []
    num = 0
    for i in data:
        i = i.strip().split(",")
        if(i[13] != "None" or i[7] == "1"):
            continue
        tx = (num, i[3], i[4], int(i[8]), eval(i[10]) * int(i[11]))
        txs.append(tx)
        num += 1
    return txs
    
    
# write data1.txt    
def writeFile(k_number, slice, txs, capacitys, path = "./test.txt"):
    with open(path, "w") as f:
        f.write("Knapsack numbr : " + str(k_number) + "\n")
        f.write("Slice : " + str(slice[0]) + "," + str(slice[1]) + "\n")
        f.write("Ctxs : " + str(len(txs)) + "\n")
        item = []
        for tx in txs:
            # tx (Fee, Value, Source Shard, Dest. Shard)  
            f.write("(" + str(tx[0][-1]) + "," + str(tx[0][-2])+ "," + str(tx[1])+ "," + str(tx[2]) + ") ")
        f.write("\n")
        f.write(str(capacitys[0]))
        for capacity in capacitys[1:]:
            f.write(", " + str(capacity))
   
# write txt         
def writeSol(status, round_times, cpu_times, using_time, k_number, txsNumber, ctxs, capacitys, s_x, path):
    with open(path + get_runtime() + "{0}K{1}T.txt".format(k_number, txsNumber), "w") as f:
        for cpu_time in cpu_times:
            f.write("CpuTime : " + str(cpu_time) + "\n")
        f.write("Time : " + str(using_time) + "\n")
        f.write("Knapsack numbr : " + str(k_number) + "\n")
        f.write("Ctx : " + str(len(ctxs)) + "\n")
        if(s_x == -1):
            f.write("Objective value is : " + str(-1) + "\n")
            return;
        text = ""
        weights_capacities = [0 for i in capacitys]
        for key in s_x.keys():
            weights_capacities[key[1]] += s_x[key] * ctxs[key[0]][1]
            
        for i in range(len(capacitys)):
            text += "C{0}:{1}/{2}={3}%\n".format(i, weights_capacities[i], capacitys[i], (weights_capacities[i] * 1.0 / capacitys[i] * 100))
        text += "=" * 50 + "\n"
        value = 0
        for key in s_x.keys():
            value += s_x[key] * ctxs[key[0]][0]
            if(s_x[key] == 0.0):
                continue
            text += "x({0}, {1}, {2})".format(key[0], key[1], ctxs[key[0]][2]) + " : " + str(s_x[key])+ "\n"
        
        f.write("Objective value is : " + str(value) + "\n")
        for i in round_times:
            f.write("Rounding time : " + str(i) + "\n")
        f.write("Status : " + status + "\n")
        all_weight = 0
        all_capacity = 0
        for i in range(len(weights_capacities)):
            all_weight += weights_capacities[i]
            all_capacity += capacitys[i]
            
        f.write("Usage : {0}%\n".format(str(all_weight / all_capacity * 100)))
        f.write(text)  


def write_csv(k_number, txsNumber, ctxs, capacitys, ns_x, feeRatio, resultPath, scale, sorted_ids):
    # print(f"capacitys : {str(capacitys[0])}")
    # formatted_number = format(capacitys[0], '.0f')
    # print(formatted_number)
    with open(resultPath + ".csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Number of CTXs served", "revenue rate(Fee * feeRatio / brokerAmount)", "Usage", "Revenue", "Balance"])

        weights_capacities = [0 for _ in capacitys]
        brokerRevenue = [0 for _ in capacitys]
        
        for key in ns_x.keys():
            weights_capacities[key[1]] += ns_x[key] * ctxs[key[0]][1]
            brokerRevenue[key[1]] += ns_x[key] * ctxs[key[0]][0]
            
        for i in range(len(capacitys)):
            served_ctxs_number = 0 
            revenueRate = Decimal('0')
            
            for j in range(len(ctxs)):
                served_ctxs_number += ns_x[(j,i)]
                revenueRate += Decimal(str(ns_x[(j,i)])) * Decimal(str(feeRatio)) * Decimal(str(ctxs[j][0]))
            
            revenueRate /= Decimal(str(capacitys[i]))
            
            writer.writerow([
                sorted_ids[i],
                served_ctxs_number,
                f"{revenueRate:.20f}",
                f"{(weights_capacities[i] * 100.0 / capacitys[i]):.2f}%",
                f"{int(brokerRevenue[i] * scale)}",
                f"{int(capacitys[i] * scale)}"
            ])
    
def write_ctx_csv(k_number, txsNumber, ctxs, capacitys, ns_x, feeRatio, resultPath, sorted_ids):
    with open(resultPath + "/Ctx.csv", "r") as f:
        data = f.readlines()
        res = [line.strip().split(",") for line in data]

    with open(resultPath + "/Ctx_broker.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(res[0] + ["ID"])
        
        for i in range(len(capacitys)):
            for j in range(len(ctxs)):
                if ns_x[(j,i)] > 0:
                    res[j + 1].append(str(sorted_ids[i]))
        
        writer.writerows(res[1:])
    print(resultPath + "/Ctx.csv")
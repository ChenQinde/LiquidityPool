import os
import json
from typing import List, Dict, Tuple, Any
import sys
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, int):
            return str(obj)  # Convert all integers to strings to avoid scientific notation
        return super(DecimalEncoder, self).default(obj)

class B2ETester:
    def __init__(self, config: Dict, B2ERoundingClass=None):
        """初始化B2E测试器

        Args:
            config: 配置字典，包含测试所需的所有参数
            B2ERoundingClass: B2E算法类，如果为None则尝试动态导入
        """
        self.config = config
        
        # 确保输出目录存在
        os.makedirs(self.config["output_path"], exist_ok=True)
        
        self.B2ERoundingClass = B2ERoundingClass
        
        # 记录初始资金情况
        self.broker_ids = []
        self.broker_funds = []
        self.broker_original_funds = {}
        self.b2e_rates = {}
        self.test_results = []
        
    def load_data(self, data_file: str) -> None:
        """从指定文件加载测试数据

        Args:
            data_file: 数据文件路径
        """
        print(f"加载数据文件: {data_file}")
        with open(data_file, "r") as f:
            allData = f.readlines()
            
            # 解析交易数量信息
            slice = allData[1].strip().split(" : ")[1].split(",")
            self.current_txs_number = int(slice[1]) - int(slice[0])
            
            # 解析交易项数据
            data = allData[3:]
            self.current_items = []

            fee_amount = 0
            transaction_amount = 0
            for item in data[0].strip().split(" "):
                item = item[1:-1].split(",")
                transaction_amount += int(item[0])
                fee_amount += int(item[1])
                self.current_items.append((int(item[0]), int(item[1]), item[2], item[3]))
                
            print(f"加载了 {len(self.current_items)} 个交易项")
            print(f"总交易金额: {transaction_amount}, 总手续费: {fee_amount}")
    
    def set_broker_funds(self, broker_funds: List[Tuple[str, float]]) -> None:
        """设置经纪人资金

        Args:
            broker_funds: 经纪人ID和资金的列表 [(id, fund), ...]
        """
        # 按资金量排序
        sorted_brokers = sorted(broker_funds, key=lambda x: x[1], reverse=True)
        
        # 分离ID和资金
        self.broker_ids = [b[0] for b in sorted_brokers]
        self.broker_funds = [b[1] for b in sorted_brokers]
        
        # 保存原始资金，用于后续计算收益率
        self.broker_original_funds = {b[0]: b[1] for b in broker_funds}
        
        print(f"设置了 {len(broker_funds)} 个经纪人的资金")
        
    def add_broker_fund(self, add_broker_funds: List[Tuple[str, float]]) -> None:
        
        broker_funds = [(self.broker_ids[i],self.broker_funds[i]) for i in range(len(self.broker_ids))]
        for i in add_broker_funds:
            broker_funds.append((i[0],i[1]))
        
        
        self.set_broker_funds(broker_funds)
        
    def load_broker_funds_from_file(self, file_path: str) -> None:
        """从文件加载经纪人资金，每行格式为"账户ID 金额"

        Args:
            file_path: 资金文件路径
        """
        print(f"从文件加载经纪人资金: {file_path}")
        broker_funds = []
        
        with open(file_path, "r") as f:
            balances = [i.strip() for i in f.readlines()]
            for broker_id, balance in enumerate(balances):
                broker_funds.append((broker_id, int(balance)))
                
        # broker_funds.append(("BrokerHub1", 1e18))
        # broker_funds.append(("BrokerHub2", 1e18))
        
        if broker_funds:
            self.set_broker_funds(broker_funds)
        
    def run_b2e(self, result_path: str = None) -> Dict:
        """运行B2E算法

        Args:
            result_path: 可选的结果输出路径

        Returns:
            包含收益和收益率的字典
        """
        if not result_path:
            result_path = os.path.join(self.config["output_path"], "test_result")
        
        # 确保结果目录存在
        os.makedirs(result_path, exist_ok=True)
        
        print("运行B2E算法...")
        using_time, value, sorted_funds = self.B2ERoundingClass(
            dataEpoch = [
                self.current_items,
                self.broker_funds,
                self.current_txs_number
            ],
            var_type = self.config["b2e"]["varType"],
            resultPath = result_path,
            iter_num = self.config["b2e"]["iterNum"],
            alpha = self.config["b2e"]["alpha"],
            feeRatio = self.config["b2e"]["feeRatio"],
            sorted_ids = self.broker_ids
        )
        
        # 将ID和收益进行对应
        id_earning_dict = {}
        for i, broker_id in enumerate(self.broker_ids):
            id_earning_dict[broker_id] = sorted_funds[i]
        
        # 计算收益率
        rates = {
            k: v / float(self.broker_original_funds[k]) if float(self.broker_original_funds[k]) != 0 else 0 
            for k, v in id_earning_dict.items()
        }
        
        # 更新收益率记录
        self.b2e_rates.update(rates)
        
        result = {
            'earnings': id_earning_dict, 
            'rates': rates, 
            "original_funds": self.broker_original_funds
        }
        
        print(f"B2E算法完成，耗时: {using_time}秒")
        return result

    def save_results(self):
        with open(f'{self.config["output_path"]}/b2e_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        

    def run_single_test(self) -> Dict:
        """执行单次B2E测试，使用当前配置

        Returns:
            B2E算法结果
        """
        print("执行单次B2E测试...")
        
        # 运行B2E
        # result_path = os.path.join(self.config["output_path"], "item0")
        b2e_result = self.run_b2e(self.config["output_path"])
        
        # 保存结果
        test_result = {
            'earnings': b2e_result['earnings'],
            'rates': b2e_result['rates'],
            'original_funds': b2e_result['original_funds']
        }
        
        # 将结果添加到测试结果列表
        self.test_results.append(test_result)
        
        # 打印部分结果
        print("\n--- B2E 测试结果 ---")
        print(f"总参与者数量: {len(b2e_result['earnings'])}")
        
        # 打印前5个收益最高的结果
        sorted_earnings = sorted(b2e_result['earnings'].items(), key=lambda x: x[1], reverse=True)
        print("\n收益最高的前5个参与者:")
        for i, (broker_id, earning) in enumerate(sorted_earnings[:5], 1):
            print(f"{i}. ID: {broker_id}, 收益: {earning}, 收益率: {b2e_result['rates'][broker_id]:.4f}, 原始资金: {b2e_result['original_funds'][broker_id]}")
        
        return b2e_result
        
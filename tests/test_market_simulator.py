import unittest
import json
import os
from src.market.market_simulator import MarketSimulator

class TestMarketSimulator(unittest.TestCase):
    def setUp(self):
        self.config_path = 'config/simulation_config.json'
        
        # 创建临时的配置文件和数据文件用于测试
        self.create_test_files()
        
        self.simulator = MarketSimulator(self.config_path)

    def tearDown(self):
        # 删除临时文件
        os.remove(self.config_path)
        os.remove(self.simulator.config['brokerBalancePath'])
        os.rmdir(os.path.dirname(self.simulator.config['brokerBalancePath']))
        os.rmdir(self.simulator.config['dataPath'])

    def create_test_files(self):
        config = {
            "brokerhubs": [
                {
                    "id": "BrokerHub1",
                    "initial_funds": 1000000,
                    "initial_tax_rate": 0.01,
                    "min_tax_rate": 0.001,
                    "max_tax_rate": 0.1,
                    "learning_rate": 0.01,
                    "memory_size": 5
                }
            ],
            "brokerBalancePath": "./config/brokerBalance.txt",
            "volunteerRiskToleranceRange": [0.3, 0.7],
            "dataPath": "./src/data",
            "outputPath": "./src/output"
        }
        
        os.makedirs("./config", exist_ok=True)
        os.makedirs("./data/item0", exist_ok=True)
        os.makedirs("./output", exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        with open(config['brokerBalancePath'], 'w') as f:
            f.write('1000\n2000\n3000\n')
        
        with open('./data/item0/data1.txt', 'w') as f:
            f.write("Knapsack number : 3\n")
            f.write("Slice : 0,100\n")
            f.write("Ctxs : 2\n")
            f.write("(10,100,0,1) (20,200,1,2)\n")
            f.write("1000, 2000, 3000")

    def test_initialization(self):
        self.assertEqual(len(self.simulator.brokerhubs), 1)
        self.assertEqual(len(self.simulator.volunteer_manager.volunteers), 3)

    def test_run_simulation(self):
        self.simulator.run_simulation(1)
        self.assertEqual(len(self.simulator.market_history), 1)
        
        state = self.simulator.get_market_state()
        self.assertIsNotNone(state)
        self.assertEqual(state['epoch'], 0)
        self.assertEqual(len(state['brokerhubs']), 1)
        self.assertEqual(len(state['volunteers']), 3)

if __name__ == '__main__':
    unittest.main()
import unittest
import json
import os
from src.volunteer.volunteer import Volunteer
from src.volunteer.volunteer_manager import VolunteerManager
from src.brokerhub.brokerhub import BrokerHub

class TestVolunteer(unittest.TestCase):
    def setUp(self):
        with open('config/simulation_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 创建临时的brokerBalance.txt文件用于测试
        with open(self.config['brokerBalancePath'], 'w') as f:
            f.write('1000\n2000\n3000\n4000\n5000\n')
        
        self.volunteer_manager = VolunteerManager(self.config)
        self.brokerhubs = [BrokerHub(bh_config) for bh_config in self.config['brokerhubs']]

    def tearDown(self):
        # 删除临时文件
        os.remove(self.config['brokerBalancePath'])

    def test_volunteer_initialization(self):
        volunteer = self.volunteer_manager.volunteers[0]
        self.assertEqual(volunteer.balance, 1000)
        self.assertTrue(0.3 <= volunteer.risk_tolerance <= 0.7)

    def test_volunteer_decision_making(self):
        market_data = {
            'b2e_rates': [0.05] * (len(self.volunteer_manager.volunteers) + 1)
        }
        decisions = self.volunteer_manager.make_decisions(market_data, self.brokerhubs)
        self.assertEqual(len(decisions), len(self.volunteer_manager.volunteers))
        self.assertIn('volunteer_id', decisions[0])
        self.assertIn('decision', decisions[0])

    def test_volunteer_state_update(self):
        results = [
            {'volunteer_id': 0, 'earnings': 100, 'joined_brokerhub': 'BrokerHub1'},
            {'volunteer_id': 1, 'earnings': 200},
        ]
        self.volunteer_manager.update_states(results)
        states = self.volunteer_manager.get_states()
        self.assertEqual(states[0]['balance'], 1100)
        self.assertEqual(states[0]['current_brokerhub'], 'BrokerHub1')
        self.assertEqual(states[1]['balance'], 2200)
        self.assertIsNone(states[1]['current_brokerhub'])

if __name__ == '__main__':
    unittest.main()
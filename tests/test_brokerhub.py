 
import unittest
import json
from src.brokerhub.brokerhub import BrokerHub

class TestBrokerHub(unittest.TestCase):
    def setUp(self):
        with open('config/simulation_config.json', 'r') as f:
            self.config = json.load(f)
        self.brokerhub = BrokerHub(self.config['brokerhub'])

    def test_initialization(self):
        self.assertEqual(self.brokerhub.id, "BrokerHub1")
        self.assertEqual(self.brokerhub.initial_funds, 1000000)
        self.assertEqual(self.brokerhub.current_funds, 1000000)
        self.assertEqual(self.brokerhub.tax_rate, 0.01)

    def test_make_decision(self):
        market_data = {
            'b2e_rates': [0.05] * (self.config['market']['num_users'] + 1),
            'b2e_earnings': [1000] * (self.config['market']['num_users'] + 1),
            'num_users': self.config['market']['num_users'],
            'total_investment': 1000000,
            'current_funds': [10000] * self.config['market']['num_users']
        }
        decision = self.brokerhub.make_decision(market_data)
        self.assertIn('tax_rate', decision)
        self.assertTrue(0 <= decision['tax_rate'] <= 1)

    def test_update_state(self):
        market_result = {
            'brokerhub_funds': 1100000,
            'users': [1, 2, 3, 4, 5],
            'revenue': 10000,
            'user_funds': [11000] * 5
        }
        self.brokerhub.update_state(market_result)
        self.assertEqual(self.brokerhub.current_funds, 1100000)
        self.assertEqual(len(self.brokerhub.users), 5)
        self.assertEqual(self.brokerhub.revenue_history[-1], 10000)

    def test_get_state(self):
        state = self.brokerhub.get_state()
        self.assertIn('id', state)
        self.assertIn('current_funds', state)
        self.assertIn('tax_rate', state)
        self.assertIn('num_users', state)
        self.assertIn('total_user_funds', state)

    def test_calculate_expected_revenue(self):
        total_earnings = 100000
        expected_revenue = self.brokerhub.calculate_expected_revenue(total_earnings)
        self.assertEqual(expected_revenue, total_earnings * self.brokerhub.tax_rate)

    def test_calculate_user_earnings_rate(self):
        b2e_rate = 0.05
        user_rate = self.brokerhub.calculate_user_earnings_rate(b2e_rate)
        self.assertEqual(user_rate, (1 - self.brokerhub.tax_rate) * b2e_rate)

if __name__ == '__main__':
    unittest.main()
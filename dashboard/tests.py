import pandas as pd
from django.test import TestCase

from .analytics import var_es


class RiskTests(TestCase):
    def test_var_es_historical_sign(self):
        # Mostly positive returns with one large negative shock
        rets = pd.Series([0.01] * 100 + [-0.2])
        r = var_es(rets, method="historical", confidence=0.95, horizon=1)
        self.assertIsNotNone(r.var)
        self.assertGreaterEqual(r.var, 0.0)

    def test_var_es_delta_normal_basic(self):
        rets = pd.Series([0.0, 0.01, -0.01] * 50)
        r = var_es(rets, method="delta_normal", confidence=0.95, horizon=1)
        self.assertIsNotNone(r.var)
        self.assertIsNotNone(r.es)
        self.assertGreaterEqual(r.var, 0.0)
        self.assertGreaterEqual(r.es, 0.0)

    def test_var_es_monte_carlo_runs(self):
        rets = pd.Series([0.0, 0.01, -0.01] * 50)
        r = var_es(rets, method="monte_carlo", confidence=0.95, horizon=5, mc_sims=5000)
        self.assertIsNotNone(r.var)
        self.assertGreaterEqual(r.var, 0.0)

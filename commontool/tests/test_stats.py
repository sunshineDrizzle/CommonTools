import numpy as np
import pandas as pd

from commontool import stats as ct_stats


class TestANOVA:

    def test_one_way(self):
        # ground truth
        sum_sq = [3.76634, 10.49209]
        df = [2, 27]
        F = [4.846088]
        PR = [0.01591]
        eta_sq = [0.264148]
        omega_sq = [0.204079]

        # test
        data = pd.read_csv('../data/PlantGrowth.csv')
        anova = ct_stats.ANOVA()
        aov_tabel = anova.one_way(data, 'weight', 'group')
        np.testing.assert_almost_equal(np.asarray(aov_tabel['sum_sq']), sum_sq, 6)
        assert df == aov_tabel['df'].to_list()
        np.testing.assert_almost_equal(np.asarray(aov_tabel['F'][:1]), F, 6)
        np.testing.assert_almost_equal(np.asarray(aov_tabel['PR(>F)'][:1]), PR, 6)
        np.testing.assert_almost_equal(np.asarray(aov_tabel['eta_sq'][:1]), eta_sq, 6)
        np.testing.assert_almost_equal(np.asarray(aov_tabel['omega_sq'][:1]), omega_sq, 6)

    def test_two_way(self):
        # ground truth
        sum_sq = [205.350000, 2426.434333, 108.319000, 712.106000]
        df = [1, 2, 2, 54]
        F = [15.571979, 91.999965, 4.106991]
        PR = [2.311828e-04, 4.046291e-18, 2.186027e-02]
        eta_sq = [0.059484, 0.702864, 0.031377]
        omega_sq = [0.055452, 0.692579, 0.023647]

        # test
        data = pd.read_csv('../data/ToothGrowth.csv')
        anova = ct_stats.ANOVA()
        aov_tabel = anova.two_way(data, 'len', 'supp', 'dose')
        np.testing.assert_almost_equal(np.asarray(aov_tabel['sum_sq']), sum_sq, 6)
        assert df == aov_tabel['df'].to_list()
        np.testing.assert_almost_equal(np.asarray(aov_tabel['F'][:3]), F, 6)
        np.testing.assert_almost_equal(np.asarray(aov_tabel['PR(>F)'][:3]), PR, 6)
        np.testing.assert_almost_equal(np.asarray(aov_tabel['eta_sq'][:3]), eta_sq, 6)
        np.testing.assert_almost_equal(np.asarray(aov_tabel['omega_sq'][:3]), omega_sq, 6)

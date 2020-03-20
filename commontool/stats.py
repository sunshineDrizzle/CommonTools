import numpy as np

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm, AnovaRM


class ANOVA:
    """
    Methods:
    -------
    eta_squared, omega_squared:
        As a Psychologist most of the journals we publish in requires to report effect sizes.
        Common software, such as, SPSS have eta squared as output.
        However, eta squared is an overestimation of the effect.
        To get a less biased effect size measure we can use omega squared.
        The two methods adds eta squared and omega squared to the DataFrame that contains the ANOVA table.

    References:
    ----------
    1. http://www.pybloggers.com/2016/03/three-ways-to-do-a-two-way-anova-with-python/
    2. https://pythonfordatascience.org/anova-2-way-n-way/
    3. https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
    4. http://www.pybloggers.com/2018/10/repeated-measures-anova-in-python-using-statsmodels/
    5. https://www.marsja.se/repeated-measures-anova-in-python-using-statsmodels/
    """
    def one_way(self, data, dep_var, factor):
        """
        one-way ANOVA

        Parameters:
        ----------
        data: DataFrame
            Contains at least 2 columns that are 'dependent variable' and 'factor' respectively.
        dep_var: str
            Name of the 'dependent variable' column.
        factor: str
            Name of the 'factor' column.

        Return:
        ------
        aov_table: DataFrame
            ANOVA table
        """
        formula = '{} ~ {}'.format(dep_var, factor)
        print('formula:', formula)
        model = ols(formula, data).fit()
        aov_table = anova_lm(model, typ=2)
        self.eta_squared(aov_table)
        self.omega_squared(aov_table)

        return aov_table

    def two_way(self, data, dep_var, factor1, factor2):
        """
        two-way ANOVA

        Parameters:
        ----------
        data: DataFrame
            Contains at least 3 columns that are 'dependent variable', 'factor1', and 'factor2' respectively.
        dep_var: str
            Name of the 'dependent variable' column.
        factor1: str
            Name of the 'factor1' column.
        factor2: str
            Name of the 'factor2' column.

        Return:
        ------
        aov_table: DataFrame
            ANOVA table
        """
        formula = '{0} ~ C({1}) + C({2}) + C({1}):C({2})'.format(dep_var, factor1, factor2)
        print('formula:', formula)
        model = ols(formula, data).fit()
        aov_table = anova_lm(model, typ=2)
        self.eta_squared(aov_table)
        self.omega_squared(aov_table)

        return aov_table

    def rm(self, data, dep_var, subject, within, aggregate_func=None):
        """
        Repeated Measures ANOVA

        Parameters:
        ----------
        data: DataFrame
            Contains at least 3 columns that are 'dependent variable', 'subject', and 'factor' respectively.
        dep_var: str
            Name of the 'dependent variable' column.
        subject: str
            Name of the 'subject' column. (subject identifier)
        within: a list of strings
            Names of the at least one 'factor' columns.

        Return:
        ------
        aov_table: DataFrame
            ANOVA table
        """
        aov_rm = AnovaRM(data, dep_var, subject, within, aggregate_func=aggregate_func)
        aov_table = aov_rm.fit().anova_table

        return aov_table

    @staticmethod
    def eta_squared(aov):
        aov['eta_sq'] = 'NaN'
        aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
        return aov

    @staticmethod
    def omega_squared(aov):
        mse = aov['sum_sq'][-1] / aov['df'][-1]
        aov['omega_sq'] = 'NaN'
        aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
        return aov


class EffectSize:

    def cohen_d(self, sample1, sample2):
        """
        Calculate Cohen's d.

        Parameters:
        ----------
        sample1: array-like with one dimension
        sample2: array-like with one dimension

        Return:
        ------
        d: float
            the value of the Cohen's d between sample1 and sample2

        References:
        ----------
        1. https://machinelearningmastery.com/effect-size-measures-in-python/
        2. https://www.statisticshowto.datasciencecentral.com/cohens-d/
        3. https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
        """
        # calculate the size of samples
        n1, n2 = len(sample1), len(sample2)

        # calculate the variance of the samples
        # the divisor used in the calculation is n1, n2 respectively.
        v1, v2 = np.var(sample1), np.var(sample2)

        # calculate the pooled standard deviation
        s = np.sqrt((n1 * v1 + n2 * v2) / (n1 + n2 - 2))

        # calculate the means of the samples
        u1, u2 = np.mean(sample1), np.mean(sample2)

        # calculate the effect size
        d = (u1 - u2) / s
        return d

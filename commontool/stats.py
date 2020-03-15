from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


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

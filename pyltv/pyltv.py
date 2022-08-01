# -----------------------------------------------------------------------------------------------------------------
# pyltv Module
#
# This module contains functions to generate all features of the pLTV data. It also contains a Data Manager class
# that allows data cleaning, feature generation, and plotting functionality. Forecasting & backtesting models can
# be built on top of this base class.
# -----------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from config import config
from plotly import graph_objects as go
import plotly.io as pio

# change default plotly theme
pio.templates.default = "plotly_white"


# --- DATA FUNCTIONS --- #
def borrower_retention(cohort_data):
    """
    Computes borrower retention from count_borrowers. At each time period,
    retention is simply count_borrowers divided by the original cohort size.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    borrower_retention : pandas Series
    """
    return cohort_data['count_borrowers'] / cohort_data['count_borrowers'].max()


def borrower_survival(cohort_data):
    """
    Computes borrower survival from count_borrowers. At each time period,
    survival is equal to borrower_retention divided by borrower_retention
    in the previous period. This is equivalent to count_borrowers divided
    by count_borrowers in the previous period.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    borrower_survival : pandas Series
    """
    return cohort_data['borrower_retention'] / cohort_data['borrower_retention'].shift(1)


def loans_per_borrower(cohort_data):
    """
    Average number of loans per borrower.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    loans_per_borrower : pandas Series
    """
    return cohort_data['count_loans'] / cohort_data['count_borrowers']


def loan_size(cohort_data):
    """
    Average loan size per borrower.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    loan_size : pandas Series
    """
    return cohort_data['total_amount'] / cohort_data['count_loans']


def interest_rate(cohort_data):
    """
    Average interest rate per loan.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    interest_rate : pandas Series
    """
    return cohort_data['total_interest_assessed'] / cohort_data['total_amount']


def default_rate(cohort_data, market, recovery_rates, dpd=7):
    """
    Default rate for a specified days past due (dpd). 7dpd data is assumed to be baked.
    During the data pull, only loans that were disbursed 97 days prior to the current
    date are pulled, ensuring 7dpd default rates are baked. Where 51dpd or 365dpd data
    is baked, it's taken as is. For unbaked data, 365dpd is derived from 51dpd, 51dpd
    from 30dpd, and 30dpd from 7dpd using recovery rates. The recovery rates can be
    found in the recovery_rates.csv file in data/model_dependencies/.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    market : str
        The market the data corresponds to (KE, PH, MX, etc.).

    recovery_rates : pandas DataFrame
        Contains recovery rates.

    dpd : int
        Days past due.

    Returns
    -------
    default_rate : pandas Series
    """
    if dpd == 7:
        return cohort_data['default_rate_amount_7d'].copy()

    elif dpd == 51:
        dr = cohort_data['default_rate_amount_51d'].copy()

        # determine if dpd 51 dr is baked. dpd 51 dr requires 2 additional months to bake
        final_month = len(cohort_data)

        # iterate through each month of data
        for month in cohort_data.index:

            # if the current month is within 5 months of the last month of data (not 51 dpd baked)
            if month > final_month - 3:

                # derive dr based on recovery rate for the given month
                recovery_rate_51 = float(recovery_rates[recovery_rates.month == month].loc[market, 'recovery_30-51'])

                derived_51dpd = cohort_data['default_rate_amount_30d'] * (1-recovery_rate_51)

                dr.loc[month] = derived_51dpd.loc[month]

                # if current month is within 4 months of the last month of data (not 30 dpd baked)
                if month > final_month - 2:
                    # derive dr based on recovery rate for the given month
                    recovery_rate_7_30 = float(
                        recovery_rates[recovery_rates.month == month].loc[market, 'recovery_7-30'])
                    recovery_rate_51 = float(
                        recovery_rates[recovery_rates.month == month].loc[market, 'recovery_30-51'])

                    derived_30dpd = cohort_data['default_rate_amount_7d'] * (1 - recovery_rate_7_30)
                    derived_51dpd = derived_30dpd * (1 - recovery_rate_51)

                    dr.loc[month] = derived_51dpd.loc[month]

        return dr

    elif dpd == 365:

        dr = cohort_data['default_rate_amount_365d'].copy()

        # determine if 365dpd dr is baked
        final_month = len(cohort_data)

        for month in cohort_data.index:

            # if the current month is within 13 months of the last month of data (not 365 dpd baked)
            if month > final_month - 13:

                recovery_rate_365 = float(recovery_rates[recovery_rates.month == month].loc[market, 'recovery_51_'])

                derived_365dpd = cohort_data['default_rate_51dpd'] * (1-recovery_rate_365)

                dr.loc[month] = derived_365dpd.loc[month]

                # if the current month is within 5 months of the last month of data (not 51 dpd baked)
                if month > final_month - 3:

                    # derive dr based on recovery rate for the given month
                    recovery_rate_51 = float(
                        recovery_rates[recovery_rates.month == month].loc[market, 'recovery_30-51'])
                    recovery_rate_365 = float(
                        recovery_rates[recovery_rates.month == month].loc[market, 'recovery_51_'])

                    derived_51dpd = cohort_data['default_rate_amount_30d'] * (1 - recovery_rate_51)
                    derived_365dpd = derived_51dpd * (1 - recovery_rate_365)

                    dr.loc[month] = derived_365dpd.loc[month]

                    # if current month is within 4 months of the last month of data (not 30 dpd baked)
                    if month > final_month - 2:
                        # derive dr based on recovery rate for the given month
                        recovery_rate_7_30 = float(
                            recovery_rates[recovery_rates.month == month].loc[market, 'recovery_7-30'])
                        recovery_rate_51 = float(
                            recovery_rates[recovery_rates.month == month].loc[market, 'recovery_30-51'])
                        recovery_rate_365 = float(
                            recovery_rates[recovery_rates.month == month].loc[market, 'recovery_51_'])

                        derived_30dpd = cohort_data['default_rate_amount_7d'] * (1 - recovery_rate_7_30)
                        derived_51dpd = derived_30dpd * (1 - recovery_rate_51)
                        derived_365dpd = derived_51dpd * (1 - recovery_rate_365)

                        dr.loc[month] = derived_365dpd.loc[month]

        return dr


def loans_per_original(cohort_data):
    """
    Average number of loans per original.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    loans_per_original : pandas Series
    """
    return cohort_data['count_loans'] / cohort_data['count_borrowers'].max()


def origination_per_original(cohort_data):
    """
    Average origination per original.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    origination_per_original : pandas Series
    """
    return cohort_data['loans_per_original'] * cohort_data['loan_size']


def revenue_per_original(cohort_data):
    """
    Average revenue per original.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    revenue_per_original : pandas Series
    """
    interest_revenue = cohort_data['origination_per_original'] * cohort_data['interest_rate']
    late_fee_revenue = (cohort_data['origination_per_original'] + interest_revenue) * \
        cohort_data['default_rate_7dpd'] * config['late_fee']

    return interest_revenue + late_fee_revenue


def credit_margin(cohort_data):
    """
    Average credit margin per original.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    CrM_per_original : pandas Series
    """
    bad_debt = (cohort_data['origination_per_original'] + cohort_data['revenue_per_original']) * \
        cohort_data['default_rate_365dpd']

    return cohort_data['revenue_per_original'] - bad_debt


def opex_per_original(cohort_data, market):
    """
    Average opex per original. OPEX consists of the cost per loan and cost of capital.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    market : str
        The market the data corresponds to (KE, PH, MX, etc.).

    Returns
    -------
    opex_per_original : pandas Series
    """
    opex_cost_per_loan = float(config['opex_cpl'][market])
    cost_of_capital = float(config['opex_coc'][market]) / 12

    total_opex = opex_cost_per_loan * cohort_data['loans_per_original'] + \
        (cost_of_capital * cohort_data['origination_per_original'])

    return total_opex


def opex_coc_per_original(cohort_data, market):
    """
    Average cost of capital (coc) per original.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    market : str
        The market the data corresponds to (KE, PH, MX, etc.).

    Returns
    -------
    opex_coc_per_original : pandas Series
    """
    cost_of_capital = float(config['opex_coc'][market]) / 12

    return cost_of_capital * cohort_data['origination_per_original']


def opex_cpl_per_original(cohort_data, market):
    """
    Average cost per loan (cpl) per original.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    market : str
        The market the data corresponds to (KE, PH, MX, etc.).

    Returns
    -------
    opex_cpl_per_original : pandas Series
    """
    opex_cost_per_loan = float(config['opex_cpl'][market])

    return opex_cost_per_loan * cohort_data['loans_per_original']


def ltv_per_original(cohort_data):
    """
    Lifetime value (LTV) per original. LTV is equal to credit margin minus OPEX.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    ltv_per_original : pandas Series
    """
    return cohort_data['crm_per_original'] - cohort_data['opex_per_original']


def dcf_ltv_per_original(cohort_data):
    """
    Discounted Lifetime value (LTV) per original. The LTV is discounted based on
    a discounted rate of return specified in the config file.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    ltv_per_original : pandas Series
    """
    return cohort_data['ltv_per_original']/(1+(config['dcf']/12)*cohort_data['ltv_per_original'].index)


def credit_margin_percent(cohort_data):
    """
    Credit margin percent (CM%) per original. LTV as a proportion of
    revenue_per_original.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    crm%_per_original : pandas Series
    """
    return cohort_data['ltv_per_original'] / cohort_data['revenue_per_original']


# --- DATA MANAGER CLASS --- #
class DataManager:
    """
    Contains functionality to clean data, generate features, and visualize LTV data.

    Parameters
    ----------
    data : pandas DataFrame
        Raw data pulled from Looker. E.g. https://inventure.looker.com/looks/7451
    market : str
        The market the data corresponds to (KE, PH, MX, etc.).
    to_usd : bool
        If True, convert fields in local currency to USD. If False, leave fields as
        local currency.
    bake_duration : int
        Number of months to consider data fully baked. The last bake_duration number
        of months is removed from the data during cleaning.

    Methods
    -------
    load_dependent_data()
        Load data that the models depend on such as recovery rates, opex, and cost of
        capital.

    clean_data()
        Performs data cleaning steps required before modeling.
    """

    def __init__(self, data, market, to_usd=True, bake_duration=4):
        """
        Parameters
        ----------
        data : pandas DataFrame
            Raw data pulled from Looker. E.g. https://inventure.looker.com/looks/7451
        market : str
            The market the data corresponds to (KE, PH, MX, etc.).
        to_usd : bool
            If True, convert fields in local currency to USD. If False, leave fields as
            local currency.
        bake_duration : int
            Number of months to consider data fully baked. The last bake_duration number
            of months is removed from the data during cleaning.
        """
        self.data = data
        self.market = market
        self.to_usd = to_usd
        self.bake_duration = bake_duration
        self.config = config

        # placeholder attributes for forecast & backtest data
        self.forecast = None
        self.backtest = None
        self.backtest_report = None

        # load recovery rates
        self.recovery_rates = pd.read_csv('data/model_dependencies/recovery_rates.csv').set_index('market')

        # clean data and generate features
        self.clean_data()
        self.data = self.generate_features(self.data)

    def clean_data(self):
        """
        Performs various data cleaning steps to prepare data for model.
        """
        # sort by months since first disbursement
        self.data = self.data.sort_values(['first_loan_local_disbursement_month',
                                           'months_since_first_loan_disbursed'])

        # add cohort label column
        self.data['cohort'] = self.data['first_loan_local_disbursement_month']

        # convert cash cols from local currency to USD
        cash_cols = ['total_amount', 'total_interest_assessed', 'total_rollover_charged', 'total_rollover_reversed']
        for col in cash_cols:
            if self.to_usd:
                self.data[col] /= config['forex'][self.market]

        cohort_data = []
        for c in self.data.cohort.unique():
            c_data = self.data[self.data.cohort == c].copy()

            # start indexing from 1
            c_data.index = np.arange(1, len(c_data)+1)

            # remove last "bake_duration" months of data (ensures default_rate_7dpd data is baked)
            c_data = c_data.iloc[:-self.bake_duration]

            cohort_data.append(c_data)

        self.data = pd.concat(cohort_data, axis=0)

    def summarize_data(self):
        """
        Prints out some general information about the data
        """
        # min and max dates that the data spans
        min_date = str(pd.to_datetime(self.data.cohort).min())[:7]
        max_date = str(pd.to_datetime(self.data.cohort).max())[:7]

        # number of cohorts
        n_cohorts = self.data.cohort.nunique()

        print(f'Cleaned data spans {min_date} to {max_date}')
        print(f'Total # of cohorts: {n_cohorts}')

    def generate_features(self, data):
        """
        Generate all features required for pLTV model.

        Parameters
        ----------
        data : pandas DataFrame
            Data to generate features for. Can be raw or cleaned data.
        """
        cohort_dfs = []

        # iterate through all cohorts individually
        for c in data.cohort.unique():
            # create copy of current cohort dataframe
            cohort_data = data[data.cohort == c].copy()

            # generate features
            cohort_data['borrower_retention'] = borrower_retention(cohort_data)
            cohort_data['borrower_survival'] = borrower_survival(cohort_data)
            cohort_data['loans_per_borrower'] = loans_per_borrower(cohort_data)
            cohort_data['loan_size'] = loan_size(cohort_data)
            cohort_data['interest_rate'] = interest_rate(cohort_data)
            cohort_data['default_rate_7dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=7)
            cohort_data['default_rate_51dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=51)
            cohort_data['default_rate_365dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=365)
            cohort_data['loans_per_original'] = loans_per_original(cohort_data)
            cohort_data['cumulative_loans_per_original'] = cohort_data['loans_per_original'].cumsum()
            cohort_data['origination_per_original'] = origination_per_original(cohort_data)
            cohort_data['cumulative_origination_per_original'] = cohort_data['origination_per_original'].cumsum()
            cohort_data['revenue_per_original'] = revenue_per_original(cohort_data)
            cohort_data['cumulative_revenue_per_original'] = cohort_data['revenue_per_original'].cumsum()
            cohort_data['crm_per_original'] = credit_margin(cohort_data)
            cohort_data['cumulative_crm_per_original'] = cohort_data['crm_per_original'].cumsum()
            cohort_data['opex_per_original'] = opex_per_original(cohort_data, self.market)
            cohort_data['cumulative_opex_per_original'] = cohort_data['opex_per_original'].cumsum()
            cohort_data['opex_coc_per_original'] = opex_coc_per_original(cohort_data, self.market)
            cohort_data['cumulative_opex_coc_per_original'] = cohort_data['opex_coc_per_original'].cumsum()
            cohort_data['opex_cpl_per_original'] = opex_cpl_per_original(cohort_data, self.market)
            cohort_data['cumulative_opex_cpl_per_original'] = cohort_data['opex_cpl_per_original'].cumsum()
            cohort_data['ltv_per_original'] = ltv_per_original(cohort_data)
            cohort_data['cumulative_ltv_per_original'] = cohort_data['ltv_per_original'].cumsum()
            cohort_data['dcf_ltv_per_original'] = dcf_ltv_per_original(cohort_data)
            cohort_data['cumulative_dcf_ltv_per_original'] = cohort_data['dcf_ltv_per_original'].cumsum()
            cohort_data['crm_perc_per_original'] = credit_margin_percent(cohort_data)

            cohort_dfs.append(cohort_data)

        return pd.concat(cohort_dfs)

    def plot_cohorts(self, field, dataset='data', show=False):
        """
        Generate various plots for a specified field in the data.

        Parameters
        ----------
        field : str
            Field name to plot. Can be any column in the dataset.
        dataset : str
            Specifies which dataset to plot from. Options are:
                - data
                - forecast
                - backtest
                - backtest_report
            If data hasn't been forecast or backtested yet, run forecast_data or
            backtest_data commands first to generate those datasets.
        show : bool
            If True, prints out the figure. If False, returns the figure object without
            rendering it.
        """
        # print message about what datasets are available to plot
        if dataset not in ['data', 'forecast', 'backtest', 'backtest_report']:
            print('Dataset must be one of the following: ')
            print('data, forecast, backtest, backtest_report')

        else:
            # print message if data has not been forecast or backtested yet
            if (dataset in ['forecast', 'backtest', 'backtest_report']) and \
                    self.__getattribute__(dataset) is None:
                print("Data has not been forecast or backtested yet.")
                print('Run forecast_data() or backtest_data() methods first.')

            else:
                # check that specified field exists in dataset
                if field not in self.__getattribute__(dataset).columns:
                    print('Not a valid field name! Available fields:')
                    print('')
                    print(self.__getattribute__(dataset).columns)

                # generate plots according to dataset
                else:
                    if dataset in ['data', 'forecast', 'backtest']:
                        curves = []

                        if dataset == 'forecast':
                            for cohort in self.forecast.cohort.unique():
                                c_data = self.forecast[self.forecast.cohort == cohort]
                                for dtype in c_data.data_type.unique():
                                    output = c_data[c_data.data_type == dtype][field]

                                    output.name = cohort + '-' + dtype

                                    curves.append(output)

                        elif dataset == 'backtest':
                            for cohort in self.backtest.cohort.unique():
                                c_data = self.backtest[self.backtest.cohort == cohort]

                                # append raw data
                                output = self.data[self.data.cohort == cohort][field]
                                output.name = cohort + '-actual'

                                curves.append(output)

                                # append forecast
                                output = c_data[c_data.data_type == 'forecast'][field]
                                output.name = cohort + '-forecast'

                                curves.append(output)

                        elif dataset == 'data':
                            for cohort in self.data.cohort.unique():
                                output = self.data[self.data.cohort == cohort][field]

                                output.name = cohort

                                curves.append(output)

                        traces = []

                        for cohort in curves:
                            if 'forecast' in cohort.name:
                                traces.append(go.Scatter(name=cohort.name, x=cohort.index, y=cohort, mode='lines',
                                                         line=dict(width=3, dash='dash')))
                            else:
                                if cohort.notnull().any():
                                    traces.append(go.Scatter(name=cohort.name, x=cohort.index, y=cohort,
                                                             mode='markers+lines', line=dict(width=2)))

                        fig = go.Figure(traces)

                        if 'default' in field:
                            y_format = dict(title=field, tickformat=".2%")
                        else:
                            y_format = dict(title=field)

                        fig.update_layout(title=f'{field} - {dataset.upper()}',
                                          xaxis=dict(title='Month Since First Disbursement'),
                                          yaxis=y_format)

                        if show:
                            fig.show()

                        return fig

                    elif dataset == 'backtest_report':
                        print(dataset)
                        curves = []
                        for cohort in self.backtest_report.cohort.unique():
                            c_data = self.backtest_report[self.backtest_report.cohort == cohort]
                            output = c_data[field]

                            output.name = cohort

                            curves.append(output)

                        traces = []
                        for cohort in curves:
                            traces.append(go.Bar(name=cohort.name, x=cohort.index, y=cohort))

                        # add a line showing the overall mean
                        mean_df = pd.DataFrame(float(self.backtest_report[field].mean()),
                                               index=self.backtest_report.index, columns=['mean'])
                        traces.append(go.Scatter(name='mean', x=mean_df.index, y=mean_df['mean'], mode='lines'))

                        metric = field.split('-')[1].upper()
                        fig = go.Figure(traces)

                        if 'mpe' in field or 'mape' in field or 'default' in field or 'retention' in field:
                            y_format = dict(title=field, tickformat=".2%")
                        else:
                            y_format = dict(title=field)

                        fig.update_layout(title=f'{metric} Backtest',
                                          xaxis=dict(title='Month Since First Disbursement'),
                                          yaxis=y_format)

                        if show:
                            fig.show()

                        return fig


class PyLTV:
    """
    Contains functionality to clean data, generate features, and visualize LTV data.

    Parameters
    ----------
    data : pandas DataFrame
        Raw data pulled from Looker. E.g. https://inventure.looker.com/looks/7451
    market : str
        The market the data corresponds to (KE, PH, MX, etc.).
    to_usd : bool
        If True, convert fields in local currency to USD. If False, leave fields as
        local currency.
    bake_duration : int
        Number of months to consider data fully baked. The last bake_duration number
        of months is removed from the data during cleaning.

    Methods
    -------
    load_dependent_data()
        Load data that the models depend on such as recovery rates, opex, and cost of
        capital.

    clean_data()
        Performs data cleaning steps required before modeling.
    """

    def __init__(self, data, market, segment_by=None, to_usd=True, bake_duration=4):
        """
        Parameters
        ----------
        data : pandas DataFrame
            Raw data pulled from Looker. E.g. https://inventure.looker.com/looks/7451
        market : str
            The market the data corresponds to (KE, PH, MX, etc.).
        to_usd : bool
            If True, convert fields in local currency to USD. If False, leave fields as
            local currency.
        bake_duration : int
            Number of months to consider data fully baked. The last bake_duration number
            of months is removed from the data during cleaning.
        """
        # set class attributes from args
        self.data = data
        self.market = market
        self.segment_by = segment_by
        self.to_usd = to_usd
        self.bake_duration = bake_duration

        # set model config attribute
        self.config = config

        # placeholder attributes for forecast & backtest data
        self.forecast = None
        self.backtest = None
        self.backtest_report = None

        # load recovery rates
        self.recovery_rates = pd.read_csv('data/model_dependencies/recovery_rates.csv').set_index('market')

        # clean data and generate features
        self.clean_data()
        self.data = self.generate_features(self.data)

    def clean_data(self):
        """
        Performs various data cleaning steps to prepare data for model.
        """
        # sort by months since first disbursement
        self.data = self.data.sort_values(['first_loan_local_disbursement_month',
                                           'months_since_first_loan_disbursed'])

        # add cohort label column
        self.data['cohort'] = self.data['first_loan_local_disbursement_month']

        # convert cash cols from local currency to USD
        cash_cols = ['total_amount', 'total_interest_assessed', 'total_rollover_charged', 'total_rollover_reversed']
        for col in cash_cols:
            if self.to_usd:
                self.data[col] /= config['forex'][self.market]

        cohort_data = []
        for c in self.data.cohort.unique():
            c_data = self.data[self.data.cohort == c].copy()

            # start indexing from 1
            c_data.index = np.arange(1, len(c_data)+1)

            # remove last "bake_duration" months of data (ensures default_rate_7dpd data is baked)
            c_data = c_data.iloc[:-self.bake_duration]

            cohort_data.append(c_data)

        self.data = pd.concat(cohort_data, axis=0)

    def summarize_data(self):
        """
        Prints out some general information about the data
        """
        # min and max dates that the data spans
        min_date = str(pd.to_datetime(self.data.cohort).min())[:7]
        max_date = str(pd.to_datetime(self.data.cohort).max())[:7]

        # number of cohorts
        n_cohorts = self.data.cohort.nunique()

        print(f'Cleaned data spans {min_date} to {max_date}')
        print(f'Total # of cohorts: {n_cohorts}')

    def generate_features(self, data):
        """
        Generate all features required for pLTV model.

        Parameters
        ----------
        data : pandas DataFrame
            Data to generate features for. Can be raw or cleaned data.
        """
        cohort_dfs = []

        # iterate through all cohorts individually
        for c in data.cohort.unique():
            # create copy of current cohort dataframe
            cohort_data = data[data.cohort == c].copy()

            # generate features
            cohort_data['borrower_retention'] = borrower_retention(cohort_data)
            cohort_data['borrower_survival'] = borrower_survival(cohort_data)
            cohort_data['loans_per_borrower'] = loans_per_borrower(cohort_data)
            cohort_data['loan_size'] = loan_size(cohort_data)
            cohort_data['interest_rate'] = interest_rate(cohort_data)
            cohort_data['default_rate_7dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=7)
            cohort_data['default_rate_51dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=51)
            cohort_data['default_rate_365dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=365)
            cohort_data['loans_per_original'] = loans_per_original(cohort_data)
            cohort_data['cumulative_loans_per_original'] = cohort_data['loans_per_original'].cumsum()
            cohort_data['origination_per_original'] = origination_per_original(cohort_data)
            cohort_data['cumulative_origination_per_original'] = cohort_data['origination_per_original'].cumsum()
            cohort_data['revenue_per_original'] = revenue_per_original(cohort_data)
            cohort_data['cumulative_revenue_per_original'] = cohort_data['revenue_per_original'].cumsum()
            cohort_data['crm_per_original'] = credit_margin(cohort_data)
            cohort_data['cumulative_crm_per_original'] = cohort_data['crm_per_original'].cumsum()
            cohort_data['opex_per_original'] = opex_per_original(cohort_data, self.market)
            cohort_data['cumulative_opex_per_original'] = cohort_data['opex_per_original'].cumsum()
            cohort_data['opex_coc_per_original'] = opex_coc_per_original(cohort_data, self.market)
            cohort_data['cumulative_opex_coc_per_original'] = cohort_data['opex_coc_per_original'].cumsum()
            cohort_data['opex_cpl_per_original'] = opex_cpl_per_original(cohort_data, self.market)
            cohort_data['cumulative_opex_cpl_per_original'] = cohort_data['opex_cpl_per_original'].cumsum()
            cohort_data['ltv_per_original'] = ltv_per_original(cohort_data)
            cohort_data['cumulative_ltv_per_original'] = cohort_data['ltv_per_original'].cumsum()
            cohort_data['dcf_ltv_per_original'] = dcf_ltv_per_original(cohort_data)
            cohort_data['cumulative_dcf_ltv_per_original'] = cohort_data['dcf_ltv_per_original'].cumsum()
            cohort_data['crm_perc_per_original'] = credit_margin_percent(cohort_data)

            cohort_dfs.append(cohort_data)

        return pd.concat(cohort_dfs)

    def plot_cohorts(self, field, dataset='data', show=False):
        """
        Generate various plots for a specified field in the data.

        Parameters
        ----------
        field : str
            Field name to plot. Can be any column in the dataset.
        dataset : str
            Specifies which dataset to plot from. Options are:
                - data
                - forecast
                - backtest
                - backtest_report
            If data hasn't been forecast or backtested yet, run forecast_data or
            backtest_data commands first to generate those datasets.
        show : bool
            If True, prints out the figure. If False, returns the figure object without
            rendering it.
        """
        # print message about what datasets are available to plot
        if dataset not in ['data', 'forecast', 'backtest', 'backtest_report']:
            print('Dataset must be one of the following: ')
            print('data, forecast, backtest, backtest_report')

        else:
            # print message if data has not been forecast or backtested yet
            if (dataset in ['forecast', 'backtest', 'backtest_report']) and \
                    self.__getattribute__(dataset) is None:
                print("Data has not been forecast or backtested yet.")
                print('Run forecast_data() or backtest_data() methods first.')

            else:
                # check that specified field exists in dataset
                if field not in self.__getattribute__(dataset).columns:
                    print('Not a valid field name! Available fields:')
                    print('')
                    print(self.__getattribute__(dataset).columns)

                # generate plots according to dataset
                else:
                    if dataset in ['data', 'forecast', 'backtest']:
                        curves = []

                        if dataset == 'forecast':
                            for cohort in self.forecast.cohort.unique():
                                c_data = self.forecast[self.forecast.cohort == cohort]
                                for dtype in c_data.data_type.unique():
                                    output = c_data[c_data.data_type == dtype][field]

                                    output.name = cohort + '-' + dtype

                                    curves.append(output)

                        elif dataset == 'backtest':
                            for cohort in self.backtest.cohort.unique():
                                c_data = self.backtest[self.backtest.cohort == cohort]

                                # append raw data
                                output = self.data[self.data.cohort == cohort][field]
                                output.name = cohort + '-actual'

                                curves.append(output)

                                # append forecast
                                output = c_data[c_data.data_type == 'forecast'][field]
                                output.name = cohort + '-forecast'

                                curves.append(output)

                        elif dataset == 'data':
                            for cohort in self.data.cohort.unique():
                                output = self.data[self.data.cohort == cohort][field]

                                output.name = cohort

                                curves.append(output)

                        traces = []

                        for cohort in curves:
                            if 'forecast' in cohort.name:
                                traces.append(go.Scatter(name=cohort.name, x=cohort.index, y=cohort, mode='lines',
                                                         line=dict(width=3, dash='dash')))
                            else:
                                if cohort.notnull().any():
                                    traces.append(go.Scatter(name=cohort.name, x=cohort.index, y=cohort,
                                                             mode='markers+lines', line=dict(width=2)))

                        fig = go.Figure(traces)

                        if 'default' in field:
                            y_format = dict(title=field, tickformat=".2%")
                        else:
                            y_format = dict(title=field)

                        fig.update_layout(title=f'{field} - {dataset.upper()}',
                                          xaxis=dict(title='Month Since First Disbursement'),
                                          yaxis=y_format)

                        if show:
                            fig.show()

                        return fig

                    elif dataset == 'backtest_report':
                        print(dataset)
                        curves = []
                        for cohort in self.backtest_report.cohort.unique():
                            c_data = self.backtest_report[self.backtest_report.cohort == cohort]
                            output = c_data[field]

                            output.name = cohort

                            curves.append(output)

                        traces = []
                        for cohort in curves:
                            traces.append(go.Bar(name=cohort.name, x=cohort.index, y=cohort))

                        # add a line showing the overall mean
                        mean_df = pd.DataFrame(float(self.backtest_report[field].mean()),
                                               index=self.backtest_report.index, columns=['mean'])
                        traces.append(go.Scatter(name='mean', x=mean_df.index, y=mean_df['mean'], mode='lines'))

                        metric = field.split('-')[1].upper()
                        fig = go.Figure(traces)

                        if 'mpe' in field or 'mape' in field or 'default' in field or 'retention' in field:
                            y_format = dict(title=field, tickformat=".2%")
                        else:
                            y_format = dict(title=field)

                        fig.update_layout(title=f'{metric} Backtest',
                                          xaxis=dict(title='Month Since First Disbursement'),
                                          yaxis=y_format)

                        if show:
                            fig.show()

                        return fig

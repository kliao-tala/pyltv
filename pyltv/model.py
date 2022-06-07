# -----------------------------------------------------------------------------------------------------------------
# Data Model
#
# This library defines the base data model class from which all pyLTV models can be built from.
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
    Computes borrower retention from Count Borrowers. At each time period,
    retention is simply Count Borrowers divided by the original cohort size.

    Parameters
    ----------
    cohort_data : pandas DataFrame
        Data for a single cohort by months since first loan disbursement.

    Returns
    -------
    borrower_retention : pandas Series
    """
    return cohort_data['Count Borrowers'] / cohort_data['Count Borrowers'].max()


def borrower_survival(cohort_data):
    """
    Computes borrower survival from Count Borrowers. At each time period,
    survival is equal to borrower_retention divided by borrower_retention
    in the previous period. This is equivalent to Count Borrowers divided
    by Count Borrowers in the previous period.

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
    return cohort_data['Count Loans'] / cohort_data['Count Borrowers']


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
    return cohort_data['Total Amount'] / cohort_data['Count Loans']


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
    return cohort_data['Total Interest Assessed'] / cohort_data['Total Amount']


def default_rate(cohort_data, market, recovery_rates, dpd=7):
    """
    Default rate for a specified days past due (dpd). The 7dpd default rate is
    taken as is from the raw LTV data and null values are filled in with 0.

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
        return cohort_data['Default Rate Amount 7D']

    elif dpd == 51:
        dr = cohort_data['Default Rate Amount 51D'].copy()

        recovery_rate_30 = float(recovery_rates.loc[market, 'recovery_7-30'])
        recovery_rate_51 = float(recovery_rates.loc[market, 'recovery_30-51'])
        derived_30dpd = cohort_data['Default Rate Amount 7D']*(1-recovery_rate_30)
        derived_51dpd = derived_30dpd * (1-recovery_rate_51)

        return dr.fillna(derived_51dpd)

    elif dpd == 365:
        # get actual cohort_data if it exists
        dr = np.nan * cohort_data['Default Rate Amount 51D'].copy()

        recovery_rate_365 = float(recovery_rates.loc[market, 'recovery_51_'])

        derived_365dpd = cohort_data['default_rate_51dpd'] * (1-recovery_rate_365)

        return dr.fillna(derived_365dpd)


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
    return cohort_data['Count Loans'] / cohort_data['Count Borrowers'].max()


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
    cost_of_capital = float(config['coc'][market]) / 12

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
    cost_of_capital = float(config['coc'][market]) / 12

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
    return cohort_data['cm$_per_original'] - cohort_data['opex_per_original']


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
    """Data Manager Class

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
        Performs all data cleaning steps required before modeling.
    """

    def __init__(self, data, market, to_usd=True, bake_duration=None):
        """
        Sets model attributes, loads additional data required for models (inputs &
        ltv_expected), and cleans data.

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

        # load data that the models depend on
        self.recovery_rates = pd.read_csv('data/model_dependencies/recovery_rates.csv').set_index('market')

        self.clean_data()
        self.raw = self.generate_features(self.raw)
        self.data = self.generate_features(self.data)

        # print the date range that the data spans
        min_date = str(pd.to_datetime(self.data.cohort).min())[:7]
        max_date = str(pd.to_datetime(self.data.cohort).max())[:7]
        n_months = self.data.cohort.nunique()
        print(f'Clean data spans {min_date} to {max_date}')
        print(f'Total # of cohorts: {n_months}')
        print('')

    def clean_data(self):
        """
        Performs various data clean up steps to prepare data for model.
        """
        # remove any leading or trailing spaces in col names
        self.data.columns = [c.strip() for c in self.data.columns]

        # rename any columns with mismatching names
        self.data = self.data.rename(columns={'First Loan Disbursement Month': 'First Loan Local Disbursement Month',
                                              'Total Principal Amount': 'Total Amount',
                                              'Default Rate Amount 7d': 'Default Rate Amount 7D',
                                              'Default Rate Amount 51d': 'Default Rate Amount 51D',
                                              'Default Rate Amount 30d': 'Default Rate Amount 30D'})

        # add a leading 0 to the month if there isn't one already to match rest of the data
        for date in self.data['First Loan Local Disbursement Month'].unique():
            if len(date) < 7:
                year, month = date.split('-')
                month = '0' + month
                self.data = self.data.replace({date: year + '-' + month})

        # convert cols to appropriate datatypes
        int_cols = ['Count Borrowers',
                    'Count Loans',
                    'Total Amount',
                    'Total Interest Assessed',
                    'Total Rollover Charged',
                    'Total Rollover Reversed']
        for col in int_cols:
            try:
                self.data[col] = pd.to_numeric(self.data[col].str.replace(',', ''))
            except AttributeError:
                self.data[col] = pd.to_numeric(self.data[col])

        # convert month since disbursement to int
        if self.data['Months Since First Loan Disbursed'].dtype == 'O':
            self.data['Months Since First Loan Disbursed'] = self.data['Months Since First Loan Disbursed'].apply(
                lambda x: int(x.split(' ')[0]))

        # drop any negative months since first loan
        self.data = self.data[~self.data['Months Since First Loan Disbursed'] < 0]

        # sort by months since first disbursement
        self.data = self.data.sort_values(['First Loan Local Disbursement Month',
                                           'Months Since First Loan Disbursed'])

        # remove all columns calculated through looker
        self.data = self.data.loc[:, :"Default Rate Amount 51D"]

        # add more convenient cohort label column
        self.data['cohort'] = self.data['First Loan Local Disbursement Month']

        # convert local currency to USD
        if self.to_usd:
            self.data['Total Amount'] /= config['forex'][self.market]
            self.data['Total Interest Assessed'] /= config['forex'][self.market]
            self.data['Total Rollover Charged'] /= config['forex'][self.market]
            self.data['Total Rollover Reversed'] /= config['forex'][self.market]

        # save raw data df before removing data for forecast
        self.raw = self.data.copy()

        # remove the last month of the raw data (incomplete month)
        cohort_data = []
        for c in self.raw.cohort.unique():
            c_data = self.raw[self.raw.cohort == c]
            c_data.index = np.arange(1, len(c_data) + 1)

            cohort_data.append(c_data.iloc[:-1])

        self.raw = pd.concat(cohort_data, axis=0)

        cohort_data = []
        for c in self.data.cohort.unique():
            c_data = self.data[self.data.cohort == c]
            c_data.index = np.arange(1, len(c_data)+1)

            # remove the last "bake_duration" months of data for each cohort
            # this is to ensure default_rate_7dpd data is fully baked
            if self.bake_duration is not None:
                print('removing 1 month')
                cohort_data.append(c_data.iloc[:-self.bake_duration])
            else:
                cohort_data.append(c_data)

        self.data = pd.concat(cohort_data, axis=0)

    # --- CONVENIENCE FUNCTIONS --- #
    def generate_features(self, data):
        """
        Generate all features required for pLTV model.
        """
        cohort_dfs = []
        for c in data.cohort.unique():
            cohort_data = data[data.cohort == c].copy()

            cohort_data['borrower_retention'] = borrower_retention(cohort_data)
            cohort_data['borrower_survival'] = borrower_survival(cohort_data)
            cohort_data['loans_per_borrower'] = loans_per_borrower(cohort_data)
            cohort_data['loan_size'] = loan_size(cohort_data)
            cohort_data['interest_rate'] = interest_rate(cohort_data)
            cohort_data['default_rate_7dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=7)
            cohort_data['default_rate_51dpd'] = default_rate(cohort_data, self.market, self.recovery_rates, dpd=51)
            cohort_data['default_rate_365dpd'] = default_rate(cohort_data,self.market, self.recovery_rates, dpd=365)
            cohort_data['loans_per_original'] = loans_per_original(cohort_data)
            cohort_data['cumulative_loans_per_original'] = cohort_data['loans_per_original'].cumsum()
            cohort_data['origination_per_original'] = origination_per_original(cohort_data)
            cohort_data['cumulative_origination_per_original'] = cohort_data['origination_per_original'].cumsum()
            cohort_data['revenue_per_original'] = revenue_per_original(cohort_data)
            cohort_data['cumulative_revenue_per_original'] = cohort_data['revenue_per_original'].cumsum()
            cohort_data['cm$_per_original'] = credit_margin(cohort_data)
            cohort_data['cumulative_cm$_per_original'] = cohort_data['cm$_per_original'].cumsum()
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
            cohort_data['cm%_per_original'] = credit_margin_percent(cohort_data)

            cohort_dfs.append(cohort_data)

        return pd.concat(cohort_dfs)

    def plot_cohorts(self, param, data='raw', show=False):
        """
        Generate scatter plot for a specific paramter.

        Parameters
        ----------

        """

        if data == 'clean' or data == 'raw' or data == 'forecast' or data == 'backtest':
            curves = []

            if data == 'forecast':
                for cohort in self.forecast.cohort.unique():
                    c_data = self.forecast[self.forecast.cohort == cohort]
                    for dtype in c_data.data_type.unique():
                        output = c_data[c_data.data_type == dtype][param]

                        output.name = cohort + '-' + dtype

                        curves.append(output)

            elif data == 'backtest':
                for cohort in self.backtest.cohort.unique():
                    c_data = self.backtest[self.backtest.cohort == cohort]

                    # append raw data
                    output = self.data[self.data.cohort == cohort][param]
                    output.name = cohort + '-actual'

                    curves.append(output)

                    # append forecast
                    output = c_data[c_data.data_type == 'forecast'][param]
                    output.name = cohort + '-forecast'

                    curves.append(output)

            elif data == 'clean':
                for cohort in self.data.cohort.unique():
                    output = self.data[self.data.cohort == cohort][param]

                    output.name = cohort

                    curves.append(output)

            elif data == 'raw':
                for cohort in self.raw.cohort.unique():
                    output = self.raw[self.raw.cohort == cohort][param]

                    output.name = cohort

                    curves.append(output)

            traces = []

            for cohort in curves:
                if 'forecast' in cohort.name:
                    traces.append(go.Scatter(name=cohort.name, x=cohort.index, y=cohort, mode='lines',
                                             line=dict(width=3, dash='dash')))
                else:
                    if cohort.notnull().any():
                        traces.append(go.Scatter(name=cohort.name, x=cohort.index, y=cohort, mode='markers+lines',
                                                 line=dict(width=2)))

            fig = go.Figure(traces)
            fig.update_layout(title=f'{param} - {data.upper()}',
                              xaxis=dict(title='Month Since First Disbursement'),
                              yaxis=dict(title=param))

            if show:
                fig.show()

            return fig

        elif data == 'backtest_report':
            curves = []
            for cohort in self.backtest_report.cohort.unique():
                c_data = self.backtest_report[self.backtest_report.cohort == cohort]
                output = c_data[param]

                output.name = cohort

                curves.append(output)

            traces = []
            for cohort in curves:
                traces.append(go.Bar(name=cohort.name, x=cohort.index, y=cohort))

            # add a line showing the overall mean
            mean_df = pd.DataFrame(float(self.backtest_report[param].mean()),
                                   index=self.backtest_report.index, columns=['mean'])
            traces.append(go.Scatter(name='mean', x=mean_df.index, y=mean_df['mean'], mode='lines'))

            metric = param.split('-')[1].upper()
            fig = go.Figure(traces)
            fig.update_layout(title=f'{self.backtest_months} Month Backtest - {metric}',
                                  xaxis=dict(title='Month Since First Disbursement'),
                                  yaxis=dict(title=param))

            if show:
                fig.show()

            return fig
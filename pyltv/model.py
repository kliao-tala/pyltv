# -----------------------------------------------------------------------------------------------------------------
# Data Model
#
# This library defines the base data model class from which all pyLTV models can be built from.
# -----------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from constants import forex, epsilon, dcf, late_fee
from plotly import graph_objects as go
import plotly.io as pio

# change default plotly theme
pio.templates.default = "plotly_white"


class DataManager:
    """Data Manager Class

    Contains functionality to clean, model, and visualize LTV data, as well as implement
    various forecasting strategies and backtest them.

    Parameters
    ----------
    data : pandas DataFrame
        Raw data pulled from Looker.
    market : str
        Market that the data corresponds to.
    retention_method

    Methods
    -------
    load_dependent_data()
        Load data that the models depend on such as recovery rates, opex, and cost of
        capital.

    clean_data()
        Performs all data cleaning steps required before modeling.

    borrower_retention(cohort_data)
        Computes borrower retention.
    """

    def __init__(self, data, market, to_usd=True, bake_duration=4, convenient=True):
        """
        Sets model attributes, loads additional data required for models (inputs &
        ltv_expected), and cleans data.

        Parameters
        ----------
        data : pandas DataFrame
            LTV data loaded from the Looker dashboard:
            https://inventure.looker.com/looks/7451

        market : str
            The market the data corresponds to (KE, PH, MX, etc.).

        to_usd : bool
            If True, local market currency is converted to USD using the foreign exchange
            rate defined in constants.py. If False, data is unchanged.

        convenient : float [0, None]
            Must be greater than 0. This is the stress to default rates and is an additive
            term in the model. E.g. a 0.01 stress means a 5% default becomes 6%.
        """
        self.data = data
        self.market = market
        self.to_usd = to_usd
        self.bake_duration = bake_duration
        self.fx = forex[market]
        self.eps = epsilon
        self.dcf = dcf
        self.late_fee = late_fee
        self.label_cols = ['First Loan Local Disbursement Month', 'Total Interest Assessed', 'Total Rollover Charged',
       'Total Rollover Reversed', 'Months Since First Loan Disbursed', 'Default Rate Amount 7D',
       'Default Rate Amount 30D', 'Default Rate Amount 51D', 'cohort', 'data_type']

        # model attributes to be defined later on
        self.raw = None
        self.inputs = None

        # load data that the models depend on and clean LTV data from Looker
        self.load_dependent_data()
        if convenient:
            self.clean_data()
            self.raw = self.generate_features(self.raw)
            self.data = self.generate_features(self.data)

        # print the date range that the data spans
        min_date = str(pd.to_datetime(self.data.cohort).min())[:7]
        max_date = str(pd.to_datetime(self.data.cohort).max())[:7]
        n_months = self.data.cohort.nunique()
        print(f'Raw data spans {min_date} to {max_date}')
        print(f'Total # of cohorts: {n_months}')
        print('')

    def load_dependent_data(self):
        """
        Loads other data that the model depends on. ltv_inputs.csv contains various inputs required for the forecasting
        models. ltv_expected.csv contains the historical LTV data.
        """
        self.inputs = pd.read_csv('data/model_dependencies/ltv_inputs.csv').set_index('market')

    def clean_data(self):
        """
        Performs various data clean up steps to prepare data for model.
        """
        # remove any leading or trailing spaces in col names
        self.data.columns = [c.strip() for c in self.data.columns]

        # rename any columns with unmatching names
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
            self.data['Total Amount'] /= self.fx
            self.data['Total Interest Assessed'] /= self.fx
            self.data['Total Rollover Charged'] /= self.fx
            self.data['Total Rollover Reversed'] /= self.fx

        # save raw data df before removing data for forecast
        self.raw = self.data.copy()

        # remove the last mo of the raw data
        cohort_data = []
        for c in self.raw.cohort.unique():
            c_data = self.raw[self.raw.cohort == c]
            c_data.index = np.arange(1, len(c_data) + 1)

            cohort_data.append(c_data.iloc[:-1])

        self.raw = pd.concat(cohort_data, axis=0)

        # remove the last 4 months of data for each cohort
        # this is to ensure default_rate_51dpd data is fully baked
        cohort_data = []
        for c in self.data.cohort.unique():
            c_data = self.data[self.data.cohort == c]
            c_data.index = np.arange(1, len(c_data)+1)

            cohort_data.append(c_data.iloc[:-self.bake_duration])

        self.data = pd.concat(cohort_data, axis=0)

    # --- DATA FUNCTIONS --- #
    def borrower_retention(self, cohort_data):
        """
        Computes borrower retention from Count Borrowers. At each time period,
        retention is simply Count Borrowers divided by the original cohort size.

        Parameters
        ----------
        cohort_cohort_data : pandas DataFrame
            Dataframe containing Count Borrowers for a single cohort.

        Returns
        -------
        borrower_retention : pandas Series
            Retention rate for each time period for the given cohort.
        """
        return cohort_data['Count Borrowers'] / cohort_data['Count Borrowers'].max()

    def borrower_survival(self, cohort_data):
        """
        Computes borrower survival from Count Borrowers. At each time period,
        survival is equal to borrower_retention divided by borrower_retention
        in the previous period. This is equivalent to Count Borrowers divided
        by Count Borrowers in the previous period.

        Parameters
        ----------
        cohort_data : pandas DataFrame
            Dataframe containing borrower_retention for a single cohort.

        Returns
        -------
        borrower_survival : pandas Series
            Survival rate for each time period for the given cohort.
        """
        return cohort_data['borrower_retention'] / cohort_data['borrower_retention'].shift(1)

    def loans_per_borrower(self, cohort_data):
        """
        Computes the average number of loans per borrower. Loans per borrower
        is equal to Count Loans divided by Count Borrowers.

        Parameters
        ----------
        cohort_data : pandas DataFrame
            Dataframe containing Count Loans and Count Borrowers for a single
            cohort.

        Returns
        -------
        loans_per_borrower : pandas Series
            The average number of loans per borrower for each time period for
            the given cohort.
        """
        return cohort_data['Count Loans'] / cohort_data['Count Borrowers']

    def loan_size(self, cohort_data):
        """
        Computes the average loan size per borrower. Loan size is equal to
        Total Amount divided by Count Loans.

        Parameters
        ----------
        cohort_data : pandas DataFrame
            Dataframe containing Total Amount and Count Loans for a single cohort.

        to_usd : bool
            If True, converts local currency to USD using the exchange rate
            set by self.fx. If false, the local currency is not converted
            to USD.

        Returns
        -------
        loan_size : pandas Series
            The average loan size for each time period for the given cohort.
        """
        return cohort_data['Total Amount'] / cohort_data['Count Loans']

    def interest_rate(self, cohort_data):
        """
        Computes the average interest rate per loan. Interest Rate is equal to
        Total Interest Assessed divided by Total Amount.

        Parameters
        ----------
        cohort_data : pandas DataFrame
            Dataframe containing Total Interest Assessed and Total Amount for
            a single cohort.

        Returns
        -------
        loan_size : pandas Series
            The average loan size for each time period for the given cohort.
        """
        return cohort_data['Total Interest Assessed'] / cohort_data['Total Amount']

    def default_rate(self, cohort_data, dpd=7):
        """
        Computes the default rate for a specified days past due (dpd). The 7dpd
        default rate is taken as is from the raw LTV data and null values are
        filled in with 0.

        Parameters
        ----------
        cohort_data : pandas DataFrame
            Dataframe containing default rates for a single cohort.

        dpd : int
            The number of days past due which defines the range of time to compute
            the default rate over.
        default_scaling : float
            Scaling factor to test what-if scenarios. A value greater than 1 increases
            the 7dpd default rate while a value less than 1 decreases it by the scale.

        Returns
        -------
        default_rate : pandas Series
            The average default rate at the specified dpd.
        """
        if dpd == 7:
            return cohort_data['Default Rate Amount 7D']

        elif dpd == 51:
            default_rate = cohort_data['Default Rate Amount 51D'].copy()

            recovery_rate_30 = float(self.inputs.loc[self.market, 'recovery_7-30'])
            recovery_rate_51 = float(self.inputs.loc[self.market, 'recovery_30-51'])
            derived_30dpd = cohort_data['Default Rate Amount 7D']*(1-recovery_rate_30)
            derived_51dpd = derived_30dpd*(1-recovery_rate_51)

            return default_rate.fillna(derived_51dpd)

        elif dpd == 365:
            # get actual cohort_data if it exists
            default_rate = np.nan * cohort_data['Default Rate Amount 51D'].copy()

            recovery_rate_365 = float(self.inputs.loc[self.market, 'recovery_51_'])

            derived_365dpd = cohort_data['default_rate_51dpd']*(1-recovery_rate_365)

            return default_rate.fillna(derived_365dpd)

    def loans_per_original(self, cohort_data):
        """
        Computes the average number of loans per original (original number of
        borrowers in the cohort).

        Parameters
        ----------
        cohort_data : pandas DataFrame
            Dataframe containing Count Borrowers for a single cohort.

        dpd : int
            The number of days past due which defines the range of time to compute
            the default rate over.

        Returns
        -------
        default_rate : pandas Series
            The average default rate at the specified dpd.
        """
        return cohort_data['Count Loans'] / cohort_data['Count Borrowers'].max()

    def origination_per_original(self, cohort_data):
        return cohort_data['loans_per_original'] * cohort_data['loan_size']

    def revenue_per_original(self, cohort_data):
        interest_revenue = cohort_data['origination_per_original'] * cohort_data['interest_rate']
        late_fee_revenue = (cohort_data['origination_per_original'] + interest_revenue) * \
                           cohort_data['default_rate_7dpd'] * late_fee

        return interest_revenue + late_fee_revenue

    def credit_margin(self, cohort_data):
        return cohort_data['revenue_per_original'] - \
               (cohort_data['origination_per_original'] + cohort_data['revenue_per_original']) * \
               cohort_data['default_rate_365dpd']

    def opex_per_original(self, cohort_data):
        opex_cost_per_loan = float(self.inputs.loc[self.market, 'opex cost per loan'])
        cost_of_capital = float(self.inputs.loc[self.market, 'cost of capital']) / 12

        return opex_cost_per_loan * cohort_data['loans_per_original'] + \
               cost_of_capital * cohort_data['origination_per_original']

    def opex_coc_per_original(self, cohort_data):
        cost_of_capital = float(self.inputs.loc[self.market, 'cost of capital']) / 12

        return cost_of_capital * cohort_data['origination_per_original']

    def opex_cpl_per_original(self, cohort_data):
        opex_cost_per_loan = float(self.inputs.loc[self.market, 'opex cost per loan'])

        return opex_cost_per_loan * cohort_data['loans_per_original']

    def ltv_per_original(self, cohort_data):
        return cohort_data['cm$_per_original'] - cohort_data['opex_per_original']

    def dcf_ltv_per_original(self, cohort_data):
        return cohort_data['ltv_per_original']/(1+(dcf/12)*cohort_data['ltv_per_original'].index)

    def credit_margin_percent(self, cohort_data):
        return cohort_data['ltv_per_original'] / cohort_data['revenue_per_original']

    # --- CONVENIENCE FUNCTIONS --- #
    def generate_features(self, data):
        """
        Generate all features required for pLTV model.
        """
        cohort_dfs = []
        for c in data.cohort.unique():
            cohort_data = data[data.cohort == c].copy()

            cohort_data['borrower_retention'] = self.borrower_retention(cohort_data)
            cohort_data['borrower_survival'] = self.borrower_survival(cohort_data)
            cohort_data['loans_per_borrower'] = self.loans_per_borrower(cohort_data)
            cohort_data['loan_size'] = self.loan_size(cohort_data)
            cohort_data['interest_rate'] = self.interest_rate(cohort_data)
            cohort_data['default_rate_7dpd'] = self.default_rate(cohort_data, dpd=7)
            cohort_data['default_rate_51dpd'] = self.default_rate(cohort_data, dpd=51)
            cohort_data['default_rate_365dpd'] = self.default_rate(cohort_data, dpd=365)
            cohort_data['loans_per_original'] = self.loans_per_original(cohort_data)
            cohort_data['cumulative_loans_per_original'] = cohort_data['loans_per_original'].cumsum()
            cohort_data['origination_per_original'] = self.origination_per_original(cohort_data)
            cohort_data['cumulative_origination_per_original'] = cohort_data['origination_per_original'].cumsum()
            cohort_data['revenue_per_original'] = self.revenue_per_original(cohort_data)
            cohort_data['cumulative_revenue_per_original'] = cohort_data['revenue_per_original'].cumsum()
            cohort_data['cm$_per_original'] = self.credit_margin(cohort_data)
            cohort_data['cumulative_cm$_per_original'] = cohort_data['cm$_per_original'].cumsum()
            cohort_data['opex_per_original'] = self.opex_per_original(cohort_data)
            cohort_data['cumulative_opex_per_original'] = cohort_data['opex_per_original'].cumsum()
            cohort_data['opex_coc_per_original'] = self.opex_coc_per_original(cohort_data)
            cohort_data['cumulative_opex_coc_per_original'] = cohort_data['opex_coc_per_original'].cumsum()
            cohort_data['opex_cpl_per_original'] = self.opex_cpl_per_original(cohort_data)
            cohort_data['cumulative_opex_cpl_per_original'] = cohort_data['opex_cpl_per_original'].cumsum()
            cohort_data['ltv_per_original'] = self.ltv_per_original(cohort_data)
            cohort_data['cumulative_ltv_per_original'] = cohort_data['ltv_per_original'].cumsum()
            cohort_data['dcf_ltv_per_original'] = self.dcf_ltv_per_original(cohort_data)
            cohort_data['cumulative_dcf_ltv_per_original'] = cohort_data['dcf_ltv_per_original'].cumsum()
            cohort_data['cm%_per_original'] = self.credit_margin_percent(cohort_data)

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

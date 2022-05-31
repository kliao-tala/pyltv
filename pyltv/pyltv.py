# -----------------------------------------------------------------------------------------------------------------
# LTV Forecasting Library
#
# This library defines a Model class that provides functionality for LTV data modeling and forecasting.
# -----------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from sbg import s, log_likelihood

# plotting
from plotly import graph_objects as go
import plotly.io as pio

# change default plotly theme
pio.templates.default = "plotly_white"


# --- MODEL PARAMETERS --- #
# exchange rates
forex = {'ke': 115, 'ph': 51, 'mx': 20}
# epsilon to avoid division by 0
epsilon = 1e-50
# initial values for alpha & beta in sbg model.
alpha = beta = 1
# discounted annual rate for discounted cash flow (DCF) framework
dcf = 0.15


# --- MODEL --- #
class Model:
    """Data Modeling Class

    Contains functionality to clean, model, and visualize LTV data, as well as implement
    various forecasting strategies and backtest them.

    Parameters
    ----------
    data : pandas DataFrame
        Raw data pulled from Looker.
    market : str
        Market that the data corresponds to.
    fcast_method

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

    def __init__(self, data, market='ke', fcast_method='powerslope', bake_duration=4, default_stress=None,
                 retention_effect=False, convenient=True, historic=False, expectations=None):
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

        fcast_method : str
            String specifying the forecasting methodology to employ. Currently there are 4
            methods:
                - powerslope:
                - sbg:
                - sbg-slope:
                - sbg-slope-scaled:
        default_stress : float [0, None]
            Must be greater than 0. This is the stress to default rates and is an additive
            term in the model. E.g. a 0.01 stress means a 5% default becomes 6%.
        """
        self.data = data
        self.market = market
        self.method = fcast_method
        self.historic = historic
        self.bake_duration = bake_duration
        self.alpha = alpha
        self.beta = beta
        self.fx = forex[market]
        self.eps = epsilon
        self.default_stress = default_stress
        self.retention_effect = retention_effect
        self.expectations = expectations
        self.label_cols = ['First Loan Local Disbursement Month', 'Total Interest Assessed', 'Total Rollover Charged',
       'Total Rollover Reversed', 'Months Since First Loan Disbursed', 'Default Rate Amount 7D',
       'Default Rate Amount 30D', 'Default Rate Amount 51D', 'cohort', 'data_type']

        # model attributes to be defined later on
        self.raw = None
        self.inputs = None
        self.ltv_expected = None
        self.min_months = None
        self.backtest_months = None

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
        if self.historic:
            self.ltv_expected = pd.read_csv(f'data/model_dependencies/{self.market}_historical_ltv_expected.csv')
        elif self.expectations:
            self.ltv_expected = pd.read_csv(f'data/model_dependencies/{self.expectations}')
        else:
            self.ltv_expected = pd.read_csv(f'data/model_dependencies/{self.market}_ltv_expected.csv')
        self.ltv_expected.index = np.arange(1, len(self.ltv_expected)+1)

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

        # save raw data df before removing data for forecast
        self.raw = self.data.copy()

        # remove the last mo of the raw data
        cohort_data = []
        for c in self.raw.cohort.unique():
            c_data = self.raw[self.raw.cohort == c]

            cohort_data.append(c_data.iloc[:-1])

        self.raw = pd.concat(cohort_data, axis=0)

        # remove the last 4 months of data for each cohort
        # this is to ensure default_rate_51dpd data is fully baked
        cohort_data = []
        for c in self.data.cohort.unique():
            c_data = self.data[self.data.cohort == c]

            cohort_data.append(c_data.iloc[:-self.bake_duration])

        self.data = pd.concat(cohort_data, axis=0)

    # --- DATA FUNCTIONS --- #
    def borrower_retention(self, cohort_data):
        """
        Computes borrower retention from Count Borrowers. At each time period,
        retention is simply Count Borrowers divided by the original cohort size.

        Parameters
        ----------
        cohort_data : pandas DataFrame
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

    def loan_size(self, cohort_data, to_usd=True):
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
        df = cohort_data['Total Amount'] / cohort_data['Count Loans']
        if to_usd:
            df /= self.fx
        return df

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
            # get actual data if it exists
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
        df = cohort_data['loans_per_original'] * cohort_data['loan_size']
        return df

    def revenue_per_original(self, cohort_data):
        interest_revenue = cohort_data['origination_per_original'] * cohort_data['interest_rate']

        # 0.08 is the % fee we charge to defaulted customers
        revenue = interest_revenue + (cohort_data['origination_per_original'] + interest_revenue) * \
                  cohort_data['default_rate_7dpd'] * 0.08

        # note that origination_per_original is already in USD so no conversion is necessary
        return revenue

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

    def generate_features(self, data, to_usd=True):
        """
        Generate all features required for pLTV model.
        """
        cohorts = []

        # for each cohort
        for cohort in data.loc[:, 'First Loan Local Disbursement Month'].unique():
            # omit the last month of incomplete data
            cohort_data = data[data['First Loan Local Disbursement Month'] == cohort].copy()
            cohort_data.index = np.arange(1, len(cohort_data)+1)

            cohort_data['loans_per_borrower'] = self.loans_per_borrower(cohort_data)
            cohort_data['loan_size'] = self.loan_size(cohort_data, to_usd)
            cohort_data['interest_rate'] = self.interest_rate(cohort_data)

            # call data functions to generate calculated features
            if self.retention_effect:
                # adjust Count Loans & Count Borrowers
                new_loan_count = {}
                new_borrower_count = {}
                new_total_amount = {}
                new_interest_assessed = {}
                lost_loan_sum = 0
                lost_borrower_sum = 0
                lost_amount_sum = 0
                lost_interest_sum = 0
                for t in cohort_data.index:
                    if t == 1:
                        new_loan_count[t] = cohort_data.loc[t, 'Count Loans']
                        new_borrower_count[t] = cohort_data.loc[t, 'Count Borrowers']
                        new_total_amount[t] = cohort_data.loc[t, 'Total Amount']
                        new_interest_assessed[t] = cohort_data.loc[t, 'Total Interest Assessed']
                    else:
                        # the number of lost borrowers is simply equal to the number of defaulted loans
                        borrowers_defaulted_from_stress = self.default_stress * new_loan_count[t - 1]

                        lost_borrower_sum += borrowers_defaulted_from_stress
                        new_borrower_count[t] = cohort_data.loc[t, 'Count Borrowers'] - lost_borrower_sum

                        new_loan_count[t] = cohort_data.loc[t, 'Count Loans'] - \
                                            lost_borrower_sum * cohort_data.loc[t, 'loans_per_borrower']

                        new_total_amount[t] = cohort_data.loc[t, 'Total Amount'] - \
                                              lost_borrower_sum * cohort_data.loc[t, 'loans_per_borrower'] * \
                                              cohort_data.loc[t, 'loan_size'] * self.fx

                        new_interest_assessed[t] = cohort_data.loc[t, 'Total Interest Assessed'] - \
                                                   lost_borrower_sum * cohort_data.loc[t, 'loans_per_borrower'] * \
                                                   cohort_data.loc[t, 'interest_rate'] * \
                                                   cohort_data.loc[t, 'loan_size'] * self.fx

                # need the round because astype floors to the lowest int
                cohort_data['Count Loans'] = pd.Series(new_loan_count).round(0).astype(int)
                cohort_data['Count Borrowers'] = pd.Series(new_borrower_count).round(0).astype(int)
                cohort_data['Total Amount'] = pd.Series(new_total_amount).round(0).astype(int)
                cohort_data['Total Interest Assessed'] = pd.Series(new_interest_assessed).round(0).astype(int)

            cohort_data['borrower_retention'] = self.borrower_retention(cohort_data)
            cohort_data['borrower_survival'] = self.borrower_survival(cohort_data)
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
            cohort_data['cumulative_opex_coc_per_original'] = self.opex_coc_per_original(cohort_data).cumsum()
            cohort_data['opex_cpl_per_original'] = self.opex_cpl_per_original(cohort_data)
            cohort_data['cumulative_opex_cpl_per_original'] = self.opex_cpl_per_original(cohort_data).cumsum()
            cohort_data['ltv_per_original'] = self.ltv_per_original(cohort_data)
            cohort_data['cumulative_ltv_per_original'] = cohort_data['ltv_per_original'].cumsum()
            cohort_data['dcf_ltv_per_original'] = self.dcf_ltv_per_original(cohort_data)
            cohort_data['cumulative_dcf_ltv_per_original'] = cohort_data['dcf_ltv_per_original'].cumsum()
            cohort_data['cm%_per_original'] = self.credit_margin_percent(cohort_data)

            # append the data
            cohorts.append(cohort_data)

        self.cohorts = cohorts
        return pd.concat(cohorts, axis=0)

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

    # --- FORECAST FUNCTIONS --- #
    def forecast_data(self, data, min_months=5, n_months=50):
        """
        Generates a forecast of "Count Borrowers" out to the input number of months.
        The original and forecasted values are returned as a new dataframe, set as
        a new attribute of the model, *.forecast*.

        Parameters
        ----------
        data : pandas dataframe
        method : str
        months : int
            Number of months to forecast to.
        to_usd : bool
        """

        self.min_months = min_months

        # initialize alpha and beta, optimized later by model
        alpha = beta = 1

        # list to hold individual cohort forecasts
        forecast_dfs = []

        # range of desired time periods
        times = np.arange(1, n_months+1)
        times_dict = {i: i-1 for i in times}

        # --- DEFAULT RATE FACTORS --- #
        # compute the default rate std dev across cohorts for the first 12 months
        default_std = self.data[['cohort', 'default_rate_7dpd']].copy()
        default_std = default_std.set_index('cohort', append=True).unstack(-2).iloc[:, :12]
        default_std = default_std.std()
        default_std.index = np.arange(1, len(default_std) + 1)

        def func(t, A, B):
            return A * t ** B

        params, covs = curve_fit(func, default_std.index, default_std)

        default_std_fit = func(times, params[0], params[1])
        default_std_fit = pd.Series(default_std_fit, index=times)

        default_expected_7 = self.ltv_expected['default_rate_7dpd']
        default_expected_51 = self.ltv_expected['default_rate_51dpd']
        default_expected_365 = self.ltv_expected['default_rate_365dpd']

        default_factors = []
        for c in self.data.cohort.unique():
            c_data = self.data[self.data.cohort==c]['default_rate_7dpd']

            default_factors.append(np.mean((c_data - default_expected_7[:len(c_data)])/default_std_fit[:len(c_data)]))
        default_factors = pd.Series(default_factors, index=self.data.cohort.unique())
        # -------------------------------#

        for cohort in data.cohort.unique():
            # data for current cohort
            c_data = data[data.cohort == cohort].copy()

            # starting cohort size
            n = c_data.loc[1, 'Count Borrowers']
            n_valid = len(c_data)

            # only for cohorts with at least 4 data points
            if len(c_data) >= min_months:

                # null df used to extend original cohort df to desired number of forecast months
                dummy_df = pd.DataFrame(np.nan, index=times, columns=['null'])

                # create label column to denote actual vs forecast data
                c_data.loc[:, 'data_type'] = 'actual'

                # extend cohort df
                c_data = pd.concat([c_data, dummy_df], axis=1).drop('null', axis=1)

                # fill missing values in each col
                c_data.cohort = c_data.cohort.ffill()
                c_data['First Loan Local Disbursement Month'] = \
                    c_data['First Loan Local Disbursement Month'].ffill()
                c_data['Months Since First Loan Disbursed'] = \
                    c_data['Months Since First Loan Disbursed'].fillna(times_dict).astype(int)

                # label forecasted data
                c_data.data_type = c_data.data_type.fillna('forecast')

                if self.method == 'powerslope':
                    def power_fcast(c_data, param='borrower_retention'):

                        c = c_data[param].dropna()

                        def power_fit(times, a, b):
                            return a * np.array(times)**b

                        # fit actuals and extract a & b params
                        popt, pcov = curve_fit(power_fit, c.index, c)

                        a = 1#popt[0]
                        b = popt[1]

                        # scale b according to market
                        if self.market=='ke':
                            if len(c) < 6:
                                b = b + .02 * (6 - len(c) - 1)
                        if self.market=='ph':
                            if len(c) < 6:
                                b = b + .02 * (6 - len(c) - 1)
                        if self.market=='mx':
                            b = b - .015 * (18 - len(c) - 1)

                        # get max survival from inputs
                        max_survival = self.inputs.loc[self.market, 'max_monthly_borrower_retention']

                        # take the slope of the power fit between the current and previous time periods
                        # errstate handles division by 0 errors
                        with np.errstate(divide='ignore'):
                            shifted_fit = power_fit(times-1, a, b)
                            shifted_fit[np.isinf(shifted_fit)] = 1
                        power_slope = power_fit(times, a, b) / shifted_fit

                        # apply max survival condition
                        power_slope[power_slope > max_survival] = max_survival
                        # only need values for times we're going to forecast for.
                        power_slope = power_slope[len(c):]
                        power_slope = pd.Series(power_slope, index=[t for t in times[len(c):]])

                        c_fcast = c.copy()
                        for t in times[len(c):]:
                            c_fcast.loc[t] = c_fcast[t - 1] * power_slope[t]

                        return c_fcast

                    forecast = power_fcast(c_data)
                    forecast.index = np.arange(1, len(c_data)+1)
                    # fill in the forecasted data
                    c_data['borrower_retention'] = c_data['borrower_retention'].fillna(forecast)

                    # compute Count Borrowers
                    fcast_count = []
                    for t in times:
                        if t < len(c_data['Count Borrowers'].dropna()):
                            fcast_count.append(c_data.loc[t, 'Count Borrowers'])
                        else:
                            fcast_count.append(n * forecast[t])

                    c_data['Count Borrowers'] = pd.Series(fcast_count, index=times)

                elif self.method == 'sbg':
                    c = c_data['Count Borrowers'].dropna()

                    # define bounds for alpha and beta (must be positive)
                    bounds = ((0, 1e5), (0, 1e5))

                    # use scipy's minimize function on log_likelihood to optimize alpha and beta
                    results = minimize(log_likelihood, np.array([alpha, beta]), args=c, bounds=bounds)

                    # list to hold forecasted values
                    forecast = []
                    for t in times:
                        forecast.append(n * s(t, results.x[0], results.x[1]))

                    # convert list to dataframe
                    forecast = pd.DataFrame(forecast, index=times, columns=['Count Borrowers'])

                    # fill in the forecasted data
                    c_data['Count Borrowers'] = c_data['Count Borrowers'].fillna(forecast['Count Borrowers'])

                    # add retention
                    c_data['borrower_retention'] = self.borrower_retention(c_data)

                elif self.method == 'sbg-slope':
                    c = c_data['Count Borrowers'].dropna()

                    # define bounds for alpha and beta (must be positive)
                    bounds = ((0, 1e5), (0, 1e5))

                    # use scipy's minimize function on log_likelihood to optimize alpha and beta
                    results = minimize(log_likelihood, np.array([alpha, beta]), args=c, bounds=bounds)
                    alpha, beta = results.x

                    # list to hold forecasted values
                    forecast = [c.iloc[0]]
                    for t in times:
                        forecast.append(n * s(t, alpha, beta))

                    # convert list to dataframe
                    count_forecast = pd.DataFrame(forecast, index=[0]+times, columns=['Count Borrowers'])
                    survival_forecast = count_forecast/count_forecast.shift(1)

                    # get max survival from inputs
                    max_survival = self.inputs.loc[self.market, 'max_monthly_borrower_retention'].astype(float)

                    # cap survival at max from inputs
                    survival_forecast = survival_forecast['Count Borrowers'].apply(lambda x: \
                                                           x if x <= max_survival else max_survival)

                    c_fcast = c_data['Count Borrowers'].copy()
                    for t in times[len(c)-1:]:
                        c_fcast.loc[t] = float(c_fcast.loc[t - 1] * survival_forecast.loc[t])

                    c_data['Count Borrowers'] = c_data['Count Borrowers'].fillna(c_fcast)

                    # add retention
                    c_data['borrower_retention'] = self.borrower_retention(c_data)

                elif self.method == 'sbg-slope-scaled':
                    c = c_data['Count Borrowers'].dropna()

                    # define bounds for alpha and beta (must be positive)
                    bounds = ((0, 1e5), (0, 1e5))

                    # use scipy's minimize function on log_likelihood to optimize alpha and beta
                    results = minimize(log_likelihood, np.array([alpha, beta]), args=c, bounds=bounds)
                    alpha, beta = results.x

                    # list to hold forecasted values
                    forecast = [c.iloc[0]]
                    for t in times:
                        if t > 1:
                            forecast.append(n * s(t, alpha, beta))

                    # convert list to dataframe
                    count_forecast = pd.DataFrame(forecast, index=times, columns=['Count Borrowers'])
                    survival_forecast = 0.98 * count_forecast/count_forecast.shift(1)

                    # get max survival from inputs
                    max_survival = self.inputs.loc[self.market, 'max_monthly_borrower_retention'].astype(float)

                    # cap survival at max from inputs
                    survival_forecast = survival_forecast['Count Borrowers'].apply(lambda x: \
                                                           x if x <= max_survival else max_survival)

                    c_fcast = c_data['Count Borrowers'].copy()
                    for t in times[len(c)-1:]:
                        c_fcast.loc[t] = float(c_fcast.loc[t - 1] * survival_forecast.loc[t])

                    c_data['Count Borrowers'] = c_data['Count Borrowers'].fillna(c_fcast)

                    # add retention
                    c_data['borrower_retention'] = self.borrower_retention(c_data)

                # --- ALL OTHERS --- #
                # compute survival
                c_data['borrower_survival'] = self.borrower_survival(c_data)

                # forecast loan size
                for i in c_data[c_data.loan_size.isnull()].index:
                    c_data.loc[i, 'loan_size'] = c_data.loc[i - 1, 'loan_size'] * \
                                                 self.ltv_expected.loc[i, 'loan_size'] / self.ltv_expected.loc[
                                                     i - 1, 'loan_size']

                # forecast loans_per_borrower
                for i in c_data[c_data.loans_per_borrower.isnull()].index:
                    c_data.loc[i, 'loans_per_borrower'] = self.ltv_expected.loc[i, 'loans_per_borrower']

                # forecast Count Loans
                c_data['Count Loans'] = c_data['Count Loans'].fillna(
                    (c_data['loans_per_borrower']) * c_data['Count Borrowers'])

                # forecast Total Amount
                c_data['Total Amount'] = c_data['Total Amount'].fillna(
                    (c_data['loan_size'] * self.fx) * c_data['Count Loans'])

                # forecast Interest Rate
                for i in c_data[c_data.interest_rate.isnull()].index:
                    c_data.loc[i, 'interest_rate'] = c_data.loc[i - 1, 'interest_rate'] * \
                                                     self.ltv_expected.loc[i, 'interest_rate'] / self.ltv_expected.loc[
                                                         i - 1, 'interest_rate']

                # Forecast default rates
                # 7DPD
                default_fcast = []
                for t in times:
                    if t < n_valid+1:
                        default_fcast.append(c_data.loc[t, 'default_rate_7dpd'])
                    else:
                        default_fcast.append(default_expected_7[t] + default_factors[cohort]*default_std_fit[t])
                default_fcast = pd.Series(default_fcast, index=times)

                c_data['default_rate_7dpd'] = default_fcast

                # 51DPD
                default_fcast = []
                for t in times:
                    if t < n_valid+1:
                        default_fcast.append(c_data.loc[t, 'default_rate_51dpd'])
                    else:
                        default_fcast.append(default_expected_51[t] + default_factors[cohort] * default_std_fit[t])
                default_fcast = pd.Series(default_fcast, index=times)

                c_data['default_rate_51dpd'] = default_fcast

                # 365DPD
                default_fcast = []
                for t in times:
                    if t < n_valid+1:
                        default_fcast.append(c_data.loc[t, 'default_rate_365dpd'])
                    else:
                        default_fcast.append(default_expected_365[t] + default_factors[cohort] * default_std_fit[t])
                default_fcast = pd.Series(default_fcast, index=times)

                c_data['default_rate_365dpd'] = default_fcast

                if self.default_stress:
                    c_data['default_rate_7dpd'] += self.default_stress
                    c_data['default_rate_365dpd'] += self.default_stress

                # compute remaining columns from forecasts
                c_data['loans_per_original'] = self.loans_per_original(c_data)
                c_data['cumulative_loans_per_original'] = c_data['loans_per_original'].cumsum()
                c_data['origination_per_original'] = self.origination_per_original(c_data)
                c_data['cumulative_origination_per_original'] = c_data['origination_per_original'].cumsum()
                c_data['revenue_per_original'] = self.revenue_per_original(c_data)
                c_data['cumulative_revenue_per_original'] = c_data['revenue_per_original'].cumsum()
                c_data['cm$_per_original'] = self.credit_margin(c_data)
                c_data['cumulative_cm$_per_original'] = c_data['cm$_per_original'].cumsum()
                c_data['opex_per_original'] = self.opex_per_original(c_data)
                c_data['cumulative_opex_per_original'] = c_data['opex_per_original'].cumsum()
                c_data['opex_coc_per_original'] = self.opex_coc_per_original(c_data)
                c_data['cumulative_opex_coc_per_original'] = self.opex_coc_per_original(c_data).cumsum()
                c_data['opex_cpl_per_original'] = self.opex_cpl_per_original(c_data)
                c_data['cumulative_opex_cpl_per_original'] = self.opex_cpl_per_original(c_data).cumsum()
                c_data['ltv_per_original'] = self.ltv_per_original(c_data)
                c_data['cumulative_ltv_per_original'] = c_data['ltv_per_original'].cumsum()
                c_data['dcf_ltv_per_original'] = self.dcf_ltv_per_original(c_data)
                c_data['cumulative_dcf_ltv_per_original'] = c_data['dcf_ltv_per_original'].cumsum()
                c_data['cm%_per_original'] = self.credit_margin_percent(c_data)

                # add the forecasted data for the cohort to a list, aggregating all cohort forecasts
                forecast_dfs.append(c_data)

        forecast_df = pd.concat(forecast_dfs)

        return forecast_df

    def backtest_data(self, data, min_months=4, hold_months=4, fcast_months=50,
                      metrics=['rmse', 'me', 'mape', 'mpe']):
        """
        Backtest forecasted values against actuals.

        Parameters
        ----------


        """

        # print the number of cohorts that will be backtested.
        cohort_count = 0
        for cohort in data.cohort.unique():
            if len(data[data.cohort == cohort]) - hold_months >= self.min_months:
                cohort_count += 1

        self.backtest_months = hold_months
        print(f'Backtesting {hold_months} months.')
        print(f'{cohort_count} cohorts will be backtested.')

        def compute_error(actual, forecast, metric):
            """
            Test forecast performance against actuals using method defined by metric.
            """
            # root mean squared error
            if metric == 'rmse':
                error = np.sqrt((1 / len(actual)) * sum((forecast[:len(actual)] - actual) ** 2))
            # mean absolute error
            elif metric == 'mae':
                error = np.mean(abs(forecast[:len(actual)] - actual))
            # mean error
            elif metric == 'me':
                error = np.mean(forecast[:len(actual)] - actual)
            # mean absolute percent error
            elif metric == 'mape':
                error = round(100 * (1 / len(actual)) * sum(abs((forecast[:len(actual)] - actual) / actual)), 2)
            # mean percent error
            elif metric == 'mpe':
                error = round(100 * (1 / len(actual)) * sum((forecast[:len(actual)] - actual) / actual), 2)
            return error

        # --- Generate backtest data --- #
        backtest_report = []
        backtest_data = []

        for cohort in data.cohort.unique():
            # data for current cohort
            c_data = data[data.cohort == cohort]

            # only backtest if remaining data has at least 4 data points
            if len(c_data) - hold_months >= self.min_months:
                # limit data
                c_data = c_data.iloc[:len(c_data) - hold_months, :]

                # forecast the limited data
                predicted_data = self.forecast_data(c_data, min_months=min_months, n_months=fcast_months)

                # get forecast overlap with actuals
                actual = self.data[self.data['First Loan Local Disbursement Month'] == cohort]

                start = predicted_data[predicted_data.data_type == 'forecast'].index.min()
                stop = actual.index.max()

                # compute errors
                backtest_report_cols = []
                errors = []

                cols = [c for c in self.data.columns if c not in self.label_cols]
                cols.remove('Count First Loans')

                for col in cols:
                    for metric in metrics:
                        error = compute_error(actual.loc[start:stop, col], predicted_data.loc[start:stop, col],
                                              metric=metric)

                        backtest_report_cols += [f'{col}-{metric}']

                        errors.append(error)

                backtest_report.append(pd.DataFrame.from_dict({cohort: errors}, orient='index',
                                                              columns=backtest_report_cols))
                backtest_data.append(predicted_data)

        backtest_data = pd.concat(backtest_data)
        backtest_report = pd.concat(backtest_report, axis=0)
        backtest_report['cohort'] = backtest_report.index

        return backtest_data, backtest_report

    def run_all(self, backtest_months=4):
        self.data = self.generate_features(self.data)
        self.forecast = self.forecast_data(self.data)
        self.backtest, self.backtest_report = self.backtest_data(self.data, months=backtest_months)

        print('...')
        print('Data is clean, forecasted, and backtested.')
        print('')
        print('-----')
        print('Access forecast data with {model name}.forecast, and backtest')
        print('data with {model name}.backtest and {model name}.backtest_report')
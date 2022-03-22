import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
import snowflake.connector

# for private key handling
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# plotting
from plotly import graph_objects as go
import plotly.io as pio
# change default plotly theme
pio.templates.default = "plotly_white"


# --- METHODS --- #
def p(t, alpha, beta):
    """
    Probability that a customer fails to take out another loan (probability to churn).
    For the derivation of this equation, see the original Fader & Hardie paper. This
    recursion formula takes two constants, alpha and beta, which are fit to actual data.
    It then allows you to compute the probability of churn for a given time period, t.

    Parameters
    ----------
    t : int
        Time period.
    alpha : float
        Fitting parameter.
    beta : float
        Fitting parameter.

    Returns
    -------
    P : float
        Probability of churn.
    """

    eps = 1e-50

    if alpha + beta < eps:
        if t == 1:
            return alpha / (eps)
        else:
            return p(t-1, alpha, beta) * (beta + t-2) / (eps + t-1)
    else:
        if t == 1:
            return alpha / (alpha + beta)
        else:
            return p(t-1, alpha, beta) * (beta + t-2) / (alpha + beta + t-1)


def s(t, alpha, beta):
    """
    Survival function: the probability that a customer has survived to time t.
    For the derivation of this equation, see the original Fader & Hardie paper. This
    recursion formula takes two constants, alpha and beta, which are fit to actual data.
    It also requires computation of P (probability of a customer churning).

    Parameters
    ----------
    t : int
        Time period.
    alpha : float
        Fitting parameter.
    beta : float
        Fitting parameter.

    Returns
    -------
    S : float
        Probability of survival.
    """

    if t == 1:
        return 1 - p(t, alpha, beta)
    else:
        return s(t - 1, alpha, beta) - p(t, alpha, beta)


def log_likelihood(params, c):
    """
    Computes the *negative* log-likelihood of the probability distribution of customers
    still being active at time t. For a derivation of the log-likelihood, see Appendix A
    in the original Fader & Hardie paper. The function computes the log-likelihood at
    every time step, t, leading up to the last time period T. The final value is simply
    the sum of the log-likelihood computed at each time step. In the end, we return the
    negative of the log-likelihood so that we can use scipy's minimize function to optimize
    for the values of alpha and beta.

    Parameters
    ----------
    params : array
        Array containing alpha and beta values.
    c : array
        Array containing borrower count for a given cohort.

    Returns
    -------
    ll : float
        log-likelihood value
    """

    alpha, beta = params
    eps = 1e-50

    # initialize log-likelihood (ll) value at 0
    ll = 0

    # for each time period in the *actual* data, compute ll and add it to the running total
    for t in c[1:].index:
        # if P is less than epsilon, replace it with epsilon.
        if p(t, alpha, beta) < eps:
            ll += (c[t-1] - c[t]) * np.log(eps)
        else:
            ll += (c[t-1] - c[t]) * np.log(p(t, alpha, beta))

    # add the final term which is associated with customers who are still active at the end
    # of the final period.

    # replace the argument of the np.log() function with epsilon if smaller than epsilon.
    if s((len(c) - 1) - 1, alpha, beta) - p(len(c) - 1, alpha, beta) < eps:
        ll += c.iloc[-1] * np.log(eps)
    else:
        ll += c.iloc[-1] * np.log(s((len(c) - 1) - 1, alpha, beta) - p(len(c) - 1, alpha, beta))

    return -ll


# --- DATABASE MANAGER --- #
class DBM:
    """
    Database manager.
    """

    def __init__(self, user=None, account='ng95977', warehouse='BUSINESS_WH'):
        self.user = user
        self.account = account
        self.warehouse = warehouse
        self.ctx = None
        self.data = None

        # get private key
        pkey = self.get_private_key_bytes(f'/Users/{self.user}/.ssh/snowflake_private_keypair.pem', None)

        # connect to snowflake
        self.connect(pkey)

    def get_private_key_bytes(self, keyfile, keypass):
        """

        :param keyfile:
        :param keypass:
        :return:
        """

        with open(keyfile, "rb") as key:
            keypass_encoded = None
            if keypass:
                keypass_encoded = keypass.encode()
            p_key = serialization.load_pem_private_key(
                key.read(),
                password=keypass_encoded,
                backend=default_backend())

            return p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption())

    def connect(self, pkey):
        self.ctx = snowflake.connector.connect(
            user=f'{self.user}@tala.co',
            account=f'{self.account}',
            private_key=pkey,
            warehouse=''
        )

    def run_sql(self, sql_file_path=None):
        with open(sql_file_path) as f:
            query = f.read()
            with self.ctx.cursor() as curs:
                results = curs.execute(query)
                return pd.DataFrame.from_records(iter(results), columns=[c[0] for c in results.description])

    def query_db(self, query):
        with self.ctx.cursor() as curs:
            results = curs.execute(query)
            return pd.DataFrame.from_records(iter(results), columns=[c[0] for c in results.description])

    def get_market_data(self, market='ke', start_date='2020-09-01', days_before=60):

        cols = ['First Loan Local Disbursement Month',
                'Months Since First Loan Disbursed',
                'Count First Loans',
                'Count Borrowers',
                'Count Loans',
                'Total Amount',
                'Total Interest Assessed',
                'Total Rollover Charged',
                'Total Rollover Reversed',
                'Default Rate Amount 7D',
                'Default Rate Amount 30D',
                'Default Rate Amount 51D',
                'Default Rate Amount 365D']

        if market != 'ke':
            query_params = {'REPLACE_DATE': start_date,
                            'REPLACE_DAYS': str(days_before),
                            '_KE': f'_{market.upper()}'}
        else:
            query_params = {'REPLACE_DATE': start_date,
                            'REPLACE_DAYS': str(days_before)}

        with open('queries/ke_ltv.sql') as f:
            sql = f.read()
            for p in query_params:
                sql = sql.replace(p, query_params[p])

        data = self.query_db(sql)

        data.columns = cols

        self.data = data

        return data


# --- MODEL --- #
class Model:
    """
    sBG model class containing all functionality for creating, analyzing, and backtesting
    the sBG model.

    Parameters
    ----------
    data : pandas DataFrame
        Raw data pulled from Looker.

    Methods
    -------
    clean_data
        Performs all data cleaning steps and returns the cleaned data.

    borrower_retention(cohort_data)
        Computes borrower retention.

    """

    def __init__(self, data, market='ke', fcast_method='powerslope',
                 alpha=1, beta=1, dollar_ex=0.00925):
        self.data = data
        self.inputs = None
        self.ltv_expected = None
        self.market = market
        self.method = fcast_method
        self.alpha = alpha
        self.beta = beta
        self.dollar_ex = dollar_ex
        self.min_months = None
        self.forecast_cols = ['Count Borrowers', 'borrower_retention', 'borrower_survival', 'loan_size',
                              'loans_per_borrower', 'Count Loans', 'Total Amount', 'interest_rate', 'default_rate_7dpd',
                              'default_rate_51dpd', 'default_rate_365dpd', 'loans_per_original',
                              'origination_per_original', 'revenue_per_original', 'cm$_per_original',
                              'opex_per_original', 'ltv_per_original', 'cm%_per_original']

        self.load_dependent_data()
        self.clean_data()

    def load_dependent_data(self):
        self.inputs = pd.read_csv('data/ltv_inputs.csv').set_index('market')
        self.ltv_expected = pd.read_csv('data/ltv_expected.csv')

    def clean_data(self):
        # fix date inconsistencies
        self.data = self.data.replace({'2021-9': '2021-09', '2021-8': '2021-08', \
                                       '2021-7': '2021-07', '2021-6': '2021-06', \
                                       '2021-5': '2021-05', '2021-4': '2021-04', \
                                       '2020-9': '2020-09'})

        # sort by months since first disbursement
        self.data = self.data.sort_values(['First Loan Local Disbursement Month',
                                           'Months Since First Loan Disbursed'])

        # remove all columns calculated through looker
        self.data = self.data.loc[:, :"Default Rate Amount 51D"]

        # add more convenient cohort column
        self.data['cohort'] = self.data['First Loan Local Disbursement Month']

    # --- DATA FUNCTIONS --- #
    def borrower_retention(self, cohort_data):
        return cohort_data['Count Borrowers'] / cohort_data['Count Borrowers'].max()

    def borrower_survival(self, cohort_data):
        return cohort_data['borrower_retention'] / cohort_data['borrower_retention'].shift(1)

    def loans_per_borrower(self, cohort_data):
        return cohort_data['Count Loans'] / cohort_data['Count Borrowers']

    def loan_size(self, cohort_data, to_usd):
        df = cohort_data['Total Amount'] / cohort_data['Count Loans']
        if to_usd:
            df *= self.dollar_ex
        return df

    def interest_rate(self, cohort_data):
        return cohort_data['Total Interest Assessed'] / cohort_data['Total Amount']

    def default_rate(self, cohort_data, period=7):
        if period == 7:
            return cohort_data['Default Rate Amount 7D'].fillna(0)

        elif period == 51:
            default_rate = cohort_data['Default Rate Amount 51D']

            recovery_rate_51 = float(self.inputs.loc['ke', 'recovery_7-30'] + \
                                     self.inputs.loc['ke', 'recovery_30-51'])

            ## fill null 51dpd values with 7dpd values based on recovery rates
            derived_51dpd = (cohort_data['Count Loans'] * (cohort_data['default_rate_7dpd']) - \
                             cohort_data['Count Loans'] * (cohort_data['default_rate_7dpd']) * recovery_rate_51) / \
                            cohort_data['Count Loans']

            return default_rate.fillna(derived_51dpd)

        elif period == 365:
            # get actual data if it exists
            default_rate = np.nan * cohort_data['Default Rate Amount 51D']

            recovery_rate_365 = float(self.inputs.loc['ke', 'recovery_51_'])

            ## fill null 365dpd values with 51dpd values based on recovery rates
            derived_365dpd = (cohort_data['Count Loans'] * (cohort_data['default_rate_51dpd']) - \
                              cohort_data['Count Loans'] * (cohort_data['default_rate_51dpd']) * \
                              recovery_rate_365) / cohort_data['Count Loans']

            return default_rate.fillna(derived_365dpd)

    def loans_per_original(self, cohort_data):
        return cohort_data['Count Loans'] / cohort_data['Count Borrowers'].max()

    def origination_per_original(self, cohort_data, to_usd):
        df = cohort_data['Total Amount'] / cohort_data['Count Borrowers'].max()
        if to_usd:
            df *= self.dollar_ex
        return df

    def revenue_per_original(self, cohort_data, to_usd):
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
        opex_cost_per_loan = float(self.inputs.loc['ke', 'opex cost per loan'])
        cost_of_capital = float(self.inputs.loc['ke', 'cost of capital']) / 12

        return opex_cost_per_loan * cohort_data['loans_per_original'] + \
               cost_of_capital * cohort_data['origination_per_original']

    def ltv_per_original(self, cohort_data):
        return cohort_data['cm$_per_original'] - cohort_data['opex_per_original']

    def credit_margin_percent(self, cohort_data):
        return cohort_data['ltv_per_original'] / cohort_data['revenue_per_original']

    def generate_features(self, to_usd=True):
        """
        Generate all features required for pLTV model.
        """
        cohorts = []

        # for each cohort
        for cohort in self.data.loc[:, 'First Loan Local Disbursement Month'].unique():
            # omit the last two months of incomplete data
            cohort_data = self.data[self.data['First Loan Local Disbursement Month'] == cohort].iloc[:-2, :]

            # call data functions to generate calculated features
            cohort_data['borrower_retention'] = self.borrower_retention(cohort_data)
            cohort_data['borrower_survival'] = self.borrower_survival(cohort_data)
            cohort_data['loans_per_borrower'] = self.loans_per_borrower(cohort_data)
            cohort_data['loan_size'] = self.loan_size(cohort_data, to_usd)
            cohort_data['interest_rate'] = self.interest_rate(cohort_data)
            cohort_data['default_rate_7dpd'] = self.default_rate(cohort_data, period=7)
            cohort_data['default_rate_51dpd'] = self.default_rate(cohort_data, period=51)
            cohort_data['default_rate_365dpd'] = self.default_rate(cohort_data, period=365)
            cohort_data['loans_per_original'] = self.loans_per_original(cohort_data)
            cohort_data['origination_per_original'] = self.origination_per_original(cohort_data, to_usd)
            cohort_data['revenue_per_original'] = self.revenue_per_original(cohort_data, to_usd)
            cohort_data['cm$_per_original'] = self.credit_margin(cohort_data)
            cohort_data['opex_per_original'] = self.opex_per_original(cohort_data)
            cohort_data['ltv_per_original'] = self.ltv_per_original(cohort_data)
            cohort_data['cm%_per_original'] = self.credit_margin_percent(cohort_data)

            # reset the index and append the data
            cohorts.append(cohort_data.reset_index(drop=True))

        self.cohorts = cohorts
        self.data = pd.concat(cohorts, axis=0)

    def plot_cohorts(self, param, data='raw'):
        """
        Generate scatter plot for a specific paramter.

        Parameters
        ----------

        """

        if data == 'raw' or data == 'forecast' or data == 'backtest':
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

            elif data == 'raw':
                for cohort in self.data.cohort.unique():
                    output = self.data[self.data.cohort == cohort][param]

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

            fig.show()

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

            metric = param.split('-')[1].upper()
            fig = go.Figure(traces)
            fig.update_layout(title=f'{self.backtest_months} Month Backtest - {metric}',
                              xaxis=dict(title='Month Since First Disbursement'),
                              yaxis=dict(title=param))

            fig.show()

    # --- FORECAST FUNCTIONS --- #
    def forecast_features(self, data, min_months=4, months=24, to_usd=True):
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
        times = list(range(1, months + 1))
        times_dict = {i: i for i in times}

        for cohort in data.cohort.unique():
            # data for current cohort
            c_data = data[data.cohort == cohort].copy()

            # starting cohort size
            n = c_data.loc[0, 'Count Borrowers']

            # only for cohorts with at least 4 data points
            if len(c_data) >= min_months:

                # null df used to extend original cohort df to desired number of forecast months
                dummy_df = pd.DataFrame(np.nan, index=range(0, months + 1), columns=['null'])

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
                c_data['Count First Loans'] = c_data['Count First Loans'].ffill()

                # label forecasted data
                c_data.data_type = c_data.data_type.fillna('forecast')

                if self.method == 'powerslope':
                    def power_fcast(c_data, param='borrower_retention'):

                        c = c_data[param].dropna()
                        c.index = np.arange(1, len(c) + 1)

                        def power_fit(times, a, b):
                            y = []
                            for t in times:
                                if t == 0:
                                    y.append(1)
                                else:
                                    y.append(a * t ** b)
                            return np.array(y)

                        # fit actuals and extract a & b params
                        popt, pcov = curve_fit(power_fit, c.index, c)

                        # generate the full range of times to forecast over
                        times = np.arange(1, months + 2)

                        a = popt[0]
                        b = popt[1]

                        # if there is less than 6 months of actuals, scale data.
                        if len(c) < 6:
                            b = b + .2 * (6 - len(c) - 1)

                        # get max survival from inputs
                        max_survival = self.inputs.loc['ke', 'max_monthly_borrower_retention']

                        # take the slope of the power fit between the current and previous time periods
                        power_slope = power_fit(times, a=a, b=b) / power_fit(times - 1, a=a, b=b)

                        # apply max survival condition
                        power_slope_capped = np.array([i if i < max_survival else max_survival for i in power_slope])
                        # only need values for times we're going to forecast for.
                        power_slope_capped = power_slope_capped[len(c):]
                        power_slope_capped = pd.Series(power_slope_capped, index=[t for t in times[len(c):]])

                        c_fcast = c.copy()
                        for t in times[len(c):]:
                            c_fcast.loc[t] = c_fcast[t - 1] * power_slope_capped[t]

                        return c_fcast.reset_index(drop=True)

                    forecast = power_fcast(c_data)
                    # fill in the forecasted data
                    c_data['borrower_retention'] = c_data['borrower_retention'].fillna(forecast)

                    # compute Count Borrowers
                    fcast_count = []
                    for t in forecast.index:
                        if t < len(c_data['Count Borrowers'].dropna()):
                            fcast_count.append(c_data.loc[t, 'Count Borrowers'])
                        else:
                            fcast_count.append(c_data.loc[0, 'Count Borrowers'] * forecast[t])

                    c_data['Count Borrowers'] = pd.Series(fcast_count)

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
                    max_survival = self.inputs.loc['ke', 'max_monthly_borrower_retention'].astype(float)

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
                    c_data.loc[i, 'loans_per_borrower'] = c_data.loc[i - 1, 'loans_per_borrower'] * \
                                                          self.ltv_expected.loc[i, 'loans_per_borrower'] / \
                                                          self.ltv_expected.loc[i - 1, 'loans_per_borrower']

                # forecast Count Loans
                c_data['Count Loans'] = c_data['Count Loans'].fillna(
                    (c_data['loans_per_borrower']) * c_data['Count Borrowers'])

                # forecast Total Amount
                c_data['Total Amount'] = c_data['Total Amount'].fillna(
                    (c_data['loan_size'] / self.dollar_ex) * c_data['Count Loans'])

                # forecast Interest Rate
                for i in c_data[c_data.interest_rate.isnull()].index:
                    c_data.loc[i, 'interest_rate'] = c_data.loc[i - 1, 'interest_rate'] * \
                                                     self.ltv_expected.loc[i, 'interest_rate'] / self.ltv_expected.loc[
                                                         i - 1, 'interest_rate']

                # forecast default rate 7dpd
                default = c_data.default_rate_7dpd.dropna()
                default.index = np.arange(1, len(default) + 1)

                def func(t, A, B):
                    return A * (t ** B)

                params, covs = curve_fit(func, default.index, default)

                t = list(range(1, months + 2))
                fit = func(t, params[0], params[1])
                fit = pd.Series(fit, index=t).reset_index(drop=True)

                c_data['default_rate_7dpd'] = c_data['default_rate_7dpd'].fillna(fit)

                # derive 51dpd and 365 dpd from 7dpd
                c_data['default_rate_51dpd'] = self.default_rate(c_data, period=51)
                c_data['default_rate_365dpd'] = self.default_rate(c_data, period=365)

                # compute remaining columns from forecasts
                c_data['loans_per_original'] = self.loans_per_original(c_data)
                c_data['origination_per_original'] = self.origination_per_original(c_data, to_usd)
                c_data['revenue_per_original'] = self.revenue_per_original(c_data, to_usd)
                c_data['cm$_per_original'] = self.credit_margin(c_data)
                c_data['opex_per_original'] = self.opex_per_original(c_data)
                c_data['ltv_per_original'] = self.ltv_per_original(c_data)
                c_data['cm%_per_original'] = self.credit_margin_percent(c_data)

                # add the forecasted data for the cohort to a list, aggregating all cohort forecasts
                forecast_dfs.append(c_data)

        return pd.concat(forecast_dfs)

    def backtest(self, data, months=4, metric='mpe'):
        """
        Backtest forecasted values against actuals.

        Parameters
        ----------


        """

        # print the number of cohorts that will be backtested.
        cohort_count = 0
        for cohort in data.cohort.unique():
            if len(data[data.cohort == cohort]) - months >= self.min_months:
                cohort_count += 1

        self.backtest_months = months
        print(f'Backtesting {months} months.')
        print(f'{cohort_count} cohorts will be backtested.')

        def compute_error(actual, forecast, metric='mape'):
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
                error = (1 / len(actual)) * sum(abs(forecast[:len(actual)] - actual) / actual)
            # mean percent error
            elif metric == 'mpe':
                error = (1 / len(actual)) * sum((forecast[:len(actual)] - actual) / actual)
            return error

        # --- Generate limited data --- #

        metrics = ['rmse', 'me', 'mape', 'mpe']

        limited_data = []
        backtest_report = []
        for cohort in data.cohort.unique():
            # data for current cohort
            c_data = data[data.cohort == cohort]

            # only backtest if remaining data has at least 4 data points
            if len(c_data) - months >= self.min_months:
                # limit data
                c_data = c_data.iloc[:len(c_data) - months, :]

                # forecast the limited data
                c_data = self.forecast_features(c_data)

                # get forecast overlap with actuals
                actual = self.data[self.data['First Loan Local Disbursement Month'] == cohort]

                start = c_data[c_data.data_type == 'forecast'].index.min()
                stop = actual.index.max()

                # compute errors
                backtest_report_cols = []
                errors = []
                for col in self.forecast_cols:
                    # error
                    c_data.loc[start:stop, f'error-{col}'] = c_data.loc[start:stop, col] - actual.loc[start:stop, col]
                    # % error
                    c_data.loc[start:stop, f'%error-{col}'] = (c_data.loc[start:stop, col] - actual.loc[start:stop,
                                                                                             col]) / \
                                                              actual.loc[start:stop, col]

                    for metric in metrics:
                        error = compute_error(actual.loc[start:stop, col], c_data.loc[start:stop, col],
                                              metric=metric)

                        backtest_report_cols += [f'{col}-{metric}']

                        errors.append(error)

                backtest_report.append(pd.DataFrame.from_dict({cohort: errors}, orient='index',
                                                              columns=backtest_report_cols))
                limited_data.append(c_data)

        backtest_data = pd.concat(limited_data)
        backtest_report = pd.concat(backtest_report, axis=0)
        backtest_report['cohort'] = backtest_report.index

        self.backtest_data = backtest_data
        self.backtest_report = backtest_report

        return backtest_data, backtest_report

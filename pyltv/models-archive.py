
#
# class Rolling4(DataManager):
#     """
#     The Rolling model uses the same methodology as the PowerSlope model for all parameters
#     except default rates. Default rates in the Rolling model are generated from rolling
#     expectations. A seed expectation curve generated from historicals is used to forecast
#     the first cohort. Each subsequent cohort uses a weighted average of the last n_trail
#     cohort forecasts as its expectation curve.
#
#     Parameters
#     ----------
#     data : pandas dataframe
#         Data to forecast. Usually will be self.data which is data that has already
#         been cleaned and processed.
#     market : str
#         The market the data corresponds to (KE, PH, MX, etc.).
#     to_usd : bool
#         If True, convert fields in local currency to USD. If False, leave fields as
#         local currency.
#     bake_duration : int
#         Number of months to consider data fully baked. The last bake_duration number
#         of months is removed from the data during cleaning.
#     """
#     def __init__(self, data, market, to_usd=True, bake_duration=4, ltv_expected=None):
#         """
#         Sets model attributes, loads additional data required for models (inputs &
#         ltv_expected), and cleans data.
#
#         Parameters
#         ----------
#         data : pandas dataframe
#             Data to forecast. Usually will be self.data which is data that has already
#             been cleaned and processed.
#         market : str
#             The market the data corresponds to (KE, PH, MX, etc.).
#         to_usd : bool
#             If True, convert fields in local currency to USD. If False, leave fields as
#             local currency.
#         bake_duration : int
#             Number of months to consider data fully baked. The last bake_duration number
#             of months is removed from the data during cleaning.
#         """
#         super().__init__(data, market, to_usd, bake_duration)
#
#         if ltv_expected:
#             self.ltv_expected = pd.read_csv(f'data/model_dependencies/{ltv_expected}')
#
#         else:
#             # read in expectations
#             self.ltv_expected = pd.read_csv(f'data/model_dependencies/{market}_ltv_expected.csv')
#
#         # set index to start at 1
#         self.ltv_expected.index = np.arange(1, len(self.ltv_expected)+1)
#
#         # initialize placeholders
#         self.min_months = None
#         self.default_stress = None
#         self.label_cols = None
#
#     # --- FORECAST FUNCTIONS --- #
#     def forecast_data(self, data, min_months=5, n_months=50, default_stress=None, weight_actuals=0.75, weight_tail=0.1):
#         """
#         Generates a forecast of "Count Borrowers" out to the input number of months.
#         The original and forecasted values are returned as a new dataframe, set as
#         a new attribute of the model, *.forecast*.
#
#         Parameters
#         ----------
#         data : pandas dataframe
#             Data to forecast. Usually will be self.data which is data that has already
#             been cleaned and processed.
#         min_months : int
#             The number of months of data a cohort must have in order to be forecast.
#             This limitation is to avoid the large errors incurred when forecasting
#             data for cohorts with few data points (<5).
#         n_months : int
#             Number of months to forecast to.
#         default_stress: float
#             If None, no default stress applied. If float, default stress is multiplied
#             times the 7dpd and 365dpd default rates to stress them.
#         """
#         self.min_months = min_months
#         self.default_stress = default_stress
#         self.dr_expectations = None
#
#         # range of desired time periods
#         times = np.arange(1, n_months+1)
#         times_dict = {i: i-1 for i in times}
#
#         # ----- Prepare Data ----- #
#         dfs = []
#         for cohort in data.cohort.unique():
#             # data for current cohort
#             c_data = data[data.cohort == cohort].copy()
#
#             # only for cohorts with at least min_months of data
#             if len(c_data) >= min_months:
#                 # null df used to extend original cohort df to desired number of forecast months
#                 dummy_df = pd.DataFrame(np.nan, index=times, columns=['null'])
#
#                 # create label column to denote actual vs forecast data
#                 c_data.loc[:, 'data_type'] = 'actual'
#
#                 # extend cohort df
#                 c_data = pd.concat([c_data, dummy_df], axis=1).drop('null', axis=1)
#                 # use cohort as df name
#                 c_data.name = cohort
#
#                 # fill missing values in each col
#                 c_data.cohort = c_data.cohort.ffill()
#                 c_data['First Loan Local Disbursement Month'] = \
#                     c_data['First Loan Local Disbursement Month'].ffill()
#                 c_data['Months Since First Loan Disbursed'] = \
#                     c_data['Months Since First Loan Disbursed'].fillna(times_dict).astype(int)
#
#                 # label forecasted data
#                 c_data.data_type = c_data.data_type.fillna('forecast')
#
#                 dfs.append(c_data)
#         data = pd.concat(dfs)
#
#         self.dr_expectations = {7: {'seed': self.ltv_expected['default_rate_7dpd_seed'],
#                                'expectations': []},
#                            51: {'seed': self.ltv_expected['default_rate_51dpd_seed'],
#                                 'expectations': []},
#                            365: {'seed': self.ltv_expected['default_rate_365dpd_seed'],
#                                 'expectations': []}
#                            }
#
#         def forecast_defaults(data, dpd=7, n_months=50, asymptote=None,
#                               weight_actuals=weight_actuals, weight_tail=weight_tail):
#             # set default rate name
#             default_rate = f'default_rate_{dpd}dpd'
#
#             fcasts = []
#             forecasted_dfs = []
#
#             for j, c in enumerate(data.cohort.unique()):
#                 # get current cohort data
#                 cohort_data = data[data.cohort == c].copy()
#                 cohort_data.name = c
#
#                 # create forecast array to store forecast to
#                 fcast = cohort_data[cohort_data.data_type == 'actual'][default_rate].copy()
#
#                 # if there are no other expectations, just use the seed
#                 if j == 0:
#                     expectation = self.dr_expectations[dpd]['seed'].iloc[:n_months].copy()
#                 # otherwise, use the last expectation
#                 else:
#                     expectation = self.dr_expectations[dpd]['expectations'][-1].copy()
#
#                 # if there are at least 5 data points, use smoothing
#                 if len(fcast) >= 5:
#                     fcast_smooth = savgol_filter(fcast, int((.8) * len(fcast)), 3)
#                     fcast = pd.Series(fcast_smooth, index=fcast.index)
#
#                 # set cohort name
#                 fcast.name = cohort_data.name
#
#                 for i_, i in enumerate(range(len(fcast) + 1, n_months + 1)):
#                     if len(fcast) < 5:
#                         s = np.mean(fcast.iloc[-1:])
#                     elif len(fcast >= 5):
#                         s = np.mean(fcast.iloc[-2:])
#
#                     # for the first point
#                     if i_ == 0:
#                         # add asymptotic condition
#                         if asymptote:
#                             if s + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                                 fcast.loc[i] = asymptote
#                             else:
#                                 fcast.loc[i] = s + (expectation.loc[i] - expectation.loc[i - 1])
#                         else:
#                             fcast.loc[i] = s + (expectation.loc[i] - expectation.loc[i - 1])
#                     # for all subsequent points
#                     else:
#                         # add asymptotic condition
#                         if asymptote:
#                             if fcast.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                                 fcast.loc[i] = asymptote
#                             else:
#                                 fcast.loc[i] = fcast.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1])
#                         else:
#                             fcast.loc[i] = fcast.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1])
#
#                 # if we're on the 1st cohort, use the forecast as the next expectation
#                 if j == 0:
#                     self.dr_expectations[dpd]['expectations'].append(fcast)
#
#                 # if we're on a subsequent cohort, modify the expectation with the current actuals.
#                 else:
#                     # get current idx of actuals
#                     n_actuals = len(cohort_data[cohort_data.data_type == 'actual'])
#
#                     # modify expectation curve
#                     modified_actuals = (fcast.loc[:n_actuals] * weight_actuals +
#                                         expectation.loc[:n_actuals]) / (1 + weight_actuals)
#
#                     modified_tail = (fcast.loc[n_actuals + 1:] * weight_tail +
#                                     expectation.loc[n_actuals + 1:]) / (1 + weight_tail)
#
#                     new_expectation = pd.concat([modified_actuals, modified_tail])
#
#                     # add new expectation to the list
#                     self.dr_expectations[dpd]['expectations'].append(new_expectation)
#
#                 cohort_data[default_rate] = cohort_data[default_rate].fillna(fcast)
#                 forecasted_dfs.append(cohort_data)
#
#             return pd.concat(forecasted_dfs)
#
#         # Forecast default rates
#         data = forecast_defaults(data=data, dpd=7, n_months=n_months, asymptote=0.0358,
#                                  weight_actuals=weight_actuals, weight_tail=weight_tail)
#         data = forecast_defaults(data=data, dpd=51, n_months=n_months, asymptote=0.0317,
#                                  weight_actuals=weight_actuals, weight_tail=weight_tail)
#         data = forecast_defaults(data=data, dpd=365, n_months=n_months, asymptote=0.0356,
#                                  weight_actuals=weight_actuals, weight_tail=weight_tail)
#
#         forecast_dfs = []
#         # ----- FORECAST BY COHORT ----- #
#         for cohort in data.cohort.unique():
#             # data for current cohort
#             c_data = data[data.cohort == cohort].copy()
#
#             # starting cohort size
#             n = c_data.loc[1, 'Count Borrowers']
#
#             def power_fcast(c_data, param='borrower_retention'):
#
#                 c = c_data[param].dropna()
#
#                 def power_fit(times, a, b):
#                     return a * np.array(times)**b
#
#                 # fit actuals and extract a & b params
#                 popt, pcov = curve_fit(power_fit, c.index, c)
#
#                 a = 1
#                 b = popt[1]
#
#                 # scale b according to market
#                 if self.market == 'ke':
#                     if len(c) < 6:
#                         b = b + .02 * (6 - len(c) - 1)
#                 if self.market == 'ph':
#                     if len(c) < 6:
#                         b = b + .02 * (6 - len(c) - 1)
#                 if self.market == 'mx':
#                     b = b - .015 * (18 - len(c) - 1)
#
#                 # get max survival from inputs
#                 max_survival = config['max_survival'][self.market]
#
#                 # take the slope of the power fit between the current and previous time periods
#                 # errstate handles division by 0 errors
#                 with np.errstate(divide='ignore'):
#                     shifted_fit = power_fit(times-1, a, b)
#                     shifted_fit[np.isinf(shifted_fit)] = 1
#                 power_slope = power_fit(times, a, b) / shifted_fit
#
#                 # apply max survival condition
#                 power_slope[power_slope > max_survival] = max_survival
#                 # only need values for times we're going to forecast for.
#                 power_slope = power_slope[len(c):]
#                 power_slope = pd.Series(power_slope, index=[t for t in times[len(c):]])
#
#                 c_fcast = c.copy()
#                 for t in times[len(c):]:
#                     c_fcast.loc[t] = c_fcast[t - 1] * power_slope[t]
#
#                 return c_fcast
#
#             forecast = power_fcast(c_data)
#             forecast.index = np.arange(1, len(c_data)+1)
#             # fill in the forecasted data
#             c_data['borrower_retention'] = c_data['borrower_retention'].fillna(forecast)
#
#             # compute Count Borrowers
#             fcast_count = []
#             for t in times:
#                 if t < len(c_data['Count Borrowers'].dropna()):
#                     fcast_count.append(c_data.loc[t, 'Count Borrowers'])
#                 else:
#                     fcast_count.append(n * forecast[t])
#
#             c_data['Count Borrowers'] = pd.Series(fcast_count, index=times)
#
#             # --- ALL OTHERS --- #
#             # compute survival
#             c_data['borrower_survival'] = borrower_survival(c_data)
#
#             # forecast loan size
#             for i in c_data[c_data.loan_size.isnull()].index:
#                 c_data.loc[i, 'loan_size'] = c_data.loc[i - 1, 'loan_size'] * \
#                                              self.ltv_expected.loc[i, 'loan_size'] / self.ltv_expected.loc[
#                                                  i - 1, 'loan_size']
#
#             # forecast loans_per_borrower
#             for i in c_data[c_data.loans_per_borrower.isnull()].index:
#                 c_data.loc[i, 'loans_per_borrower'] = self.ltv_expected.loc[i, 'loans_per_borrower']
#
#             # forecast Count Loans
#             c_data['Count Loans'] = c_data['Count Loans'].fillna(
#                 (c_data['loans_per_borrower']) * c_data['Count Borrowers'])
#
#             # forecast Total Amount
#             c_data['Total Amount'] = c_data['Total Amount'].fillna(
#                 (c_data['loan_size']) * c_data['Count Loans'])
#
#             # forecast Interest Rate
#             for i in c_data[c_data.interest_rate.isnull()].index:
#                 c_data.loc[i, 'interest_rate'] = c_data.loc[i - 1, 'interest_rate'] * \
#                                                  self.ltv_expected.loc[i, 'interest_rate'] / self.ltv_expected.loc[
#                                                      i - 1, 'interest_rate']
#
#             if self.default_stress:
#                 c_data['default_rate_7dpd'] += self.default_stress
#                 c_data['default_rate_365dpd'] += self.default_stress
#
#             # compute remaining columns from forecasts
#             c_data['loans_per_original'] = loans_per_original(c_data)
#             c_data['cumulative_loans_per_original'] = c_data['loans_per_original'].cumsum()
#             c_data['origination_per_original'] = origination_per_original(c_data)
#             c_data['cumulative_origination_per_original'] = c_data['origination_per_original'].cumsum()
#             c_data['revenue_per_original'] = revenue_per_original(c_data)
#             c_data['cumulative_revenue_per_original'] = c_data['revenue_per_original'].cumsum()
#             c_data['cm$_per_original'] = credit_margin(c_data)
#             c_data['cumulative_cm$_per_original'] = c_data['cm$_per_original'].cumsum()
#             c_data['opex_per_original'] = opex_per_original(c_data, self.market)
#             c_data['cumulative_opex_per_original'] = c_data['opex_per_original'].cumsum()
#             c_data['opex_coc_per_original'] = opex_coc_per_original(c_data, self.market)
#             c_data['cumulative_opex_coc_per_original'] = c_data['opex_coc_per_original'].cumsum()
#             c_data['opex_cpl_per_original'] = opex_cpl_per_original(c_data, self.market)
#             c_data['cumulative_opex_cpl_per_original'] = c_data['opex_cpl_per_original'].cumsum()
#             c_data['ltv_per_original'] = ltv_per_original(c_data)
#             c_data['cumulative_ltv_per_original'] = c_data['ltv_per_original'].cumsum()
#             c_data['dcf_ltv_per_original'] = dcf_ltv_per_original(c_data)
#             c_data['cumulative_dcf_ltv_per_original'] = c_data['dcf_ltv_per_original'].cumsum()
#             c_data['cm%_per_original'] = credit_margin_percent(c_data)
#
#             # add the forecasted data for the cohort to a list, aggregating all cohort forecasts
#             forecast_dfs.append(c_data)
#
#         forecast_df = pd.concat(forecast_dfs)
#
#         return forecast_df
#
#     def backtest_data(self, data, min_months=4, hold_months=4, fcast_months=50, metrics=None,
#                       weight_actuals=1.5, weight_tail=0.5):
#         """
#         Backtest forecasted values against actuals.
#
#         Parameters
#         ----------
#
#
#         """
#         self.label_cols = ['First Loan Local Disbursement Month', 'Total Interest Assessed', 'Total Rollover Charged',
#                            'Total Rollover Reversed', 'Months Since First Loan Disbursed', 'Default Rate Amount 7D',
#                            'Default Rate Amount 30D', 'Default Rate Amount 51D', 'cohort', 'data_type']
#         self.min_months = min_months
#
#         if metrics is None:
#             metrics = ['rmse', 'me', 'mape', 'mpe']
#         cohort_count = 0
#         for cohort in data.cohort.unique():
#             if len(data[data.cohort == cohort]) - hold_months >= self.min_months:
#                 cohort_count += 1
#
#         # print the number of cohorts that will be backtested.
#         self.backtest_months = hold_months
#         print(f'Backtesting {hold_months} months.')
#         print(f'{cohort_count} cohorts will be backtested.')
#
#         def compute_error(actual, forecast, metric):
#             """
#             Test forecast performance against actuals using method defined by metric.
#             """
#             # root mean squared error
#             if metric == 'rmse':
#                 error = np.sqrt((1 / len(actual)) * sum((forecast[:len(actual)] - actual) ** 2))
#             # mean absolute error
#             elif metric == 'mae':
#                 error = np.mean(abs(forecast[:len(actual)] - actual))
#             # mean error
#             elif metric == 'me':
#                 error = np.mean(forecast[:len(actual)] - actual)
#             # mean absolute percent error
#             elif metric == 'mape':
#                 error = round((1 / len(actual)) * sum(abs((forecast[:len(actual)] - actual) / actual)), 4)
#             # mean percent error
#             elif metric == 'mpe':
#                 error = round((1 / len(actual)) * sum((forecast[:len(actual)] - actual) / actual), 4)
#             return error
#
#         # --- Generate backtest data --- #
#         backtest_report = []
#         backtest_data = []
#
#         # limit cohorts by min_months and actuals by hold_months
#         for cohort in data.cohort.unique():
#             # data for current cohort
#             c_data = data[data.cohort == cohort].copy()
#
#             # only backtest if remaining data has at least min_months of data
#             if len(c_data) - hold_months >= self.min_months:
#                 # limit data
#                 c_data = c_data.iloc[:len(c_data) - hold_months, :]
#                 backtest_data.append(c_data)
#
#         backtest_data = pd.concat(backtest_data)
#
#         # create forecast on limited dataset
#         backtest = self.forecast_data(backtest_data, min_months=min_months, n_months=fcast_months,
#                                       weight_actuals=weight_actuals, weight_tail=weight_tail)
#
#         for cohort in backtest.cohort.unique():
#             # get forecast overlap with actuals
#             actual = self.data[self.data['First Loan Local Disbursement Month'] == cohort]
#             predicted = backtest[backtest.cohort == cohort]
#
#             start = backtest[backtest.data_type == 'forecast'].index.min()
#             stop = actual.index.max()
#
#             # compute errors
#             backtest_report_cols = []
#             errors = []
#
#             cols = [c for c in self.data.columns if c not in self.label_cols]
#             cols.remove('Count First Loans')
#
#             for col in cols:
#                 for metric in metrics:
#                     error = compute_error(actual.loc[start:stop, col], predicted.loc[start:stop, col],
#                                           metric=metric)
#
#                     backtest_report_cols += [f'{col}-{metric}']
#
#                     errors.append(error)
#
#             backtest_report.append(pd.DataFrame.from_dict({cohort: errors}, orient='index',
#                                                           columns=backtest_report_cols))
#
#         backtest_report = pd.concat(backtest_report, axis=0)
#         backtest_report['cohort'] = backtest_report.index
#
#         return backtest, backtest_report


#
# class Rolling3(DataManager):
#     """
#     The Rolling model uses the same methodology as the PowerSlope model for all parameters
#     except default rates. Default rates in the Rolling model are generated from rolling
#     expectations. A seed expectation curve generated from historicals is used to forecast
#     the first cohort. Each subsequent cohort uses a weighted average of the last n_trail
#     cohort forecasts as its expectation curve.
#
#     Parameters
#     ----------
#     data : pandas dataframe
#         Data to forecast. Usually will be self.data which is data that has already
#         been cleaned and processed.
#     market : str
#         The market the data corresponds to (KE, PH, MX, etc.).
#     to_usd : bool
#         If True, convert fields in local currency to USD. If False, leave fields as
#         local currency.
#     bake_duration : int
#         Number of months to consider data fully baked. The last bake_duration number
#         of months is removed from the data during cleaning.
#     """
#     def __init__(self, data, market, to_usd=True, bake_duration=4):
#         """
#         Sets model attributes, loads additional data required for models (inputs &
#         ltv_expected), and cleans data.
#
#         Parameters
#         ----------
#         data : pandas dataframe
#             Data to forecast. Usually will be self.data which is data that has already
#             been cleaned and processed.
#         market : str
#             The market the data corresponds to (KE, PH, MX, etc.).
#         to_usd : bool
#             If True, convert fields in local currency to USD. If False, leave fields as
#             local currency.
#         bake_duration : int
#             Number of months to consider data fully baked. The last bake_duration number
#             of months is removed from the data during cleaning.
#         """
#         super().__init__(data, market, to_usd, bake_duration)
#
#         # read in expectations
#         self.ltv_expected = pd.read_csv(f'data/model_dependencies/{market}_ltv_expected.csv')
#         # set index to start at 1
#         self.ltv_expected.index = np.arange(1, len(self.ltv_expected)+1)
#
#         # initialize placeholders
#         self.min_months = None
#         self.default_stress = None
#         self.label_cols = None
#
#     # --- FORECAST FUNCTIONS --- #
#     def forecast_data(self, data, min_months=5, n_months=50, default_stress=None, weights=[1,1,1]):
#         """
#         Generates a forecast of "Count Borrowers" out to the input number of months.
#         The original and forecasted values are returned as a new dataframe, set as
#         a new attribute of the model, *.forecast*.
#
#         Parameters
#         ----------
#         data : pandas dataframe
#             Data to forecast. Usually will be self.data which is data that has already
#             been cleaned and processed.
#         min_months : int
#             The number of months of data a cohort must have in order to be forecast.
#             This limitation is to avoid the large errors incurred when forecasting
#             data for cohorts with few data points (<5).
#         n_months : int
#             Number of months to forecast to.
#         default_stress: float
#             If None, no default stress applied. If float, default stress is multiplied
#             times the 7dpd and 365dpd default rates to stress them.
#         """
#         self.min_months = min_months
#         self.default_stress = default_stress
#
#         # list to hold individual cohort forecasts
#         forecast_dfs = []
#
#         # range of desired time periods
#         times = np.arange(1, n_months+1)
#         times_dict = {i: i-1 for i in times}
#
#         # optional
#         dr_expectations = {7: {'seed': self.ltv_expected['default_rate_7dpd_seed'],
#                                'expectations': []},
#                            51: {'seed': self.ltv_expected['default_rate_51dpd_seed'],
#                                 'expectations': []},
#                            365: {'seed': self.ltv_expected['default_rate_365dpd_seed'],
#                                 'expectations': []}
#                            }
#         smooth_curves = []
#
#         def forecast_defaults(cohort_data, dpd=7, n_months=50, asymptote=None, weights=weights):
#             n_trail = len(weights)
#             default_rate = f'default_rate_{dpd}dpd'
#
#             # if there are no other expectations, just use the seed
#             if len(dr_expectations[dpd]['expectations']) == 0:
#                 expectation = dr_expectations[dpd]['seed'].iloc[:n_months].copy()
#
#             elif len(dr_expectations[dpd]['expectations']) != 0:
#                 # start the weighted sum with the seed
#                 expectation_sum = dr_expectations[dpd]['seed'].iloc[:n_months].copy()
#                 # expectation_sum = pd.Series(np.zeros(n_months), index=times)
#
#                 # add in the last n_trail cohorts
#                 last_expectations = pd.concat(dr_expectations[dpd]['expectations'][-n_trail:], axis=1).T
#                 for i in range(len(last_expectations)-1, -1, -1):
#                     expectation_sum = expectation_sum + last_expectations.iloc[i] * weights[i]
#
#                 # the current expectation is the weighted average of the last n_trail expectations + seed
#                 expectation = expectation_sum / (sum(weights[-len(last_expectations):])+1)
#
#             fcast = cohort_data[cohort_data.data_type == 'actual'][default_rate].copy()
#
#             # forecast current cohort
#             # if there are at least 5 data points, use smoothing
#             if len(fcast) >= 5:
#                 fcast_smooth = savgol_filter(fcast, int((0.6) * len(fcast)), 2)
#                 fcast_smooth = pd.Series(fcast_smooth, index=fcast.index)
#                 fcast_smooth.name = cohort_data.name
#
#                 smooth_curves.append(fcast_smooth)
#
#                 for i_, i in enumerate(range(len(fcast) + 1, n_months + 1)):
#                     # for the first point
#                     if i_ == 0:
#                         # use mean of last 3 data points for starting value
#                         s = np.mean(fcast_smooth.iloc[-2:])
#
#                         # add asymptotic condition
#                         if asymptote:
#                             if fcast_smooth.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                                 fcast_smooth.loc[i] = fcast.loc[i] = asymptote
#                             else:
#                                 fcast_smooth.loc[i] = fcast.loc[i] = fcast_smooth.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1])
#                         else:
#                             fcast_smooth.loc[i] = fcast.loc[i] = fcast_smooth.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1])
#                     else:
#                         # add asymptotic condition
#                         if asymptote:
#                             if fcast_smooth.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                                 fcast_smooth.loc[i] = fcast.loc[i] = asymptote
#                             else:
#                                 fcast_smooth.loc[i] = fcast.loc[i] = fcast_smooth.loc[i - 1] + \
#                                                                      (expectation.loc[i] - expectation.loc[i - 1])
#                         else:
#                             fcast_smooth.loc[i] = fcast.loc[i] = fcast_smooth.loc[i - 1] + \
#                                                                  (expectation.loc[i] - expectation.loc[i - 1])
#
#                 dr_expectations[dpd]['expectations'].append(fcast_smooth)
#
#             # if there are less than 5 data points, don't smooth
#             else:
#                 for i_, i in enumerate(range(len(fcast) + 1, n_months + 1)):
#                     # for the first point
#                     if i_ == 0:
#                         # use mean of last 3 data points for starting value
#                         s = np.mean(fcast.iloc[-1:])
#
#                         # add asymptotic condition
#                         if asymptote:
#                             if fcast.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                                 fcast.loc[i] = asymptote
#                             else:
#                                 fcast.loc[i] = fcast.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1])
#                         else:
#                             fcast.loc[i] = fcast.loc[i-1] + (expectation.loc[i] - expectation.loc[i - 1])
#                     else:
#                         # add asymptotic condition
#                         if asymptote:
#                             if fcast.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                                 fcast.loc[i] = asymptote
#                             else:
#                                 fcast.loc[i] = fcast.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1])
#                         else:
#                             fcast.loc[i] = fcast.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1])
#
#                 # add new forecast to expectations
#                 dr_expectations[dpd]['expectations'].append(fcast)
#
#             cohort_data[default_rate] = cohort_data[default_rate].fillna(fcast)
#
#             return cohort_data[default_rate]
#
#         # ----- FORECAST BY COHORT ----- #
#         for cohort in data.cohort.unique():
#             # data for current cohort
#             c_data = data[data.cohort == cohort].copy()
#
#             # starting cohort size
#             n = c_data.loc[1, 'Count Borrowers']
#             n_valid = len(c_data)
#
#             # only for cohorts with at least 4 data points
#             if len(c_data) >= min_months:
#
#                 # null df used to extend original cohort df to desired number of forecast months
#                 dummy_df = pd.DataFrame(np.nan, index=times, columns=['null'])
#
#                 # create label column to denote actual vs forecast data
#                 c_data.loc[:, 'data_type'] = 'actual'
#
#                 # extend cohort df
#                 c_data = pd.concat([c_data, dummy_df], axis=1).drop('null', axis=1)
#                 # give cohort name
#                 c_data.name = cohort
#
#                 # fill missing values in each col
#                 c_data.cohort = c_data.cohort.ffill()
#                 c_data['First Loan Local Disbursement Month'] = \
#                     c_data['First Loan Local Disbursement Month'].ffill()
#                 c_data['Months Since First Loan Disbursed'] = \
#                     c_data['Months Since First Loan Disbursed'].fillna(times_dict).astype(int)
#
#                 # label forecasted data
#                 c_data.data_type = c_data.data_type.fillna('forecast')
#
#                 def power_fcast(c_data, param='borrower_retention'):
#
#                     c = c_data[param].dropna()
#
#                     def power_fit(times, a, b):
#                         return a * np.array(times)**b
#
#                     # fit actuals and extract a & b params
#                     popt, pcov = curve_fit(power_fit, c.index, c)
#
#                     a = 1
#                     b = popt[1]
#
#                     # scale b according to market
#                     if self.market == 'ke':
#                         if len(c) < 6:
#                             b = b + .02 * (6 - len(c) - 1)
#                     if self.market == 'ph':
#                         if len(c) < 6:
#                             b = b + .02 * (6 - len(c) - 1)
#                     if self.market == 'mx':
#                         b = b - .015 * (18 - len(c) - 1)
#
#                     # get max survival from inputs
#                     max_survival = config['max_survival'][self.market]
#
#                     # take the slope of the power fit between the current and previous time periods
#                     # errstate handles division by 0 errors
#                     with np.errstate(divide='ignore'):
#                         shifted_fit = power_fit(times-1, a, b)
#                         shifted_fit[np.isinf(shifted_fit)] = 1
#                     power_slope = power_fit(times, a, b) / shifted_fit
#
#                     # apply max survival condition
#                     power_slope[power_slope > max_survival] = max_survival
#                     # only need values for times we're going to forecast for.
#                     power_slope = power_slope[len(c):]
#                     power_slope = pd.Series(power_slope, index=[t for t in times[len(c):]])
#
#                     c_fcast = c.copy()
#                     for t in times[len(c):]:
#                         c_fcast.loc[t] = c_fcast[t - 1] * power_slope[t]
#
#                     return c_fcast
#
#                 forecast = power_fcast(c_data)
#                 forecast.index = np.arange(1, len(c_data)+1)
#                 # fill in the forecasted data
#                 c_data['borrower_retention'] = c_data['borrower_retention'].fillna(forecast)
#
#                 # compute Count Borrowers
#                 fcast_count = []
#                 for t in times:
#                     if t < len(c_data['Count Borrowers'].dropna()):
#                         fcast_count.append(c_data.loc[t, 'Count Borrowers'])
#                     else:
#                         fcast_count.append(n * forecast[t])
#
#                 c_data['Count Borrowers'] = pd.Series(fcast_count, index=times)
#
#                 # --- ALL OTHERS --- #
#                 # compute survival
#                 c_data['borrower_survival'] = borrower_survival(c_data)
#
#                 # forecast loan size
#                 for i in c_data[c_data.loan_size.isnull()].index:
#                     c_data.loc[i, 'loan_size'] = c_data.loc[i - 1, 'loan_size'] * \
#                                                  self.ltv_expected.loc[i, 'loan_size'] / self.ltv_expected.loc[
#                                                      i - 1, 'loan_size']
#
#                 # forecast loans_per_borrower
#                 for i in c_data[c_data.loans_per_borrower.isnull()].index:
#                     c_data.loc[i, 'loans_per_borrower'] = self.ltv_expected.loc[i, 'loans_per_borrower']
#
#                 # forecast Count Loans
#                 c_data['Count Loans'] = c_data['Count Loans'].fillna(
#                     (c_data['loans_per_borrower']) * c_data['Count Borrowers'])
#
#                 # forecast Total Amount
#                 c_data['Total Amount'] = c_data['Total Amount'].fillna(
#                     (c_data['loan_size']) * c_data['Count Loans'])
#
#                 # forecast Interest Rate
#                 for i in c_data[c_data.interest_rate.isnull()].index:
#                     c_data.loc[i, 'interest_rate'] = c_data.loc[i - 1, 'interest_rate'] * \
#                                                      self.ltv_expected.loc[i, 'interest_rate'] / self.ltv_expected.loc[
#                                                          i - 1, 'interest_rate']
#
#                 # Forecast default rates
#                 c_data['default_rate_7dpd'] = forecast_defaults(c_data, dpd=7, n_months=n_months, asymptote=0.0358,
#                                                                 weights=weights)
#                 c_data['default_rate_51dpd'] = forecast_defaults(c_data, dpd=51, n_months=n_months, asymptote=None,
#                                                                  weights=weights)
#                 c_data['default_rate_365dpd'] = forecast_defaults(c_data, dpd=365, n_months=n_months, asymptote=0.0356,
#                                                                   weights=weights)
#
#                 if self.default_stress:
#                     c_data['default_rate_7dpd'] += self.default_stress
#                     c_data['default_rate_365dpd'] += self.default_stress
#
#                 # compute remaining columns from forecasts
#                 c_data['loans_per_original'] = loans_per_original(c_data)
#                 c_data['cumulative_loans_per_original'] = c_data['loans_per_original'].cumsum()
#                 c_data['origination_per_original'] = origination_per_original(c_data)
#                 c_data['cumulative_origination_per_original'] = c_data['origination_per_original'].cumsum()
#                 c_data['revenue_per_original'] = revenue_per_original(c_data)
#                 c_data['cumulative_revenue_per_original'] = c_data['revenue_per_original'].cumsum()
#                 c_data['cm$_per_original'] = credit_margin(c_data)
#                 c_data['cumulative_cm$_per_original'] = c_data['cm$_per_original'].cumsum()
#                 c_data['opex_per_original'] = opex_per_original(c_data, self.market)
#                 c_data['cumulative_opex_per_original'] = c_data['opex_per_original'].cumsum()
#                 c_data['opex_coc_per_original'] = opex_coc_per_original(c_data, self.market)
#                 c_data['cumulative_opex_coc_per_original'] = c_data['opex_coc_per_original'].cumsum()
#                 c_data['opex_cpl_per_original'] = opex_cpl_per_original(c_data, self.market)
#                 c_data['cumulative_opex_cpl_per_original'] = c_data['opex_cpl_per_original'].cumsum()
#                 c_data['ltv_per_original'] = ltv_per_original(c_data)
#                 c_data['cumulative_ltv_per_original'] = c_data['ltv_per_original'].cumsum()
#                 c_data['dcf_ltv_per_original'] = dcf_ltv_per_original(c_data)
#                 c_data['cumulative_dcf_ltv_per_original'] = c_data['dcf_ltv_per_original'].cumsum()
#                 c_data['cm%_per_original'] = credit_margin_percent(c_data)
#
#                 # add the forecasted data for the cohort to a list, aggregating all cohort forecasts
#                 forecast_dfs.append(c_data)
#
#         forecast_df = pd.concat(forecast_dfs)
#
#         return forecast_df
#
#     def backtest_data(self, data, min_months=4, hold_months=4, fcast_months=50, metrics=None, weights=[1,1,1]):
#         """
#         Backtest forecasted values against actuals.
#
#         Parameters
#         ----------
#
#
#         """
#         self.label_cols = ['First Loan Local Disbursement Month', 'Total Interest Assessed', 'Total Rollover Charged',
#                            'Total Rollover Reversed', 'Months Since First Loan Disbursed', 'Default Rate Amount 7D',
#                            'Default Rate Amount 30D', 'Default Rate Amount 51D', 'cohort', 'data_type']
#
#         if metrics is None:
#             metrics = ['rmse', 'me', 'mape', 'mpe']
#         cohort_count = 0
#         for cohort in data.cohort.unique():
#             if len(data[data.cohort == cohort]) - hold_months >= self.min_months:
#                 cohort_count += 1
#
#         # print the number of cohorts that will be backtested.
#         self.backtest_months = hold_months
#         print(f'Backtesting {hold_months} months.')
#         print(f'{cohort_count} cohorts will be backtested.')
#
#         def compute_error(actual, forecast, metric):
#             """
#             Test forecast performance against actuals using method defined by metric.
#             """
#             # root mean squared error
#             if metric == 'rmse':
#                 error = np.sqrt((1 / len(actual)) * sum((forecast[:len(actual)] - actual) ** 2))
#             # mean absolute error
#             elif metric == 'mae':
#                 error = np.mean(abs(forecast[:len(actual)] - actual))
#             # mean error
#             elif metric == 'me':
#                 error = np.mean(forecast[:len(actual)] - actual)
#             # mean absolute percent error
#             elif metric == 'mape':
#                 error = round((1 / len(actual)) * sum(abs((forecast[:len(actual)] - actual) / actual)), 4)
#             # mean percent error
#             elif metric == 'mpe':
#                 error = round((1 / len(actual)) * sum((forecast[:len(actual)] - actual) / actual), 4)
#             return error
#
#         # --- Generate backtest data --- #
#         backtest_report = []
#         backtest_data = []
#
#         # limit cohorts by min_months and actuals by hold_months
#         for cohort in data.cohort.unique():
#             # data for current cohort
#             c_data = data[data.cohort == cohort].copy()
#
#             # only backtest if remaining data has at least min_months of data
#             if len(c_data) - hold_months >= self.min_months:
#                 # limit data
#                 c_data = c_data.iloc[:len(c_data) - hold_months, :]
#                 backtest_data.append(c_data)
#
#         backtest_data = pd.concat(backtest_data)
#
#         # create forecast on limited dataset
#         backtest = self.forecast_data(backtest_data, min_months=min_months, n_months=fcast_months, weights=weights)
#
#         for cohort in backtest.cohort.unique():
#             # get forecast overlap with actuals
#             actual = self.data[self.data['First Loan Local Disbursement Month'] == cohort]
#             predicted = backtest[backtest.cohort == cohort]
#
#             start = backtest[backtest.data_type == 'forecast'].index.min()
#             stop = actual.index.max()
#
#             # compute errors
#             backtest_report_cols = []
#             errors = []
#
#             cols = [c for c in self.data.columns if c not in self.label_cols]
#             cols.remove('Count First Loans')
#
#             for col in cols:
#                 for metric in metrics:
#                     error = compute_error(actual.loc[start:stop, col], predicted.loc[start:stop, col],
#                                           metric=metric)
#
#                     backtest_report_cols += [f'{col}-{metric}']
#
#                     errors.append(error)
#
#             backtest_report.append(pd.DataFrame.from_dict({cohort: errors}, orient='index',
#                                                           columns=backtest_report_cols))
#
#         backtest_report = pd.concat(backtest_report, axis=0)
#         backtest_report['cohort'] = backtest_report.index
#
#         return backtest, backtest_report


# class Rolling2(DataManager):
#     """
#     The Rolling model uses the same methodology as the PowerSlope model for all parameters
#     except default rates. Default rates in the Rolling model are generated from rolling
#     expectations. A seed expectation curve generated from historicals is used to forecast
#     the first cohort. Each subsequent cohort uses a weighted average of the last n_trail
#     cohort forecasts as its expectation curve.
#
#     Parameters
#     ----------
#     data : pandas dataframe
#         Data to forecast. Usually will be self.data which is data that has already
#         been cleaned and processed.
#     market : str
#         The market the data corresponds to (KE, PH, MX, etc.).
#     to_usd : bool
#         If True, convert fields in local currency to USD. If False, leave fields as
#         local currency.
#     bake_duration : int
#         Number of months to consider data fully baked. The last bake_duration number
#         of months is removed from the data during cleaning.
#     """
#     def __init__(self, data, market, to_usd=True, bake_duration=4):
#         """
#         Sets model attributes, loads additional data required for models (inputs &
#         ltv_expected), and cleans data.
#
#         Parameters
#         ----------
#         data : pandas dataframe
#             Data to forecast. Usually will be self.data which is data that has already
#             been cleaned and processed.
#         market : str
#             The market the data corresponds to (KE, PH, MX, etc.).
#         to_usd : bool
#             If True, convert fields in local currency to USD. If False, leave fields as
#             local currency.
#         bake_duration : int
#             Number of months to consider data fully baked. The last bake_duration number
#             of months is removed from the data during cleaning.
#         """
#         super().__init__(data, market, to_usd, bake_duration)
#
#         # read in expectations
#         self.ltv_expected = pd.read_csv(f'data/model_dependencies/{market}_ltv_expected.csv')
#         # set index to start at 1
#         self.ltv_expected.index = np.arange(1, len(self.ltv_expected)+1)
#
#         # initialize placeholders
#         self.min_months = None
#         self.default_stress = None
#         self.label_cols = None
#
#     # --- FORECAST FUNCTIONS --- #
#     def forecast_data(self, data, min_months=5, n_months=50, default_stress=None):
#         """
#         Generates a forecast of "Count Borrowers" out to the input number of months.
#         The original and forecasted values are returned as a new dataframe, set as
#         a new attribute of the model, *.forecast*.
#
#         Parameters
#         ----------
#         data : pandas dataframe
#             Data to forecast. Usually will be self.data which is data that has already
#             been cleaned and processed.
#         min_months : int
#             The number of months of data a cohort must have in order to be forecast.
#             This limitation is to avoid the large errors incurred when forecasting
#             data for cohorts with few data points (<5).
#         n_months : int
#             Number of months to forecast to.
#         default_stress: float
#             If None, no default stress applied. If float, default stress is multiplied
#             times the 7dpd and 365dpd default rates to stress them.
#         """
#         self.min_months = min_months
#         self.default_stress = default_stress
#
#         # list to hold individual cohort forecasts
#         forecast_dfs = []
#
#         # range of desired time periods
#         times = np.arange(1, n_months+1)
#         times_dict = {i: i-1 for i in times}
#
#         # optional
#         dr_expectations = {7: {'seed': self.ltv_expected['default_rate_7dpd_seed'],
#                                'expectations': []},
#                            51: {'seed': self.ltv_expected['default_rate_51dpd_seed'],
#                                 'expectations': []},
#                            365: {'seed': self.ltv_expected['default_rate_365dpd_seed'],
#                                 'expectations': []}
#                            }
#         smooth_curves = []
#
#         def forecast_defaults(cohort_data, dpd=7, n_months=50, asymptote=0.045, weights=(1, 0.75, 0.5)):
#             n_trail = len(weights)
#             default_rate = f'default_rate_{dpd}dpd'
#
#             # if there are no other expectations, just use the seed
#             if len(dr_expectations[dpd]['expectations']) == 0:
#                 expectation = dr_expectations[dpd]['seed']
#
#             elif len(dr_expectations[dpd]['expectations']) != 0:
#                 # start the weighted sum with the seed
#                 expectation_sum = dr_expectations[dpd]['seed'].copy()
#
#                 # add in the last n_trail cohorts
#                 last_expectations = pd.concat(dr_expectations[dpd]['expectations'][-n_trail:], axis=1).T
#                 for i in range(0, len(last_expectations)):
#                     expectation_sum += last_expectations.iloc[i] * weights[i]
#
#                 # the current expectation is the weighted average of the last n_trail expectations + seed
#                 expectation = expectation_sum / (sum(weights[:len(last_expectations)]) + 1)
#
#             fcast = cohort_data[cohort_data.data_type == 'actual'][default_rate].copy()
#
#             # forecast current cohort
#             # if there are at least 5 data points, use smoothing
#             if len(fcast) >= 5:
#                 fcast_smooth = savgol_filter(fcast, int((0.6) * len(fcast)), 2)
#                 fcast_smooth = pd.Series(fcast_smooth, index=fcast.index)
#                 fcast_smooth.name = cohort_data.name
#
#                 smooth_curves.append(fcast_smooth)
#
#                 for i in range(len(fcast) + 1, n_months + 1):
#                     # add asymptotic condition
#                     if asymptote:
#                         if fcast_smooth.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                             fcast_smooth.loc[i] = fcast.loc[i] = asymptote
#                         else:
#                             fcast_smooth.loc[i] = fcast.loc[i] = fcast_smooth.loc[i - 1] + \
#                                                                  (expectation.loc[i] - expectation.loc[i - 1])
#                     else:
#                         fcast_smooth.loc[i] = fcast.loc[i] = fcast_smooth.loc[i - 1] + \
#                                                              (expectation.loc[i] - expectation.loc[i - 1])
#
#                 dr_expectations[dpd]['expectations'].append(fcast_smooth)
#
#             # if there are less than 5 data points, don't smooth
#             else:
#
#                 for i in range(len(fcast) + 1, n_months + 1):
#                     # add asymptotic condition
#                     if asymptote:
#                         if fcast.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1]) < asymptote:
#                             fcast.loc[i] = asymptote
#                         else:
#                             fcast.loc[i] = fcast.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1])
#                     else:
#                         fcast.loc[i] = fcast.loc[i - 1] + (expectation.loc[i] - expectation.loc[i - 1])
#
#                 # add new forecast to expectations
#                 dr_expectations[dpd]['expectations'].append(fcast)
#
#             cohort_data[default_rate] = cohort_data[default_rate].fillna(fcast)
#
#             return cohort_data[default_rate]
#
#         # ----- FORECAST BY COHORT ----- #
#         for cohort in data.cohort.unique():
#             # data for current cohort
#             c_data = data[data.cohort == cohort].copy()
#
#             # starting cohort size
#             n = c_data.loc[1, 'Count Borrowers']
#             n_valid = len(c_data)
#
#             # only for cohorts with at least 4 data points
#             if len(c_data) >= min_months:
#
#                 # null df used to extend original cohort df to desired number of forecast months
#                 dummy_df = pd.DataFrame(np.nan, index=times, columns=['null'])
#
#                 # create label column to denote actual vs forecast data
#                 c_data.loc[:, 'data_type'] = 'actual'
#
#                 # extend cohort df
#                 c_data = pd.concat([c_data, dummy_df], axis=1).drop('null', axis=1)
#                 # give cohort name
#                 c_data.name = cohort
#
#                 # fill missing values in each col
#                 c_data.cohort = c_data.cohort.ffill()
#                 c_data['First Loan Local Disbursement Month'] = \
#                     c_data['First Loan Local Disbursement Month'].ffill()
#                 c_data['Months Since First Loan Disbursed'] = \
#                     c_data['Months Since First Loan Disbursed'].fillna(times_dict).astype(int)
#
#                 # label forecasted data
#                 c_data.data_type = c_data.data_type.fillna('forecast')
#
#                 def power_fcast(c_data, param='borrower_retention'):
#
#                     c = c_data[param].dropna()
#
#                     def power_fit(times, a, b):
#                         return a * np.array(times)**b
#
#                     # fit actuals and extract a & b params
#                     popt, pcov = curve_fit(power_fit, c.index, c)
#
#                     a = 1
#                     b = popt[1]
#
#                     # scale b according to market
#                     if self.market == 'ke':
#                         if len(c) < 6:
#                             b = b + .02 * (6 - len(c) - 1)
#                     if self.market == 'ph':
#                         if len(c) < 6:
#                             b = b + .02 * (6 - len(c) - 1)
#                     if self.market == 'mx':
#                         b = b - .015 * (18 - len(c) - 1)
#
#                     # get max survival from inputs
#                     max_survival = config['max_survival'][self.market]
#
#                     # take the slope of the power fit between the current and previous time periods
#                     # errstate handles division by 0 errors
#                     with np.errstate(divide='ignore'):
#                         shifted_fit = power_fit(times-1, a, b)
#                         shifted_fit[np.isinf(shifted_fit)] = 1
#                     power_slope = power_fit(times, a, b) / shifted_fit
#
#                     # apply max survival condition
#                     power_slope[power_slope > max_survival] = max_survival
#                     # only need values for times we're going to forecast for.
#                     power_slope = power_slope[len(c):]
#                     power_slope = pd.Series(power_slope, index=[t for t in times[len(c):]])
#
#                     c_fcast = c.copy()
#                     for t in times[len(c):]:
#                         c_fcast.loc[t] = c_fcast[t - 1] * power_slope[t]
#
#                     return c_fcast
#
#                 forecast = power_fcast(c_data)
#                 forecast.index = np.arange(1, len(c_data)+1)
#                 # fill in the forecasted data
#                 c_data['borrower_retention'] = c_data['borrower_retention'].fillna(forecast)
#
#                 # compute Count Borrowers
#                 fcast_count = []
#                 for t in times:
#                     if t < len(c_data['Count Borrowers'].dropna()):
#                         fcast_count.append(c_data.loc[t, 'Count Borrowers'])
#                     else:
#                         fcast_count.append(n * forecast[t])
#
#                 c_data['Count Borrowers'] = pd.Series(fcast_count, index=times)
#
#                 # --- ALL OTHERS --- #
#                 # compute survival
#                 c_data['borrower_survival'] = borrower_survival(c_data)
#
#                 # forecast loan size
#                 for i in c_data[c_data.loan_size.isnull()].index:
#                     c_data.loc[i, 'loan_size'] = c_data.loc[i - 1, 'loan_size'] * \
#                                                  self.ltv_expected.loc[i, 'loan_size'] / self.ltv_expected.loc[
#                                                      i - 1, 'loan_size']
#
#                 # forecast loans_per_borrower
#                 for i in c_data[c_data.loans_per_borrower.isnull()].index:
#                     c_data.loc[i, 'loans_per_borrower'] = self.ltv_expected.loc[i, 'loans_per_borrower']
#
#                 # forecast Count Loans
#                 c_data['Count Loans'] = c_data['Count Loans'].fillna(
#                     (c_data['loans_per_borrower']) * c_data['Count Borrowers'])
#
#                 # forecast Total Amount
#                 c_data['Total Amount'] = c_data['Total Amount'].fillna(
#                     (c_data['loan_size']) * c_data['Count Loans'])
#
#                 # forecast Interest Rate
#                 for i in c_data[c_data.interest_rate.isnull()].index:
#                     c_data.loc[i, 'interest_rate'] = c_data.loc[i - 1, 'interest_rate'] * \
#                                                      self.ltv_expected.loc[i, 'interest_rate'] / self.ltv_expected.loc[
#                                                          i - 1, 'interest_rate']
#
#                 # Forecast default rates
#                 c_data['default_rate_7dpd'] = forecast_defaults(c_data, dpd=7, n_months=n_months, asymptote=0.045,
#                                                     weights=(0.5, 0.75, 1))
#                 c_data['default_rate_51dpd'] = forecast_defaults(c_data, dpd=51)
#                 c_data['default_rate_365dpd'] = forecast_defaults(c_data, dpd=365)
#
#                 if self.default_stress:
#                     c_data['default_rate_7dpd'] += self.default_stress
#                     c_data['default_rate_365dpd'] += self.default_stress
#
#                 # compute remaining columns from forecasts
#                 c_data['loans_per_original'] = loans_per_original(c_data)
#                 c_data['cumulative_loans_per_original'] = c_data['loans_per_original'].cumsum()
#                 c_data['origination_per_original'] = origination_per_original(c_data)
#                 c_data['cumulative_origination_per_original'] = c_data['origination_per_original'].cumsum()
#                 c_data['revenue_per_original'] = revenue_per_original(c_data)
#                 c_data['cumulative_revenue_per_original'] = c_data['revenue_per_original'].cumsum()
#                 c_data['cm$_per_original'] = credit_margin(c_data)
#                 c_data['cumulative_cm$_per_original'] = c_data['cm$_per_original'].cumsum()
#                 c_data['opex_per_original'] = opex_per_original(c_data, self.market)
#                 c_data['cumulative_opex_per_original'] = c_data['opex_per_original'].cumsum()
#                 c_data['opex_coc_per_original'] = opex_coc_per_original(c_data, self.market)
#                 c_data['cumulative_opex_coc_per_original'] = c_data['opex_coc_per_original'].cumsum()
#                 c_data['opex_cpl_per_original'] = opex_cpl_per_original(c_data, self.market)
#                 c_data['cumulative_opex_cpl_per_original'] = c_data['opex_cpl_per_original'].cumsum()
#                 c_data['ltv_per_original'] = ltv_per_original(c_data)
#                 c_data['cumulative_ltv_per_original'] = c_data['ltv_per_original'].cumsum()
#                 c_data['dcf_ltv_per_original'] = dcf_ltv_per_original(c_data)
#                 c_data['cumulative_dcf_ltv_per_original'] = c_data['dcf_ltv_per_original'].cumsum()
#                 c_data['cm%_per_original'] = credit_margin_percent(c_data)
#
#                 # add the forecasted data for the cohort to a list, aggregating all cohort forecasts
#                 forecast_dfs.append(c_data)
#
#         forecast_df = pd.concat(forecast_dfs)
#
#         return forecast_df
#
#     def backtest_data(self, data, min_months=4, hold_months=4, fcast_months=50, metrics=None):
#         """
#         Backtest forecasted values against actuals.
#
#         Parameters
#         ----------
#
#
#         """
#         self.label_cols = ['First Loan Local Disbursement Month', 'Total Interest Assessed', 'Total Rollover Charged',
#                            'Total Rollover Reversed', 'Months Since First Loan Disbursed', 'Default Rate Amount 7D',
#                            'Default Rate Amount 30D', 'Default Rate Amount 51D', 'cohort', 'data_type']
#
#         if metrics is None:
#             metrics = ['rmse', 'me', 'mape', 'mpe']
#         cohort_count = 0
#         for cohort in data.cohort.unique():
#             if len(data[data.cohort == cohort]) - hold_months >= self.min_months:
#                 cohort_count += 1
#
#         # print the number of cohorts that will be backtested.
#         self.backtest_months = hold_months
#         print(f'Backtesting {hold_months} months.')
#         print(f'{cohort_count} cohorts will be backtested.')
#
#         def compute_error(actual, forecast, metric):
#             """
#             Test forecast performance against actuals using method defined by metric.
#             """
#             # root mean squared error
#             if metric == 'rmse':
#                 error = np.sqrt((1 / len(actual)) * sum((forecast[:len(actual)] - actual) ** 2))
#             # mean absolute error
#             elif metric == 'mae':
#                 error = np.mean(abs(forecast[:len(actual)] - actual))
#             # mean error
#             elif metric == 'me':
#                 error = np.mean(forecast[:len(actual)] - actual)
#             # mean absolute percent error
#             elif metric == 'mape':
#                 error = round((1 / len(actual)) * sum(abs((forecast[:len(actual)] - actual) / actual)), 4)
#             # mean percent error
#             elif metric == 'mpe':
#                 error = round((1 / len(actual)) * sum((forecast[:len(actual)] - actual) / actual), 4)
#             return error
#
#         # --- Generate backtest data --- #
#         backtest_report = []
#         backtest_data = []
#
#         for cohort in data.cohort.unique():
#             # data for current cohort
#             c_data = data[data.cohort == cohort]
#
#             # only backtest if remaining data has at least 4 data points
#             if len(c_data) - hold_months >= self.min_months:
#                 # limit data
#                 c_data = c_data.iloc[:len(c_data) - hold_months, :]
#
#                 # forecast the limited data
#                 predicted_data = self.forecast_data(c_data, min_months=min_months, n_months=fcast_months)
#
#                 # get forecast overlap with actuals
#                 actual = self.data[self.data['First Loan Local Disbursement Month'] == cohort]
#
#                 start = predicted_data[predicted_data.data_type == 'forecast'].index.min()
#                 stop = actual.index.max()
#
#                 # compute errors
#                 backtest_report_cols = []
#                 errors = []
#
#                 cols = [c for c in self.data.columns if c not in self.label_cols]
#                 cols.remove('Count First Loans')
#
#                 for col in cols:
#                     for metric in metrics:
#                         error = compute_error(actual.loc[start:stop, col], predicted_data.loc[start:stop, col],
#                                               metric=metric)
#
#                         backtest_report_cols += [f'{col}-{metric}']
#
#                         errors.append(error)
#
#                 backtest_report.append(pd.DataFrame.from_dict({cohort: errors}, orient='index',
#                                                               columns=backtest_report_cols))
#                 backtest_data.append(predicted_data)
#
#         backtest_data = pd.concat(backtest_data)
#         backtest_report = pd.concat(backtest_report, axis=0)
#         backtest_report['cohort'] = backtest_report.index
#
#         return backtest_data, backtest_report

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize

def forecast_data(data, min_months=5, n_months=50):
    # load dependent data

    min_months = min_months
    
    # initialize alpha and beta, optimized later by model
    alpha = beta = 1
    
    # list to hold individual cohort forecasts
    forecast_dfs = []
    
    # range of desired time periods
    times = np.arange(1, n_months+1)
    times_dict = {i: i-1 for i in times}
    
    # --- DEFAULT RATE FACTORS --- #
    # compute the default rate std dev across cohorts for the first 12 months
    default_std = data[['cohort', 'default_rate_7dpd']].copy()
    default_std = default_std.set_index('cohort', append=True).unstack(-2).iloc[:, :12]
    default_std = default_std.std()
    default_std.index = np.arange(1, len(default_std) + 1)
    
    def func(t, A, B):
        return A * t ** B
    
    params, covs = curve_fit(func, default_std.index, default_std)
    
    default_std_fit = func(times, params[0], params[1])
    default_std_fit = pd.Series(default_std_fit, index=times)
    
    default_expected_7 = ltv_expected['default_rate_7dpd']
    default_expected_51 = ltv_expected['default_rate_51dpd']
    default_expected_365 = ltv_expected['default_rate_365dpd']
    
    default_factors = []
    for c in data.cohort.unique():
        c_data = data[data.cohort==c]['default_rate_7dpd']
    
        default_factors.append(np.mean((c_data - default_expected_7[:len(c_data)])/default_std_fit[:len(c_data)]))
    default_factors = pd.Series(default_factors, index=data.cohort.unique())
    # -------------------------------#
    
    for cohort in data.cohort.unique():
        # data for current cohort
        c_data = data[data.cohort == cohort].copy()
    
        # starting cohort size
        n = c_data.loc[1, 'Count Borrowers']
        n_valid = len(c_data)
    
        # only for cohorts with at least min_months # of data points
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

            def power_fcast(c_data, param='borrower_retention'):

                c = c_data[param].dropna()

                def power_fit(times, a, b):
                    return a * np.array(times)**b

                # fit actuals and extract a & b params
                popt, pcov = curve_fit(power_fit, c.index, c)

                a = 1#popt[0]
                b = popt[1]

                # scale b according to market
                if market=='ke':
                    if len(c) < 6:
                        b = b + .02 * (6 - len(c) - 1)
                if market=='ph':
                    if len(c) < 6:
                        b = b + .02 * (6 - len(c) - 1)
                if market=='mx':
                    b = b - .015 * (18 - len(c) - 1)

                # get max survival from inputs
                max_survival = inputs.loc[market, 'max_monthly_borrower_retention']

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

    
            # --- ALL OTHERS --- #
            # compute survival
            c_data['borrower_survival'] = borrower_survival(c_data)
    
            # forecast loan size
            for i in c_data[c_data.loan_size.isnull()].index:
                c_data.loc[i, 'loan_size'] = c_data.loc[i - 1, 'loan_size'] * \
                                             ltv_expected.loc[i, 'loan_size'] / ltv_expected.loc[
                                                 i - 1, 'loan_size']
    
            # forecast loans_per_borrower
            for i in c_data[c_data.loans_per_borrower.isnull()].index:
                c_data.loc[i, 'loans_per_borrower'] = ltv_expected.loc[i, 'loans_per_borrower']
    
            # forecast Count Loans
            c_data['Count Loans'] = c_data['Count Loans'].fillna(
                (c_data['loans_per_borrower']) * c_data['Count Borrowers'])
    
            # forecast Total Amount
            c_data['Total Amount'] = c_data['Total Amount'].fillna(
                (c_data['loan_size'] * fx) * c_data['Count Loans'])
    
            # forecast Interest Rate
            for i in c_data[c_data.interest_rate.isnull()].index:
                c_data.loc[i, 'interest_rate'] = c_data.loc[i - 1, 'interest_rate'] * \
                                                 ltv_expected.loc[i, 'interest_rate'] / ltv_expected.loc[
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
    
            if default_stress:
                c_data['default_rate_7dpd'] += default_stress
                c_data['default_rate_365dpd'] += default_stress
    
            # compute remaining columns from forecasts
            c_data['loans_per_original'] = loans_per_original(c_data)
            c_data['cumulative_loans_per_original'] = c_data['loans_per_original'].cumsum()
            c_data['origination_per_original'] = origination_per_original(c_data)
            c_data['cumulative_origination_per_original'] = c_data['origination_per_original'].cumsum()
            c_data['revenue_per_original'] = revenue_per_original(c_data)
            c_data['cumulative_revenue_per_original'] = c_data['revenue_per_original'].cumsum()
            c_data['cm$_per_original'] = credit_margin(c_data)
            c_data['cumulative_cm$_per_original'] = c_data['cm$_per_original'].cumsum()
            c_data['opex_per_original'] = opex_per_original(c_data)
            c_data['cumulative_opex_per_original'] = c_data['opex_per_original'].cumsum()
            c_data['opex_coc_per_original'] = opex_coc_per_original(c_data)
            c_data['cumulative_opex_coc_per_original'] = opex_coc_per_original(c_data).cumsum()
            c_data['opex_cpl_per_original'] = opex_cpl_per_original(c_data)
            c_data['cumulative_opex_cpl_per_original'] = opex_cpl_per_original(c_data).cumsum()
            c_data['ltv_per_original'] = ltv_per_original(c_data)
            c_data['cumulative_ltv_per_original'] = c_data['ltv_per_original'].cumsum()
            c_data['dcf_ltv_per_original'] = dcf_ltv_per_original(c_data)
            c_data['cumulative_dcf_ltv_per_original'] = c_data['dcf_ltv_per_original'].cumsum()
            c_data['cm%_per_original'] = credit_margin_percent(c_data)
    
            # add the forecasted data for the cohort to a list, aggregating all cohort forecasts
            forecast_dfs.append(c_data)
    
    forecast_df = pd.concat(forecast_dfs)
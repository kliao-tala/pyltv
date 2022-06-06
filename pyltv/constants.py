# -----------------------------------------------------------------------------------------------------------------
# Model Constants
#
# This python file defines all constants used in the pyltv model.
# -----------------------------------------------------------------------------------------------------------------

forex = {'ke': 115, 'ph': 51, 'mx': 20}
# epsilon to avoid division by 0
epsilon = 1e-50
# initial values for alpha & beta in sbg model.
alpha = beta = 1
# discounted annual rate for discounted cash flow (DCF) framework
dcf = 0.15
late_fee = 0.08
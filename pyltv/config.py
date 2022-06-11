# -----------------------------------------------------------------------------------------------------------------
# Model Constants
#
# This python file defines all constants used in the pyltv model.
# -----------------------------------------------------------------------------------------------------------------

config = {
    'epsilon': 1e-50,  # small value to avoid division by 0
    'dcf': 0.15,  # discounted cash flow (dcf) rate
    'late_fee': 0.08,
    'forex': {
        'ke': 115,
        'ph': 51,
        'mx': 20
    },
    'opex_cpl': {  # cost per loan
        'ke': 1.32,
        'ph': 1.08,
        'mx': 1.37
    },
    'opex_coc': {  # cost of capital
        'ke': .13,
        'ph': .13,
        'mx': .13
    },
    'max_survival': {
        'ke': 0.96,
        'ph': 0.96,
        'mx': 0.94
    }
}

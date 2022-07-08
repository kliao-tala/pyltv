# pyLTV Library

Python library providing the following functionality:
1. Pulling data from snowflake. 
2. Cleaning data and prepping for models. 
3. Generating new data features (calculated fields). 
4. Plotting using the Plotly library. 
5. Forecasting data. 
6. Backtesting data. 
7. Publishing model output to snowflake.

For any questions or comments about this library, please reach out to *kenny.liao@tala.co*.

## Library Structure
The library consists of 3 main modules and a config file, all described below.

**dbm.py** is a Database Management module which contains the DBM class. The DBM class will establish a connection to
our Snowflake databases and allow queries as well as pulling pre-written queries for LTV analysis.

**pyltv.py** is the main module which includes the base model class, DataManager. The DataManager class provides data
cleaning, feature generation, and plotting functionality and serves as the base class which is inherited by all
forecast models.

**models.py** contains the forecasting models. Each model is created as its own class and has different logic built in
to forecast each data field. Each model class also contains a backtesting functionality.

**config.py** contains various constants and model parameters required for some forecasting models. Values such as max
survival, opex rates, and late fees can be specified here.

# Documentation
For the full documentation, visit the confluence page

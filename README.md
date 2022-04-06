# pyLTV Library

Python library for modeling customer life time value (LTV) data. Functionality includes automation of data pulling, cleaning, plotting, forecasting, and backtesting with high-level convenience functions. There are 3 main files described below.

pyltv.py
Main library containing model class and all functionality for data modeling.

sbg.py
Forecasting model based on a the shifted-beta-geometric model by Fader & Hardie.

dbm.py
Database management library which handles connection and interactions with snowflake.
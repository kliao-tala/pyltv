# -----------------------------------------------------------------
# Database management library
#
# This library defines a database management class which handles
# all interactions with snowflake.
# -----------------------------------------------------------------
import pandas as pd
import snowflake.connector

# for private key handling
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


# --- DATABASE MANAGER --- #
class DBM:
    """
    Database manager object to handle all interactions with the snowflake.

    Parameters
    ----------
    user : str
        user account name

    account : str
        snowflake account name

    warehouse : str
        snowflake warehouse name
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
        Loads private key from keyfile and sets keypass if specified.

        Parameters
        ----------
        keyfile : str
            location of keyfile

        Returns
        -------
        pkey
            private key bytes
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
        """
        Instantiates connection object to snowflake warehouse.

        Parameters
        ----------
        pkey :
            private key
        """
        self.ctx = snowflake.connector.connect(
            user=f'{self.user}@tala.co',
            account=f'{self.account}',
            private_key=pkey,
            warehouse=''
        )

    def run_sql(self, sql_file_path=None):
        """
        Run a sql query saved at sql_file_path.

        Parameters
        ----------
        sql_file_path: str
            pathname of sql file containing query to run
        """
        with open(sql_file_path) as f:
            query = f.read()
            with self.ctx.cursor() as curs:
                results = curs.execute(query)
                return pd.DataFrame.from_records(iter(results), columns=[c[0] for c in results.description])

    def query_db(self, query):
        """
        Query snowflake with the specified sql query.

        Parameters
        ----------
        query: str
            sql query statement
        """
        with self.ctx.cursor() as curs:
            results = curs.execute(query)
            return pd.DataFrame.from_records(iter(results), columns=[c[0] for c in results.description])

    def get_market_data(self, market='ke', start_date='2020-09-01', days_before=60):
        """
        Query the database with the standard query from the LTV Looker dashboard. Search
        parameters are replaced with the inputs.

        Parameters
        ----------
        market: str
            market to query data for

        start_date: str
            the earliest date to query data for

        days_before: int
            the number of days, prior to the current date, to query date up until
        """

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
# -----------------------------------------------------------------
# Database management library
#
# This library defines a database management class which handles
# all interactions with snowflake.
# -----------------------------------------------------------------
import pandas as pd
import snowflake.connector
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL

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

    def __init__(self, user=None, account='ng95977', warehouse='BI_ENGINEERING_XLARGE'):
        self.user = user
        self.account = account
        self.warehouse = warehouse
        self.ctx = None
        self.data = None
        self.engine = None
        self.pkey = None

        # get private key
        self.pkey = self.get_private_key_bytes(f'/Users/{self.user}/.ssh/snowflake_private_keypair.pem', None)

        # connect to snowflake
        self.create_engine(self.pkey)

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
            warehouse=self.warehouse
        )

    def create_engine(self, pkey, database='BUSINESS_DB', schema='FINANCE_TEST'):

        self.engine = create_engine(URL(
            account=f'{self.account}',
            user=f'{self.user}@tala.co',
            warehouse=self.warehouse,
            database=database,
            schema=schema
            ),
            connect_args={
                'private_key': pkey,
            }
        )

    def run_sql(self, sql_file_path=None):
        """
        Run a sql query saved at sql_file_path.

        Parameters
        ----------
        sql_file_path : str
            pathname of sql file containing query to run
        """
        with open(sql_file_path) as f:
            query = f.read()
            with self.ctx.cursor() as curs:
                results = curs.execute(query)
                return pd.DataFrame.from_records(iter(results), columns=[c[0] for c in results.description])

    def exec_sql(self, sql=None):
        with self.engine.connect() as connection:
            result = connection.execute(text(sql))
        for row in result:
            print(row)

    def query_db(self, query):
        """
        Query snowflake with the specified sql query.

        Parameters
        ----------
        query : str
            sql query statement
        """
        return pd.read_sql_query(query, self.engine)

    def get_market_data(self, market='ke', start_date='2020-09-01', days_before=0):
        """
        Query the database with the standard query from the LTV Looker dashboard. Search
        parameters are replaced with the inputs.

        Parameters
        ----------
        market : str
            market to query data for

        start_date : str
            the earliest date to query data for

        days_before : int
            the number of days, prior to the current date, to query date up until
        """
        if market != 'ke':
            query_params = {'REPLACE_DATE': start_date,
                            'REPLACE_DAYS': str(days_before),
                            '_KE': f'_{market.upper()}'}
        else:
            query_params = {'REPLACE_DATE': start_date,
                            'REPLACE_DAYS': str(days_before)}

        with open('queries/ltv_market_query.sql') as f:
            sql = f.read()
            for p in query_params:
                sql = sql.replace(p, query_params[p])

        df = pd.read_sql_query(sql, self.engine)

        # remove tablename from column names
        new_col_names = [c.split('.')[1] for c in df.columns]
        df.columns = new_col_names

        return df

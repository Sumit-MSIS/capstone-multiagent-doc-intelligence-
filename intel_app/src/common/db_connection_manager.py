import pymysql
import logging
from pymysql.err import MySQLError
from dbutils.pooled_db import PooledDB  # Connection pooling
from src.config.base_config import config
from src.config.config_loader import load_rds_credentials
import os
# SQL Database Configuration


class DBConnectionManager:
    """Manages MySQL database connections efficiently using a reusable connection pool."""
    
    _pool = None  # Class-level connection pool
    
    @classmethod
    def get_pool(cls):
        """Creates a connection pool if it does not exist."""

        # load_rds_credentials("DEV_DB_CREDENTIALS", region_name="us-east-1")

        # Base Configuration (Uses default DB)
        base_db_config = {
            "host": os.getenv("DB_HOST", config.DB_HOST),
            "port": int(os.getenv("DB_PORT", config.DB_PORT)),
            "user": os.getenv("DB_USER", config.DB_USER),
            "password": os.getenv("DB_PASSWORD", config.DB_PASSWORD),
            "cursorclass": pymysql.cursors.DictCursor,
            "autocommit": True
        }
        if cls._pool is None:
            cls._pool = PooledDB(
                creator=pymysql,
                mincached=1,
                maxcached=20,
                maxconnections=50, 
                blocking=True,
                ping=1,
                **base_db_config
            )
        return cls._pool

    def __init__(self, db_name: str, logger: logging.Logger):
        self.db_name = db_name
        self.logger = logger
        self.connection = None

    def __enter__(self):
        """Reuses a connection from the pool and switches to the target DB."""
        try:
            self.connection = self.get_pool().connection()
            with self.connection.cursor() as cursor:
                cursor.execute(f"USE {self.db_name};")
            self.logger.info(_log_message(f"Using database: {self.db_name}", "DBConnectionManager", "db_connection_manager"))
            return self.connection
        except pymysql.MySQLError as e:
            self.logger.error(_log_message(f"Database Connection Error: {e} | DB: {self.db_name}", "DBConnectionManager", "db_connection_manager"))
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Returns connection to pool."""
        if self.connection:
            self.connection.close()


def _log_message(message: str, function_name: str, module_name: str) -> str:
    return f"[function={function_name} | module={module_name}] - {message}"






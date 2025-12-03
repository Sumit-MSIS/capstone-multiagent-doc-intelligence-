import aiomysql
import logging
from src.config.base_config import config
from src.config.config_loader import load_rds_credentials
import os

class DBConnectionManager:
    """Manages MySQL database connections efficiently using a reusable connection pool."""
    
    _pool = None

    @classmethod
    async def get_pool(cls):

        # load_rds_credentials("DEV_DB_CREDENTIALS", region_name="us-east-1")

        base_db_config = {
            "host": os.getenv("DB_HOST", config.DB_HOST),
            "port": int(os.getenv("DB_PORT", config.DB_PORT)),
            "user": os.getenv("DB_USER", config.DB_USER),
            "password": os.getenv("DB_PASSWORD", config.DB_PASSWORD),
            "db": os.getenv("CONTRACT_INTEL_DB", config.CONTRACT_INTEL_DB),
            "cursorclass": aiomysql.DictCursor,
            "autocommit": True
        }
        if cls._pool is None:
            cls._pool = await aiomysql.create_pool(
                minsize=15,
                maxsize=30,
                **base_db_config
            )
        return cls._pool

    def __init__(self, db_name: str, logger: logging.Logger):
        self.db_name = db_name
        self.logger = logger
        self.connection = None

    async def __aenter__(self):
        try:
            pool = await self.get_pool()
            self.connection = await pool.acquire()
            
            try:
                await self.connection.ping(reconnect=True)
            except Exception as e:
                self.logger.warning(_log_message(f"Ping failed, reconnecting: {e}", "DBConnectionManager", "db_connection_manager"))
                self.connection.close()
                self.connection = await pool.acquire()
            
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"USE {self.db_name};")
            self.logger.info(_log_message(f"Using database: {self.db_name}", "DBConnectionManager", "db_connection_manager"))
            return self.connection  # Important: return the connection!
        except aiomysql.MySQLError as e:
            self.logger.error(_log_message(f"Database Connection Error: {e} | DB: {self.db_name}", "DBConnectionManager", "db_connection_manager"))
            raise

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.connection:
            pool = await self.get_pool()
            try:
                pool.release(self.connection)
            except Exception:
                self.connection.close()
            self.connection = None

def _log_message(message: str, function_name: str, module_name: str) -> str:
    return f"[function={function_name} | module={module_name}] - {message}"

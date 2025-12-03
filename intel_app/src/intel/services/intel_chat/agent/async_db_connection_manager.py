import aiomysql
import os
from dotenv import load_dotenv
load_dotenv()

class DBConnectionManager:
    """Manages MySQL database connections efficiently using a reusable connection pool."""
    
    _pool = None

    @classmethod
    async def get_pool(cls):

        # load_rds_credentials("DEV_DB_CREDENTIALS", region_name="us-east-1")

        base_db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT")),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "db": os.getenv("CONTRACT_INTEL_DB"),
            "cursorclass": aiomysql.DictCursor,
            "autocommit": True
        }
        if cls._pool is None:
            cls._pool = await aiomysql.create_pool(
                minsize=1,
                maxsize=5,
                **base_db_config
            )
        return cls._pool

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.connection = None

    async def __aenter__(self):
        try:
            pool = await self.get_pool()
            self.connection = await pool.acquire()
            
            try:
                await self.connection.ping(reconnect=True)
            except Exception as e:
                self.connection.close()
                self.connection = await pool.acquire()
            
            async with self.connection.cursor() as cursor:
                await cursor.execute(f"USE {self.db_name};")
            return self.connection  # Important: return the connection!
        except aiomysql.MySQLError as e:
            raise

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.connection:
            pool = await self.get_pool()
            try:
                pool.release(self.connection)
            except Exception:
                self.connection.close()
            self.connection = None

    @classmethod
    async def close_pool(cls):
        """Gracefully close the connection pool at application shutdown."""
        if cls._pool is not None:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._pool = None

def _log_message(message: str, function_name: str, module_name: str) -> str:
    return f"[function={function_name} | module={module_name}] - {message}"

import os
import redis
from psycopg2 import connect

def create_redis_conn():
    r = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"))
    return r

def create_postgres_conn():
    conn = connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    return conn

import os

def load_env():
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["REDIS_HOST"] = os.getenv("REDIS_HOST")
    os.environ["REDIS_PORT"] = os.getenv("REDIS_PORT")
    os.environ["REDIS_DB"] = os.getenv("REDIS_DB")
    os.environ["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST")
    os.environ["POSTGRES_PORT"] = os.getenv("POSTGRES_PORT")
    os.environ["POSTGRES_DB"] = os.getenv("POSTGRES_DB")
    os.environ["POSTGRES_USER"] = os.getenv("POSTGRES_USER")
    os.environ["POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD")

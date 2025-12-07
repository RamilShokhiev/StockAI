# backend/database.py
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


# === PostgreSQL connection config via env (prod) with sane defaults (dev) ===

# 1) If the DATABASE_URL variable is set, use it in its entirety.
#    For example:
#    postgresql+psycopg2://user:pass@host:5432/dbname
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # 2) Otherwise, we assemble the URL from separate environment variables
    #    (with defaults for local development).
    DB_USER = os.getenv("DB_USER", "stockai_user")
    DB_PASS = os.getenv("DB_PASSWORD", "123456")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "stockai")

    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQL logs are usually not needed, so echo=False is the default setting.
# You can enable it by setting DB_ECHO=true.
DB_ECHO = os.getenv("DB_ECHO", "false").lower() in ("1", "true", "yes")

engine = create_engine(
    DATABASE_URL,
    echo=DB_ECHO,
    future=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# Dependency для FastAPI — DB session via Depends
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# backend/database.py
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


# === PostgreSQL connection config via env (prod) with sane defaults (dev) ===

# 1) Если задана переменная DATABASE_URL — используем её целиком.
#    Например:
#    postgresql+psycopg2://user:pass@host:5432/dbname
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # 2) Иначе собираем URL из отдельных переменных окружения
    #    (с дефолтами для локальной разработки).
    DB_USER = os.getenv("DB_USER", "stockai_user")
    DB_PASS = os.getenv("DB_PASSWORD", "123456")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "stockai")

    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# В проде лог SQL обычно не нужен, поэтому по умолчанию echo=False.
# Включить можно через DB_ECHO=true
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


# Dependency для FastAPI — сессия БД через Depends
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

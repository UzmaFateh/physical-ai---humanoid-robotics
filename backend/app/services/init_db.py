from sqlalchemy import create_engine
from app.services.database_service import Base
from app.core.config import settings

def init_db():
    # Use the same database configuration logic as in database_service.py
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        from sqlalchemy import create_engine
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False}  # Required for SQLite
        )
    else:
        # For PostgreSQL or other databases
        from sqlalchemy import create_engine
        engine = create_engine(db_url)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()
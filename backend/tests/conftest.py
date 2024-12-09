import os
from tempfile import TemporaryDirectory

import pytest
from decouple import config
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, SQLModel, create_engine

from app import app
from similarities.db import get_session


test_db_url = config("DATABASE_URL").rsplit("/", 1)[0] + "/test_db"
engine = create_engine(test_db_url)

SQLModel.metadata.create_all(engine)

os.environ["STORAGE_DIR"] = TemporaryDirectory(prefix='storagetest').name

@pytest.fixture(scope="function", name="session")
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    TestSessionLocal = sessionmaker(class_=Session, autocommit=False, autoflush=False, bind=engine)
    session = TestSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function", name="client")
def test_client(session: Session):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override

    test_client = TestClient(app)
    yield test_client

    app.dependency_overrides.clear()

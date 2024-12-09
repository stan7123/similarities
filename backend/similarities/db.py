from decouple import config
from sqlmodel import Session, SQLModel, create_engine


engine = create_engine(config("DATABASE_URL"))

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


session_obj = Session(engine)

def get_session_instance():
    return session_obj

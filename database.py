from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# .env faylidan DATABASE_URL ni o'qiymiz
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Bazaga ulanish ("engine") yaratamiz
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Har bir so'rov uchun alohida sessiya ochuvchi klass
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ma'lumotlar bazasi modellarining asosi (poydevori)
Base = declarative_base()

# Baza bilan ulanishni nazorat qiluvchi funksiya
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



from sqlalchemy import Column, Integer, String, Text

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    role = Column(String)  # 'user' yoki 'model'
    content = Column(Text)

# Agar jadval hali yo'q bo'lsa, uni yaratamiz
Base.metadata.create_all(bind=engine)
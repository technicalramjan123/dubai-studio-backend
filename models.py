import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./dubai_studio.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    source_type = Column(String)          # "upload" or "link"
    original_filename = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    media_type = Column(String, nullable=True)   # "video" or "audio"
    source_language = Column(String, nullable=True)
    target_language = Column(String, nullable=True)
    voice = Column(String, nullable=True)         # e.g. "male" / "female"
    duration_seconds = Column(Float, nullable=True)

    status = Column(String, default="pending")
    # pending -> downloading -> extracting -> transcribing -> translating
    # -> synthesizing -> merging -> done / failed
    progress_percent = Column(Float, default=0.0)
    current_step_label = Column(String, default="Queued")
    error_message = Column(Text, nullable=True)

    input_path = Column(String, nullable=True)
    output_audio_path = Column(String, nullable=True)
    output_video_path = Column(String, nullable=True)
    srt_path = Column(String, nullable=True)
    transcript_json = Column(Text, nullable=True)  # full transcript, JSON string

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow,
                         onupdate=datetime.datetime.utcnow)

    chunks = relationship("Chunk", back_populates="job",
                           cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    index = Column(Integer)
    start_time = Column(Float)
    end_time = Column(Float)

    status = Column(String, default="pending")
    # pending -> transcribed -> translated -> synthesized -> done / failed
    source_text = Column(Text, nullable=True)
    translated_text = Column(Text, nullable=True)
    audio_chunk_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    job = relationship("Job", back_populates="chunks")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()

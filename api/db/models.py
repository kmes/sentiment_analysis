from sqlalchemy import (
    Integer, String, Float, Text,
    DateTime, func, Index, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import uuid

class Base(DeclarativeBase):
    pass

class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), default=uuid.uuid4, nullable=False, unique=True
    )
    model_load_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_load_logs.model_load_id"), nullable=False, unique=False)
    timestamp: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relazione uno-a-molti con ModelLoadLog basata su model_load_id
    model_load_log: Mapped["ModelLoadLog"] = relationship("ModelLoadLog", back_populates="inference_logs", uselist=False, foreign_keys="InferenceLog.model_load_id")

    # Relazione uno-a-uno con FeedbackLog basata su prediction_id
    feedback: Mapped["FeedbackLog"] = relationship("FeedbackLog", back_populates="inference_log", uselist=False, foreign_keys="FeedbackLog.prediction_id")

    __table_args__ = (
        Index("idx_inference_timestamp", "timestamp"),
        Index("idx_inference_prediction_id", "prediction_id"),
        Index("idx_inference_model_load_id", "model_load_id")
    )

class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("inference_logs.prediction_id"), nullable=False, unique=True)
    timestamp: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    true_label: Mapped[str] = mapped_column(String(20), nullable=False)

    # Relazione inversa uno-a-uno
    inference_log: Mapped["InferenceLog"] = relationship("InferenceLog", back_populates="feedback", uselist=False, foreign_keys=[prediction_id])

    __table_args__ = (
        Index("idx_feedback_prediction_id", "prediction_id"),
        Index("idx_feedback_timestamp", "timestamp")
    )

class ModelLoadLog(Base):
    __tablename__ = "model_load_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_load_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), default=uuid.uuid4, nullable=False, unique=True
    )
    timestamp: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    load_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relazione uno-a-molti con InferenceLog basata su model_load_id
    inference_logs: Mapped["InferenceLog"] = relationship("InferenceLog", back_populates="model_load_log", uselist=True, foreign_keys="InferenceLog.model_load_id")

    __table_args__ = (
        Index("idx_model_load_timestamp", "timestamp"),
        Index("idx_model_load_id", "model_load_id")
    )

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
    timestamp: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relazione uno-a-uno con FeedbackLog basata su prediction_id
    feedback: Mapped["FeedbackLog"] = relationship("FeedbackLog", back_populates="inference_log", uselist=False, foreign_keys="FeedbackLog.prediction_id")

    __table_args__ = (
        Index("idx_inference_timestamp", "timestamp"),
        Index("idx_inference_prediction_id", "prediction_id"),
        Index("idx_inference_model", "model_version")
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


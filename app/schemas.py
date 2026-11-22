from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        example="¿Cuáles son los requisitos para aprobar la asignatura?",
    )


class SourceChunk(BaseModel):
    page: int
    text_snippet: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]

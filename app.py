from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import DirectPipeline

pipeline = DirectPipeline()

app = FastAPI()

class QuestionContext(BaseModel):
    question: str
    context: str

@app.post("/answer")
def get_answer(question_context: QuestionContext):
    """
    Endpoint to get an answer to a question based on the provided context.
    """
    question = question_context.question
    context = question_context.context
    answer = pipeline(question, context, no_none=True)
    return {"answer": answer if answer is not None else "No answer found."}
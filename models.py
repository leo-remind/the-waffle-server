from datetime import datetime
from typing import List

from pydantic import BaseModel


class RAGQuery(BaseModel):
    query: str
    database_name: str
    filters: List[str]
    language: str


class TableVector(BaseModel):
    text: str  # this is the title of the table, the only thing which will be vectorised
    table_name: str
    year: int

import pandas as pd
import tiktoken
import textwrap as tr
from typing import List, Optional

from openai import OpenAI



def get_token_len_openAI(inp):
    client = OpenAI(max_retries=5)

    def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")

        response = client.embeddings.create(input=[text], model=model, **kwargs)

        return response.data[0].embedding

    embedding_model = "text-embedding-3-small"
    embedding_encoding = "cl100k_base"
    max_tokens = 8000  # the maximum for text-embedding-3-small is 8191

    encoding = tiktoken.get_encoding(embedding_encoding)

    return len(encoding.encode(inp))
import re
from urllib.parse import urlparse

import pandas as pd
import tiktoken
import textwrap as tr
from typing import List, Optional
import tldextract
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


def get_website_name(url):
    parsed_url = urlparse(url)
    print(parsed_url)
    website_name = parsed_url.path
    # Remove 'www.' if present
    if website_name.startswith('www.'):
        website_name = website_name[4:]

    # Remove common domain suffixes
    common_suffixes = ['.com', '.org', '.net', '.ai', '.edu', '.gov', '.io', '.co', '.us', '.uk', '.cn']
    # for suffix in common_suffixes:
    #     if suffix in common_suffixes

    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '-', website_name)
    return clean_name


if __name__ == '__main__':
    url = "www.123.com/c/vnkl;aeivasdljnel;kajsej"
    print(get_website_name(url))


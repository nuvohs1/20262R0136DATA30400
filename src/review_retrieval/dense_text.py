"""Text preparation for dense encoders (no third-party dependencies).

Unlike the BM25 tokenizer, this keeps case and punctuation, because transformer
tokenizers use them. Markup and media placeholders are removed the same way.
"""

import html
import re
import unicodedata

from .common import MEDIA, TAGS, TEXT_FIELDS

SPACES = re.compile(r"\s+")


def clean_text(text):
    text = unicodedata.normalize("NFKC", html.unescape(text or ""))
    text = MEDIA.sub(" ", text)
    text = TAGS.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def product_passage(product):
    """Title first, then store, features, and description, so truncation drops the description first."""
    parts = []
    for field in TEXT_FIELDS:
        value = clean_text(product.get(field) or "")
        if value:
            parts.append(f"{field.capitalize()}: {value}")
    return " | ".join(parts) if parts else "(empty)"


def review_texts(query):
    if len(query["reviews"]) != 5:
        raise ValueError("Each query must contain exactly five reviews")
    return [clean_text(review["text"]) or "(empty)" for review in query["reviews"]]

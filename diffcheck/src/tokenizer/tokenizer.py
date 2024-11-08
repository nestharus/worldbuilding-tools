from transformers import AutoTokenizer
import re
from tokenizer.token import Token
from typing import List

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_basic_tokenize=False)

def tokenize_english_text(text: str) -> List[Token]:
    # Step 1: Tokenize using BERT to get words and punctuation with positions
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = []

    for i, (token_id, offset) in enumerate(zip(encoded['input_ids'], encoded['offset_mapping'])):
        token_text = tokenizer.decode([token_id]).strip()
        start, end = offset

        # Handle special tokens like [UNK] by using the original text slice
        if token_text in tokenizer.all_special_tokens:
            token_text = text[start:end]

        tokens.append(Token(text=token_text, start=start))

    # Step 2: Detect whitespace sequences and add them to the token list
    for match in re.finditer(r'\s+', text):
        start, end = match.span()
        tokens.append(Token(text=match.group(), start=start))

    # Sort tokens by their start position to maintain order
    tokens.sort(key=lambda token: token.start)

    return tokens
import spacy
from tokenizer.token import Token

# Load English language model with all pipeline components
nlp = spacy.load("en_core_web_lg")


def should_merge_span(span) -> bool:
    """Determine if a spaCy span should be merged based on linguistic features"""
    try:
        if not span or len(span) == 1:
            return False

        # Don't merge titles with names
        if span[0].text.endswith('.') and len(span[0].text) <= 4:  # Likely a title
            return False

        # Don't merge ellipsis
        if any(t.text == '...' for t in span):
            return False


        # Don't merge if there's a comma in the span
        if any(t.text == ',' for t in span):
            return False

        # Check for named entities
        if span.ents:
            # Additional check for valid entity types
            for ent in span.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE'}:
                    return True

        # Handle contractions and possessives
        if any("'" in token.text for token in span):
            return True

        # Handle hyphenated words
        if any(token.text == '-' for token in span):
            return True

        # Handle abbreviations
        if all(token.is_upper or token.is_punct for token in span):
            return True

        return False
    except Exception:
        # If anything goes wrong, don't merge
        return False


def get_mergeable_spans(doc) -> list[spacy.tokens.Span]:
    """Identify spans that should be merged based on linguistic analysis"""
    spans = []

    # Get named entities
    spans.extend(doc.ents)

    # Get noun chunks (for possessives and compounds)
    spans.extend(doc.noun_chunks)

    # Handle hyphenated words and special characters
    i = 0
    while i < len(doc):
        if i < len(doc) - 2 and doc[i + 1].text == '-':
            # Look for hyphenated sequences
            end = i + 2
            while end < len(doc) - 1 and doc[end + 1].text == '-':
                end += 2
            spans.append(doc[i:end + 1])
            i = end + 1
            continue

        # Handle abbreviations and initials
        if doc[i].is_upper and i < len(doc) - 1 and doc[i + 1].text == '.':
            if i < len(doc) - 2 and doc[i + 2].is_upper:
                # Multi-letter abbreviation like U.S.A.
                end = i + 2
                while end < len(doc) - 1 and doc[end + 1].text == '.' and (
                        end + 2 >= len(doc) or doc[end + 2].is_upper):
                    end += 2
                spans.append(doc[i:end + 1])
                i = end + 1
            else:
                # Single initial like B.
                spans.append(doc[i:i + 2])
                i += 2
            continue

        # Handle contractions
        if i < len(doc) - 1 and "'" in doc[i + 1].text:
            spans.append(doc[i:i + 2])
            i += 2
            continue

        i += 1

    return [span for span in spans if should_merge_span(span)]


def tokenize_english_text(text: str) -> list[Token]:
    """Tokenize text using spaCy with linguistic-aware merging"""
    if not text:
        return []

    try:
        doc = nlp(text)
        tokens = []
        position = 0

        # Get spans to merge
        merge_spans = get_mergeable_spans(doc)
        merge_indices = set()
        for span in merge_spans:
            merge_indices.update(range(span.start, span.end))

        i = 0
        while i < len(doc):
            # Rest of the function logic...
            # Add any whitespace before the token
            if doc[i].idx > position:
                whitespace = text[position:doc[i].idx]
                if whitespace:
                    tokens.append(Token(text=whitespace, start=position))

            current_text = doc[i].text
            current_idx = doc[i].idx

            # Handle sequences of periods and ellipsis
            if current_text == '...' or current_text == '…':
                tokens.append(Token(text=current_text, start=current_idx))
                position = current_idx + len(current_text)
                i += 1
                continue
            elif current_text == '..':
                tokens.append(Token(text='.', start=current_idx))
                tokens.append(Token(text='.', start=current_idx + 1))
                position = current_idx + 2
                i += 1
                continue
                
            # Handle single letters with periods (I., B., etc)
            if (len(current_text) == 2 and 
                current_text[0].isalpha() and 
                current_text[1] == '.'):
                tokens.append(Token(text=current_text[0], start=current_idx))
                tokens.append(Token(text='.', start=current_idx + 1))
                position = current_idx + 2
                i += 1
                continue
                
            # Handle abbreviations (U.S.A.)
            if (len(current_text) > 1 and
                all(c.isupper() or c == '.' for c in current_text)):
                tokens.append(Token(text=current_text, start=current_idx))
                position = current_idx + len(current_text)
                i += 1
                continue
            elif current_text == '.':
                tokens.append(Token(text='.', start=current_idx))
                position = current_idx + 1
                i += 1
            # Handle names with commas
            elif i < len(doc) - 1 and doc[i + 1].text == ',':
                tokens.append(Token(text=current_text, start=current_idx))
                tokens.append(Token(text=',', start=doc[i + 1].idx))
                position = doc[i + 1].idx + 1
                i += 2
            # Check if current token is part of a mergeable span
            elif i in merge_indices:
                # Find the full span this token belongs to
                span = next(s for s in merge_spans if s.start <= i < s.end)
                span_text = text[doc[span.start].idx:doc[span.end - 1].idx + len(doc[span.end - 1].text)]
                tokens.append(Token(text=span_text, start=doc[span.start].idx))
                position = doc[span.end - 1].idx + len(doc[span.end - 1].text)
                i = span.end
            else:
                # Add single token
                tokens.append(Token(text=doc[i].text, start=doc[i].idx))
                position = doc[i].idx + len(doc[i].text)
                i += 1

        # Add any trailing whitespace
        if position < len(text):
            tokens.append(Token(text=text[position:], start=position))

        return process_abbreviations(tokens)
            
    except Exception as e:
        # Return simple fallback tokenization on error
        return [Token(text=text, start=0)]
        # Get spans to merge
        merge_spans = get_mergeable_spans(doc)
        merge_indices = set()
        for span in merge_spans:
            merge_indices.update(range(span.start, span.end))

        i = 0
        while i < len(doc):
            # Add any whitespace before the token
            if doc[i].idx > position:
                whitespace = text[position:doc[i].idx]
                if whitespace:
                    tokens.append(Token(text=whitespace, start=position))

            current_text = doc[i].text
            current_idx = doc[i].idx

            # Handle sequences of periods and ellipsis
            if current_text == '...' or current_text == '…':
                tokens.append(Token(text=current_text, start=current_idx))
                position = current_idx + len(current_text)
                i += 1
                continue
            elif current_text == '..':
                tokens.append(Token(text='.', start=current_idx))
                tokens.append(Token(text='.', start=current_idx + 1))
                position = current_idx + 2
                i += 1
                continue
                
            # Handle single letters with periods (I., B., etc)
            elif (len(current_text) == 2 and 
                current_text[0].isalpha() and 
                current_text[1] == '.'):
                tokens.append(Token(text=current_text[0], start=current_idx))
                tokens.append(Token(text='.', start=current_idx + 1))
                position = current_idx + 2
                i += 1
                continue
                
            # Handle single period
            elif current_text == '.':
                tokens.append(Token(text='.', start=current_idx))
                position = current_idx + 1
                i += 1
                continue
                
            # Handle names with commas - don't merge
            elif i < len(doc) - 1 and doc[i + 1].text == ',':
                tokens.append(Token(text=current_text, start=current_idx))
                tokens.append(Token(text=',', start=doc[i + 1].idx))
                position = doc[i + 1].idx + 1
                i += 2
                continue
                
            # Check if current token is part of a mergeable span
            elif i in merge_indices:
                # Find the full span this token belongs to
                span = next(s for s in merge_spans if s.start <= i < s.end)
                span_text = text[doc[span.start].idx:doc[span.end - 1].idx + len(doc[span.end - 1].text)]
                tokens.append(Token(text=span_text, start=doc[span.start].idx))
                position = doc[span.end - 1].idx + len(doc[span.end - 1].text)
                i = span.end
                continue
                
            # Default case - add single token
            else:
                tokens.append(Token(text=current_text, start=current_idx))
                position = current_idx + len(current_text)
                i += 1

    # Add any trailing whitespace
    if position < len(text):
        tokens.append(Token(text=text[position:], start=position))

    return process_abbreviations(tokens)


def is_abbreviation(text):
    # Check if the text contains at least one period
    if '.' not in text:
        return False

    # Check if all characters are uppercase letters, lowercase letters, or periods
    for char in text:
        if not (char.isalpha() or char == '.'):
            return False

    return True

def process_abbreviations(tokens: list[Token]) -> list[Token]:
    """Handle the specific case of extra periods after abbreviations."""
    for outer_index, outer in enumerate(tokens):
        if is_abbreviation(outer.text) and not outer.text.endswith('.'):
            if outer_index < len(tokens) - 1:
                inner = tokens[outer_index + 1]
                if inner.text == '.':
                    tokens[outer_index] = Token(outer.text + inner.text, outer.start)
                    tokens.pop(outer_index + 1)
    return tokens

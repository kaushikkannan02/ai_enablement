import os

def read_text_file(file_path: str):
    """Read content from a text file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_document(file_path: str):
    """Read document content (TXT only)"""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".txt":
        return read_text_file(file_path)
    else:
        raise ValueError("Only .txt files are supported")

def split_text(text: str, chunk_size: int = 200):
    """Split text into chunks while preserving sentence boundaries"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Ensure proper sentence ending
        if not sentence.endswith('.'):
            sentence += '.'

        sentence_size = len(sentence)

        # Check if adding this sentence would exceed chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
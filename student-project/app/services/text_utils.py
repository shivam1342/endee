def chunk_text(text: str, chunk_size: int = 550, overlap: int = 120) -> list[str]:
    """Split text into overlapping word chunks for stable retrieval quality."""
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks

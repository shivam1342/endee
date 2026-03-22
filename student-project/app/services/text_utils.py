def chunk_text(text: str, chunk_size: int = 550, overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks for retrieval."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start += step

    return chunks

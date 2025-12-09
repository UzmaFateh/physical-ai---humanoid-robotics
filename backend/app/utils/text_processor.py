from typing import List
import re


class TextProcessor:
    def __init__(self):
        pass

    def clean_content(self, content: str) -> str:
        """
        Clean and normalize text content
        """
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove special characters if needed
        content = content.strip()
        return content

    def extract_structure(self, content: str) -> dict:
        """
        Extract structure information like headings from content
        """
        # Simple heading extraction using regex
        headings = re.findall(r'(#{1,6})\s+(.+?)(?=\n|$)', content)
        return {
            "headings": headings,
            "word_count": len(content.split()),
            "char_count": len(content)
        }

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks of specified size with overlap
        """
        # Simple tokenization based on words
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            # Calculate the end index for this chunk
            end_idx = min(i + chunk_size, len(words))

            # Create the chunk
            chunk = ' '.join(words[i:end_idx])
            chunks.append(chunk)

            # Move to the next chunk, considering overlap
            i = end_idx - overlap if overlap < end_idx else end_idx

            # If overlap is too large, just move to the next non-overlapping position
            if overlap >= chunk_size:
                i = end_idx

        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]

        return chunks
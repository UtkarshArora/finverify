"""
Updated chunk_docs_fast.py - Works with extracted HTML
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import re


# def clean_html_text(html_content):
#     """
#     Clean HTML and extract readable text
#     """
#     # Parse with BeautifulSoup
#     soup = BeautifulSoup(html_content, "html.parser")

#     # Remove script, style, and other non-content elements
#     for element in soup(["script", "style", "meta", "link", "head", "noscript"]):
#         element.decompose()


#     # Get text
#     text = soup.get_text(separator=" ", strip=True)

#     # Clean up whitespace
#     text = re.sub(r"\s+", " ", text)
#     text = text.strip()

#     return text


def clean_html_text(html_content: str) -> str:
    """Aggressively clean SEC HTML and keep mostly narrative text."""
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Drop obvious non-content
    for tag in soup(["script", "style", "meta", "link", "head", "noscript"]):
        tag.decompose()

    # 2. Drop tables and layout-y blocks (huge source of numeric junk)
    for tag in soup(
        ["table", "thead", "tbody", "tfoot", "tr", "td", "th", "nav", "footer"]
    ):
        tag.decompose()

    # 3. Extract raw text
    text = soup.get_text(separator=" ", strip=True)

    # 4. De-HTML-entity a bit (if you like)
    replacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#8217;": "'",
        "&#8220;": '"',
        "&#8221;": '"',
        "&#x2013;": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 5. Filter out numeric-heavy / tiny lines
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        letters = sum(c.isalpha() for c in line)
        digits = sum(c.isdigit() for c in line)

        # Skip mostly-numeric stuff (tables, line items)
        if digits > 0 and digits >= 2 * max(1, letters):
            continue

        # Skip very short fragments
        if len(line) < 40:
            continue

        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quick_chunk_text(text, chunk_size=2048, overlap=200):
    """Quick character-based chunking"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk_text = text[start:end]

        if len(chunk_text.strip()) > 50:
            chunks.append(chunk_text)

        start = end - overlap

        if end >= text_len:
            break

    return chunks


def process_sec_html(input_dir):
    """
    Process extracted HTML files from SEC filings
    """

    print("\n" + "=" * 60)
    print("Processing SEC EDGAR HTML Files")
    print("=" * 60)

    # Try both possible locations
    possible_dirs = [
        Path(input_dir) / "sec_manual",  # Extracted HTML
        # Path(input_dir) / "sec_edgar" / "sec-edgar-filings"  # Original .txt
    ]

    html_files = []
    sec_dir = None

    for dir_path in possible_dirs:
        if dir_path.exists():
            # Look for .html or .htm files
            html_files = list(dir_path.rglob("*.html")) + list(dir_path.rglob("*.htm"))
            if html_files:
                sec_dir = dir_path
                break

    if not html_files:
        print("⚠️  No HTML files found!")
        print(f"   Checked: {[str(d) for d in possible_dirs]}")
        print("\n   Please run extract_html_from_sec.py first")
        return []

    print(f"✓ Found {len(html_files)} HTML files in {sec_dir}")

    all_chunks = []
    processed = 0
    skipped = 0

    for i, filepath in enumerate(tqdm(html_files, desc="Processing HTML")):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            # Extract clean text from HTML
            text = clean_html_text(html_content)

            # Skip if too short
            if len(text) < 100:
                skipped += 1
                continue

            # Chunk the text
            chunks = quick_chunk_text(text)

            for j, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "chunk_id": f"sec_{i}_{j}",
                        "source": "sec_edgar",
                        "source_file": filepath.name,
                        "text": chunk,
                    }
                )

            processed += 1

        except Exception as e:
            skipped += 1
            continue

    print(f"✓ Processed: {processed} files → {len(all_chunks)} chunks")
    print(f"⚠ Skipped: {skipped} files (errors or too short)")

    return all_chunks


def process_json_dataset(filepath, source_name):
    """Process JSON datasets (FinanceBench, TATQA)"""

    with open(filepath, "r") as f:
        data = json.load(f)

    all_chunks = []

    for i, item in enumerate(data):
        if source_name == "financebench":
            text = item.get("context", "") or item.get("answer", "")
        elif source_name == "tatqa":
            # Handle TATQA paragraphs
            paragraphs = item.get("paragraphs", [])
            if isinstance(paragraphs, list):
                text_parts = []
                for p in paragraphs:
                    if isinstance(p, str):
                        text_parts.append(p)
                    elif isinstance(p, dict):
                        text_parts.append(p.get("text", ""))
                text = " ".join(text_parts)
            else:
                text = str(paragraphs)
        else:
            text = item.get("assistant", "")

        if len(text) < 50:
            continue

        chunks = quick_chunk_text(text)

        for j, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{source_name}_{i}_{j}",
                    "source": source_name,
                    "text": chunk,
                }
            )

    return all_chunks


def main():
    """Chunking pipeline with HTML support"""

    print("=" * 60)
    print("Document Chunking Pipeline (HTML Support)")
    print("=" * 60)

    base_dir = Path("/scratch") / os.environ["USER"] / "finverify"
    input_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    # Process SEC HTML files
    print("\n📄 Processing SEC EDGAR HTML files...")
    sec_chunks = process_sec_html(input_dir)
    if sec_chunks:
        all_chunks.extend(sec_chunks)
        print(f"✓ SEC: {len(sec_chunks):,} chunks")
    else:
        print("⚠️  No SEC chunks (run extract_html_from_sec.py first)")

    # FinanceBench
    print("\n📊 Processing FinanceBench...")
    fb_file = input_dir / "financebench" / "financebench_full.json"
    if fb_file.exists():
        fb_chunks = process_json_dataset(fb_file, "financebench")
        all_chunks.extend(fb_chunks)
        print(f"✓ FinanceBench: {len(fb_chunks):,} chunks")
    else:
        print("⚠️  FinanceBench not found")

    # TATQA
    print("\n📈 Processing TATQA...")
    tatqa_dir = input_dir / "tatqa"
    if tatqa_dir.exists():
        for filename in [
            "tatqa_dataset_train.json",
            "tatqa_dataset_dev.json",
            "tatqa_dataset_test.json",
        ]:
            filepath = tatqa_dir / filename
            if filepath.exists():
                tatqa_chunks = process_json_dataset(filepath, "tatqa")
                all_chunks.extend(tatqa_chunks)
                print(f"✓ {filename}: {len(tatqa_chunks):,} chunks")
    else:
        print("⚠️  TATQA not found")

    # Save
    if all_chunks:
        output_file = output_dir / "chunks.json"
        print(f"\n💾 Saving {len(all_chunks):,} chunks...")

        with open(output_file, "w") as f:
            json.dump(all_chunks, f)

        print(f"✓ Saved to: {output_file}")

        # Metadata
        metadata = {
            "total_chunks": len(all_chunks),
            "chunk_size": 2048,
            "overlap": 200,
            "html_cleaned": True,
        }

        with open(output_dir / "chunks_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("✓ CHUNKING COMPLETE!")
        print("=" * 60)
        print(f"\nTotal chunks: {len(all_chunks):,}")
        print("\nNext steps:")
        print("1. Build BM25 index:  python3 build_bm25.py")
        print("2. Generate embeddings: python3 generate_embeddings.py")
        print("3. Build FAISS index: python3 build_faiss.py")
        print("4. Test baselines:    python3 bm25_t5.py --test-mode")
    else:
        print("\n❌ No chunks created!")
        print("   Please check that data files exist")


if __name__ == "__main__":
    main()

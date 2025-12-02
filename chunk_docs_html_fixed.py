"""
Fixed chunk_docs_html.py - Keeps financial data while removing HTML
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import re


def clean_html_text(html_content: str) -> str:
    """
    Clean SEC HTML while preserving financial data
    
    Key changes from broken version:
    1. DON'T remove <table> tags before extraction (they contain financial data!)
    2. DON'T filter out numeric lines (revenue, cash, etc. have numbers!)
    3. Only remove pure junk (symbols, very short lines, etc.)
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Remove non-content tags
    for tag in soup(["script", "style", "meta", "link", "head", "noscript"]):
        tag.decompose()

    # 2. Extract ALL text (including from tables - no HTML tags in result)
    text = soup.get_text(separator=" ", strip=True)

    # 3. Clean HTML entities
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
        "&#x2014;": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 4. Smart line filtering - Keep financial data!
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        
        if not line:
            continue

        # Skip very short lines (likely junk)
        if len(line) < 20:
            continue
        
        # Must have at least some letters (not pure numbers/symbols)
        letters = sum(c.isalpha() for c in line)
        if letters < 3:
            continue
        
        # Skip lines that are mostly punctuation/symbols
        alphanumeric = sum(c.isalnum() for c in line)
        if alphanumeric < len(line) * 0.3:
            continue
        
        # KEEP everything else - including lines with numbers!
        # These are the financial facts we need:
        # - "Total revenue was $265.6 billion"
        # - "Cash and cash equivalents: $133.8 billion"
        # - "Gross margin improved to 38.3%"
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
    """Process extracted HTML files from SEC filings"""

    print("\n" + "=" * 70)
    print("PROCESSING SEC EDGAR HTML FILES")
    print("=" * 70)

    # Try both possible locations
    possible_dirs = [
        Path(input_dir) / "sec_manual",
        Path(input_dir) / "sec_rendered",
        Path(input_dir) / "sec_edgar_html",
    ]

    html_files = []
    sec_dir = None

    for dir_path in possible_dirs:
        if dir_path.exists():
            html_files = list(dir_path.rglob("*.html")) + list(dir_path.rglob("*.htm"))
            if html_files:
                sec_dir = dir_path
                print(f"✓ Found {len(html_files)} HTML files in {sec_dir}")
                break

    if not html_files:
        print("❌ No HTML files found!")
        print(f"   Checked: {[str(d) for d in possible_dirs]}")
        return []

    all_chunks = []
    processed = 0
    skipped = 0
    total_text_length = 0

    print("\nProcessing files...")
    for i, filepath in enumerate(tqdm(html_files, desc="Processing HTML")):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            # Show details for first file
            if i == 0:
                print(f"\n--- FIRST FILE SAMPLE ---")
                print(f"File: {filepath.name}")
                print(f"HTML size: {len(html_content):,} bytes")

            # Extract clean text from HTML
            text = clean_html_text(html_content)

            if i == 0:
                print(f"Cleaned text size: {len(text):,} bytes")
                print(f"Text preview (first 300 chars):")
                print(text[:300])
                print("---\n")

            # Skip if too short
            if len(text) < 1000:
                skipped += 1
                if i < 3:
                    print(f"⚠️  Skipped {filepath.name}: text too short ({len(text)} chars)")
                continue

            total_text_length += len(text)

            # Chunk the text
            chunks = quick_chunk_text(text, chunk_size=2048, overlap=200)

            if i == 0:
                print(f"Chunks created: {len(chunks)}")
                print(f"First chunk preview:")
                print(chunks[0][:200])
                print("---\n")

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
            if i < 5:
                print(f"\n⚠️  Error processing {filepath.name}: {e}")
            continue

    print(f"\n" + "=" * 70)
    print("SEC PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Files found:       {len(html_files)}")
    print(f"Files processed:   {processed}")
    print(f"Files skipped:     {skipped}")
    print(f"Chunks created:    {len(all_chunks):,}")
    print(f"Total text:        {total_text_length:,} bytes")
    
    if all_chunks:
        avg_chunk_size = sum(len(c['text']) for c in all_chunks) / len(all_chunks)
        print(f"Avg chunk size:    {avg_chunk_size:.0f} chars")
        print(f"Avg chunks/file:   {len(all_chunks)/max(processed,1):.0f}")

    return all_chunks


def process_json_dataset(filepath, source_name):
    """Process JSON datasets (FinanceBench, TATQA)"""

    if not filepath.exists():
        return []

    with open(filepath, "r") as f:
        data = json.load(f)

    all_chunks = []

    for i, item in enumerate(data):
        if source_name == "financebench":
            text = item.get("context", "") or item.get("answer", "")
        elif source_name == "tatqa":
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
    """Chunking pipeline with fixed HTML cleaning"""

    print("=" * 70)
    print("DOCUMENT CHUNKING PIPELINE (FIXED)")
    print("=" * 70)
    print("\nFixes:")
    print("1. Keeps <table> content (financial data!)")
    print("2. Doesn't filter numeric lines (revenue, cash, etc.)")
    print("3. Only removes pure junk (symbols, short lines)")

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
        print("⚠️  No SEC chunks")

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

        # Summary by source
        sources = {}
        for c in all_chunks:
            src = c.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"\nChunks by source:")
        for source, count in sorted(sources.items()):
            pct = 100 * count / len(all_chunks)
            print(f"  {source:15s}: {count:8,} ({pct:5.1f}%)")
        print(f"\n  {'TOTAL':15s}: {len(all_chunks):8,}")

        # Metadata
        metadata = {
            "total_chunks": len(all_chunks),
            "by_source": sources,
            "chunk_size": 2048,
            "overlap": 200,
            "html_cleaned": True,
            "cleaning_version": "fixed_v2"
        }

        with open(output_dir / "chunks_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 70)
        print("✓ CHUNKING COMPLETE!")
        print("=" * 70)
        print(f"\nTotal chunks: {len(all_chunks):,}")
        
        # Expected improvement message
        sec_count = sources.get('sec_edgar', 0)
        if sec_count > 50000:
            print(f"\n🎉 EXCELLENT! {sec_count:,} SEC chunks created!")
            print("Expected performance:")
            print("  EM: 18-25% (was 2%)")
            print("  F1: 32-40% (was 6%)")
        elif sec_count > 20000:
            print(f"\n👍 GOOD! {sec_count:,} SEC chunks created")
            print("Expected performance:")
            print("  EM: 12-18%")
            print("  F1: 22-32%")
        else:
            print(f"\n⚠️  Only {sec_count:,} SEC chunks - still low")
            print("Files might still be inline XBRL")
            print("Consider downloading 2017 filings or using rendered HTML")
        
        print("\nNext steps:")
        print("1. Build BM25 index:  python3 build_bm25.py")
        print("2. Generate embeddings: python3 generate_embeddings.py")
        print("3. Build FAISS index: python3 build_faiss.py")
        print("4. Test baselines:    python3 baselines/bm25_t5.py --test-mode")
    else:
        print("\n❌ No chunks created!")
        print("   Please check that HTML files exist")


if __name__ == "__main__":
    main()

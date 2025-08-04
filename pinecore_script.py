import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# --- Setup Pinecone Client ---
PINECONE_API_KEY = "pcsk_9WSic_8DGehGtHrh3NNKhAT3LfekaJrdCLq9TLUdndNnGBkL3FKsxR7FtcrpwNx32M7gt"
INDEX_NAME = "construction-docs"

pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-west1-aws")
index = pc.Index(INDEX_NAME)

# === Helper: Flatten Docling 'texts' list into Pinecone-ready chunks ===
def flatten_docling_json(json_data, file_tag):
    chunks = []
    for text in json_data.get("texts", []):
        if not text.get("text"): continue
        prov = text.get("prov", [])
        page = prov[0].get("page_no") if prov and isinstance(prov, list) else None

        chunk = {
            "id": f"{file_tag}_{len(chunks)}",
            "text": text["text"],
            "metadata": {
                "type": file_tag,
                "label": text.get("label"),
                "page": page
            }
        }
        chunks.append(chunk)
    return chunks

# === Load all JSONs and flatten ===
json_files = ["xref_full.json", "construction_drawings.json", "specifications.json"]
combined = []

for file in json_files:
    if os.path.exists(file):
        with open(file) as f:
            raw = json.load(f)
            if isinstance(raw, list):
                combined.extend(raw)
                print(f"✅ Loaded {len(raw)} flat records from {file}")
            elif isinstance(raw, dict) and "texts" in raw:
                flat_chunks = flatten_docling_json(raw, os.path.splitext(os.path.basename(file))[0])
                combined.extend(flat_chunks)
                print(f"✅ Extracted {len(flat_chunks)} chunks from {file}")
            else:
                print(f"⚠️ Skipped: Unrecognized format in {file}")
    else:
        print(f"⚠️ File not found: {file}")

# === Embed valid chunks ===
model = SentenceTransformer("all-MiniLM-L6-v2")
valid_data, skipped = [], []

for item in tqdm(combined):
    if isinstance(item, dict) and "text" in item and isinstance(item["text"], str) and item["text"].strip():
        item["embedding"] = model.encode(item["text"]).tolist()
        valid_data.append(item)
    else:
        skipped.append(item)

print(f"\n✅ {len(valid_data)} valid chunks ready for upload")
print(f"⚠️ Skipped {len(skipped)} items due to missing 'text' or invalid format")

# === Upload to Pinecone ===
batch_size = 100
for i in range(0, len(valid_data), batch_size):
    batch = valid_data[i:i + batch_size]
    vectors = [
        {
            "id": item["id"],
            "values": item["embedding"],
            "metadata": item.get("metadata", {})
        }
        for item in batch
    ]
    index.upsert(vectors=vectors)

print(f"\n✅ Uploaded {len(valid_data)} vectors to Pinecone index '{INDEX_NAME}'")

# Optional: Save skipped entries for inspection
with open("skipped_records.json", "w") as f:
    json.dump(skipped, f, indent=2)
    print("📁 Saved skipped records to 'skipped_records.json'")
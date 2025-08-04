# Gradio Demo2 app for Construction QA
import gradio as gr
import json
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# === Load xref_full.json ===
with open("xref_full.json") as f:
    xref_pairs = json.load(f)

# === Prepare context chunks ===
chunks = []
for i, pair in enumerate(xref_pairs):
    context = f"[SPEC]\n{pair['spec_text']}\n\n[DRAWING]\n{pair['drawing_text']}"
    chunks.append({
        "id": f"xref_{i}",
        "context": context,
        "spec_text": pair["spec_text"],
        "drawing_text": pair["drawing_text"],
        "drawing_page": pair.get("drawing_page"),
        "bbox": pair.get("drawing_bbox"),
        "similarity": pair.get("similarity"),
        "drawing_id": pair.get("drawing_id"),
        "spec_id": pair.get("spec_id")
    })

# === Embed all chunks ===
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["context"] for c in chunks]
embeddings = model.encode(texts, convert_to_tensor=True)

# === Load QA model ===
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# === QA Function ===
def answer_question(question):
    q_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, embeddings)[0]
    top_idx = scores.topk(k=1).indices[0].item()

    chunk = chunks[top_idx]
    context = chunk["context"]
    result = qa_pipeline(question=question, context=context)

    response = f"""
🔍 **Answer:** {result['answer']}\n


📄 **Drawing Page:** {chunk['drawing_page']}\n
🟦 **Bounding Box:** `{chunk['bbox']}`\n
🧾 **Spec ID:** `{chunk['spec_id']}`\n
🧾 **Drawing ID:** `{chunk['drawing_id']}`\n
🔗 **Context ID:** `{chunk['id']}`\n
---

🔹 **Spec Snippet:**
{chunk['spec_text'][:400]}\n

🔹 **Drawing Snippet:**
{chunk['drawing_text'][:400]}
"""
    return response

# === Gradio Interface ===
examples = [
    "Are all fire-rated wall assemblies detailed?",
    "Type of hangers.",
    "Confirm Trench drains with lids and sanitary piping (set/embed into concrete by concrete sub)",
    "Is ADA compliance for door hardware ensured?",
    "Entry doors – smart lock or deadbolt?",
    "Does plumber provide unit garbage disposal or is it included in appliance package?",
    "Anything required for guard shack?",
    "Who provides unit water sub meters? Who installs?",
    "Number of lifts.",
    "Lighting Lead times.",
    "Pipe and Specialties - Check responsibility for electrical wiring and low voltage interlocks of EP’s, PE’s, remote temperature indication, alarms, etc. Is control panel pre-wired?",
    "What are the requirements for the guard shack?",
    "Is hot and/or cold-water insulation included?",
    "Who is responsible for plumbing inspections?",
    "Is GFCI protection required for bathrooms?"
]

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question, or pick an example"),
    outputs=gr.Markdown(label="Answer with References"),
    examples=examples,
    theme='light',
    title="Construction QA (Task 2)",
    description="Ask vague checklist-style questions. RoBERTa answers using cross-referenced spec and drawing chunks."
)

demo.launch()
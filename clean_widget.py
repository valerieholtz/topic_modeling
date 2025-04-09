import nbformat

path = "final_topic_modeling.ipynb"  # 🔁 Replace this with your actual notebook filename

with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("✅ Cleaned metadata.widgets without removing outputs.")

import chromadb
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from datetime import datetime

# Load a pretrained ResNet to extract image embeddings
_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_resnet.eval()
# Remove the final classification layer — we want the 512-dim feature vector
_feature_extractor = torch.nn.Sequential(*list(_resnet.children())[:-1])

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Persistent ChromaDB — data survives restarts
_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _client.get_or_create_collection(
    name="diagnoses",
    metadata={"hnsw:space": "cosine"}
)


def get_image_embedding(image_path):
    """Extract a 512-dim feature vector from an image using ResNet18."""
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = _feature_extractor(tensor).squeeze().tolist()
    return embedding


def store_diagnosis(image_path, disease, confidence):
    """Store a diagnosis in memory with its image embedding."""
    embedding = get_image_embedding(image_path)
    doc_id = f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[f"Diagnosed as {disease} with {confidence:.1f}% confidence"],
        metadatas=[{
            "disease": disease,
            "confidence": round(confidence, 1),
            "date": datetime.now().isoformat(),
            "image_path": image_path,
        }]
    )
    return doc_id


def search_similar(image_path, n_results=3):
    """Find the most similar past diagnoses to a given image."""
    if _collection.count() == 0:
        return []

    embedding = get_image_embedding(image_path)
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, _collection.count()),
    )

    similar_cases = []
    for i in range(len(results["ids"][0])):
        similar_cases.append({
            "id": results["ids"][0][i],
            "disease": results["metadatas"][0][i]["disease"],
            "confidence": results["metadatas"][0][i]["confidence"],
            "date": results["metadatas"][0][i]["date"],
            "similarity": round(1 - results["distances"][0][i], 3),
            "summary": results["documents"][0][i],
        })

    return similar_cases


def get_memory_context(similar_cases):
    """Format similar cases into a text context string for the LLM."""
    if not similar_cases:
        return "No similar past cases found in memory."

    lines = ["Similar past cases from memory:"]
    for i, case in enumerate(similar_cases, 1):
        lines.append(
            f"  {i}. {case['disease']} ({case['confidence']}% confidence, "
            f"similarity: {case['similarity']}, date: {case['date'][:10]})"
        )
    return "\n".join(lines)

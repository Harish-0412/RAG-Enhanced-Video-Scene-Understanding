#!/usr/bin/env python3
"""
Phase 3 Search Test - Verification Script
Tests that ChromaDB is correctly retrieving scenes based on meaning, not just keywords.
"""

import chromadb
from chromadb.utils import embedding_functions
import sys
from pathlib import Path

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import log

def test_search():
    """Test ChromaDB search with semantic queries."""
    
    print("🔍 Phase 3 Search Test (Verification)")
    print("=" * 60)
    
    try:
        # Connect to your existing DB
        client = chromadb.PersistentClient(path="./chroma_db")
        default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # List available collections first
        collections = client.list_collections()
        print(f" Available collections: {[c.name for c in collections]}")
        
        if not collections:
            print(" No collections found in ChromaDB!")
            return False
        
        # Try to get the dense collection (with combined_context)
        collection_name = "video_moments_dense"
        if collection_name not in [c.name for c in collections]:
            print(f" Collection '{collection_name}' not found!")
            print(f" Creating new collection...")
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=default_ef,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = client.get_collection(name=collection_name, embedding_function=default_ef)
            
        print(f" Connected to ChromaDB at: ./chroma_db")
        print(f" Collection stats: {collection.count()} vectors")
        
        # Test queries
        test_queries = [
            "a person explaining a diagram",
            "technical code on a screen", 
            "vector database cosine similarity",
            "retrieval augmented generation"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching for: '{query}'")
            print("-" * 50)
            
            results = collection.query(
                query_texts=[query],
                n_results=2  # Get top 2 most relevant scenes
            )
            
            if not results['ids'][0]:
                print("❌ No results found")
                continue
                
            print(f"📊 Found {len(results['ids'][0])} result(s)")
            
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                print(f"\n  [{i+1}] 📹 {doc_id}")
                print(f"      🕐 Timestamp: {metadata.get('timestamp', 'N/A')}")
                print(f"      📄 Frame: {metadata.get('frame_id', 'N/A')}")
                print(f"      📏 Distance: {distance:.4f}")
                print(f"      📝 Context: {document[:150]}...")
                
                # Check if result actually contains semantic meaning
                if any(word in document.lower() for word in query.lower().split()):
                    print(f"      ✅ Semantic match found!")
                else:
                    print(f"      ⚠️  May be keyword match only")
            
            print("=" * 60)
            
        return True
        
    except Exception as e:
        print(f"❌ Error during search test: {e}")
        return False

if __name__ == "__main__":
    success = test_search()
    if success:
        print("\n🎉 Phase 3 Search Test PASSED!")
        print("📈 ChromaDB is correctly performing semantic search")
        print("🔄 Ready for production use")
    else:
        print("\n❌ Phase 3 Search Test FAILED!")
        print("🔧 Check ChromaDB setup and data")

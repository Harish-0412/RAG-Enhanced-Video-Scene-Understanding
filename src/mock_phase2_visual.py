#!/usr/bin/env python3

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_metadata, save_metadata

def mock_phase2_visual():
    """Mock version of phase2_visual that demonstrates integration without API calls."""
    
    print("🔧 Running mock Phase 2B (Visual) integration test...")
    
    # Load enriched data from phase2_audio
    enriched_path = "../data/processed/metadata/enriched_metadata.json"
    results_path = "../data/processed/results/multimodal_data.json"
    
    try:
        scenes = load_metadata(enriched_path)
        print(f"✅ Loaded {len(scenes)} enriched scenes from phase2_audio")
        
        # Mock VLM captioning - add visual descriptions
        print("👁️  Mock VLM captioning...")
        for i, scene in enumerate(scenes):
            if not scene.get("dedup_skipped"):
                # Add mock visual data
                scene["visual_description"] = {
                    "on_screen_text": f"Mock slide {i+1} content",
                    "diagram_type": "slide",
                    "visual_actions": "person speaking",
                    "key_concepts": [f"concept_{i+1}", "rag", "retrieval"],
                    "summary": f"Mock visual summary for scene {i+1} showing RAG concepts"
                }
                scene["on_screen_text"] = f"Mock slide {i+1} content"
                scene["diagram_type"] = "slide"
                scene["key_concepts"] = [f"concept_{i+1}", "rag", "retrieval"]
                scene["visual_summary"] = f"Mock visual summary for scene {i+1} showing RAG concepts"
                
                print(f"  [{i+1}/{len(scenes)}] {scene['frame_id']} - Mock caption added")
        
        # Mock multimodal fusion
        print("🔗 Mock multimodal fusion...")
        for scene in scenes:
            parts = []
            
            if scene.get("visual_summary"):
                parts.append(f"[VISUAL] {scene['visual_summary']}")
            
            if scene.get("on_screen_text"):
                parts.append(f"[TEXT ON SCREEN] {scene['on_screen_text']}")
            
            if scene.get("scene_transcript"):
                parts.append(f"[AUDIO] {scene['scene_transcript']}")
            
            scene["combined_context"] = "\n".join(parts) if parts else ""
        
        # Save results
        save_metadata(scenes, results_path)
        print(f"✅ Mock Phase 2B complete!")
        print(f"📁 Results saved to: {results_path}")
        
        # Show sample of combined data
        print("\n📊 Sample combined data:")
        for i, scene in enumerate(scenes[:2]):
            print(f"\nScene {i+1} ({scene['frame_id']}):")
            print(f"  - Audio file: {scene.get('audio_file', 'N/A')}")
            print(f"  - Visual type: {scene.get('diagram_type', 'N/A')}")
            print(f"  - Has transcript: {bool(scene.get('scene_transcript'))}")
            print(f"  - Combined context length: {len(scene.get('combined_context', ''))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    mock_phase2_visual()

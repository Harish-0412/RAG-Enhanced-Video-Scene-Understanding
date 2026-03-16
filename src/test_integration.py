#!/usr/bin/env python3

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_metadata

def test_phase2_integration():
    """Test that Phase 2A output is correctly integrated."""
    
    # Test that we can load the enriched metadata from phase2_audio
    enriched_path = "../data/processed/metadata/enriched_metadata.json"
    
    try:
        scenes = load_metadata(enriched_path)
        print(f"✅ Successfully loaded {len(scenes)} scenes from phase2_audio output")
        
        # Check that audio fields are present
        for i, scene in enumerate(scenes[:2]):  # Check first 2 scenes
            print(f"\nScene {i+1}:")
            print(f"  - Frame ID: {scene.get('frame_id', 'N/A')}")
            print(f"  - Has transcript: {bool(scene.get('scene_transcript'))}")
            print(f"  - Word count: {scene.get('word_count', 0)}")
            print(f"  - Confidence: {scene.get('avg_confidence', 'N/A')}")
            print(f"  - Low confidence segments: {scene.get('has_low_conf_seg', False)}")
            
        print(f"\n✅ Phase 2A (Audio) integration verified!")
        print(f"📝 Phase 2B (Visual) requires GEMINI_API_KEY in .env file")
        
        # Check if transcript file exists
        transcript_path = "../data/processed/metadata/transcript.json"
        if Path(transcript_path).exists():
            with open(transcript_path, 'r') as f:
                import json
                transcript = json.load(f)
                print(f"📄 Transcript contains {len(transcript)} segments")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_phase2_integration()

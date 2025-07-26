#!/usr/bin/env python3
"""
Fast development script for Manim animations
Usage: python fast_dev.py [scene_name] [quality]
"""

import sys
import subprocess
import os
from pathlib import Path

def run_fast_render(scene_file, scene_name, quality="l"):
    """
    Run Manim with fast rendering settings
    
    Args:
        scene_file: Path to your .py file
        scene_name: Name of the scene class
        quality: 'l' (low), 'm' (medium), 'h' (high), 'u' (ultra)
    """
    
    # Quality flags
    quality_flags = {
        'l': '-pql',  # Low quality - fastest
        'm': '-pqm',  # Medium quality
        'h': '-pqh',  # High quality
        'u': '-pqu'   # Ultra quality - slowest
    }
    
    quality_flag = quality_flags.get(quality, '-pql')
    
    # Build command
    cmd = f"manim {quality_flag} {scene_file} {scene_name}"
    
    print(f"🚀 Running: {cmd}")
    print(f"📁 File: {scene_file}")
    print(f"🎬 Scene: {scene_name}")
    print(f"⚡ Quality: {quality} ({quality_flag})")
    print("-" * 50)
    
    # Run the command
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print("✅ Rendering completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Rendering failed with error: {e}")
        return False

def list_scenes():
    """List available scene files"""
    py_files = list(Path('.').glob('*.py'))
    print("📋 Available scene files:")
    for file in py_files:
        print(f"  - {file}")
    print()

def main():
    if len(sys.argv) < 2:
        print("🎬 Fast Manim Development Tool")
        print("=" * 40)
        print("Usage: python fast_dev.py <scene_file> [scene_name] [quality]")
        print()
        print("Examples:")
        print("  python fast_dev.py padim_strict_no_overlap.py PaDiMStrictNoOverlap l")
        print("  python fast_dev.py debug_scene.py DebugScene m")
        print()
        print("Quality options: l (low), m (medium), h (high), u (ultra)")
        print()
        list_scenes()
        return
    
    scene_file = sys.argv[1]
    scene_name = sys.argv[2] if len(sys.argv) > 2 else None
    quality = sys.argv[3] if len(sys.argv) > 3 else 'l'
    
    # If no scene name provided, try to guess from filename
    if not scene_name:
        scene_name = Path(scene_file).stem.replace('_', '').title()
        print(f"🤔 No scene name provided, guessing: {scene_name}")
    
    # Check if file exists
    if not os.path.exists(scene_file):
        print(f"❌ File not found: {scene_file}")
        return
    
    # Run the render
    success = run_fast_render(scene_file, scene_name, quality)
    
    if success:
        print("\n🎉 Development render completed!")
        print("💡 Tips for faster development:")
        print("  - Use quality 'l' for fastest rendering")
        print("  - Test specific parts in debug_scene.py")
        print("  - Use shorter animations for testing")

if __name__ == "__main__":
    main() 
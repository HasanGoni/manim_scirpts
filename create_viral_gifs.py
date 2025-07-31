#!/usr/bin/env python3
"""
Viral GIF Creator for PaDiM Animation
Extracts the most engaging moments for social media content
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys

def create_viral_gif(video_path, output_dir, start_time, duration, name, scale_factor=0.8):
    """Create a viral-worthy GIF from video segment"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    
    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        frame = cv2.resize(frame, (width, height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    
    # Save as GIF
    output_path = os.path.join(output_dir, f"{name}.gif")
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,  # Convert to milliseconds
            loop=0,
            optimize=True
        )
        print(f"✅ Created viral GIF: {output_path}")
        return output_path
    else:
        print(f"❌ No frames extracted for {name}")
        return None

def main():
    """Create viral GIFs from PaDiM animation"""
    
    video_path = "media/videos/padim_strict_no_overlap/480p15/PaDiMStrictNoOverlap.mp4"
    output_dir = "/home/hasan/Schreibtisch/projects/git_data/quarto_blog_hasan/posts/series/anomaly-detection"
    
    # Define viral moments with timing (in seconds)
    viral_moments = [
        {
            "name": "padim_challenge_viral",
            "start_time": 15,  # Challenge introduction
            "duration": 12,
            "description": "The Industrial Challenge - Hook viewers with the problem"
        },
        {
            "name": "padim_solution_viral", 
            "start_time": 45,  # Solution overview
            "duration": 10,
            "description": "PaDiM Solution - The 'Aha!' moment"
        },
        {
            "name": "padim_patch_extraction_viral",
            "start_time": 75,  # Patch extraction
            "duration": 8,
            "description": "Patch Extraction - Visual foundation"
        },
        {
            "name": "padim_feature_extraction_viral",
            "start_time": 105,  # Feature extraction
            "duration": 12,
            "description": "Multi-layer Features - Technical depth"
        },
        {
            "name": "padim_gaussian_viral",
            "start_time": 150,  # Gaussian modeling
            "duration": 10,
            "description": "Gaussian Modeling - The statistical magic"
        },
        {
            "name": "padim_mahalanobis_viral",
            "start_time": 200,  # Mahalanobis distance
            "duration": 8,
            "description": "Mahalanobis Distance - The detection mechanism"
        },
        {
            "name": "padim_pipeline_viral",
            "start_time": 240,  # Complete pipeline
            "duration": 10,
            "description": "Complete Pipeline - The full picture"
        },
        {
            "name": "padim_inference_viral",
            "start_time": 280,  # Inference demo
            "duration": 8,
            "description": "Inference Demo - Real-world application"
        },
        {
            "name": "padim_finale_viral",
            "start_time": 320,  # Finale
            "duration": 12,
            "description": "Key Takeaways - The grand finale"
        }
    ]
    
    print("🎬 Creating Viral PaDiM GIFs for Social Media...")
    print("=" * 60)
    
    created_gifs = []
    
    for moment in viral_moments:
        print(f"\n🎯 Creating: {moment['description']}")
        gif_path = create_viral_gif(
            video_path=video_path,
            output_dir=output_dir,
            start_time=moment['start_time'],
            duration=moment['duration'],
            name=moment['name']
        )
        if gif_path:
            created_gifs.append({
                'path': gif_path,
                'name': moment['name'],
                'description': moment['description']
            })
    
    print("\n" + "=" * 60)
    print("🎉 Viral GIF Creation Complete!")
    print(f"📁 Created {len(created_gifs)} viral GIFs in: {output_dir}")
    
    # Create a summary file for easy reference
    summary_path = os.path.join(output_dir, "viral_gifs_summary.md")
    with open(summary_path, 'w') as f:
        f.write("# PaDiM Viral GIFs for Social Media\n\n")
        f.write("## Created GIFs:\n\n")
        for gif in created_gifs:
            f.write(f"### {gif['name']}\n")
            f.write(f"- **Description**: {gif['description']}\n")
            f.write(f"- **File**: `{os.path.basename(gif['path'])}`\n\n")
        
        f.write("## Social Media Usage Tips:\n\n")
        f.write("1. **Twitter/X**: Use shorter GIFs (padim_challenge_viral, padim_solution_viral)\n")
        f.write("2. **LinkedIn**: Use technical GIFs (padim_feature_extraction_viral, padim_gaussian_viral)\n")
        f.write("3. **Instagram**: Use visual GIFs (padim_patch_extraction_viral, padim_pipeline_viral)\n")
        f.write("4. **YouTube Thumbnails**: Use padim_finale_viral for thumbnails\n")
        f.write("5. **Blog Posts**: Use all GIFs strategically throughout content\n\n")
        
        f.write("## Viral Content Strategy:\n\n")
        f.write("- **Hook**: Start with the challenge (padim_challenge_viral)\n")
        f.write("- **Solution**: Show the breakthrough (padim_solution_viral)\n")
        f.write("- **Process**: Demonstrate the method (padim_pipeline_viral)\n")
        f.write("- **Result**: End with impact (padim_finale_viral)\n")
    
    print(f"📝 Summary created: {summary_path}")
    print("\n🚀 Ready for viral social media content!")

if __name__ == "__main__":
    main() 
from manim import *
import numpy as np

class DebugScene(Scene):
    """Ultra-fast debug scene for testing specific parts"""
    
    def construct(self):
        # Test only the part you're working on
        # Example: Test text rendering
        title = Text("Debug Test", font_size=36, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        
        # Test specific animation you're debugging
        # Uncomment the part you want to test:
        
        # Test patch extraction
        # self.test_patch_extraction()
        
        # Test feature embedding
        # self.test_feature_embedding()
        
        # Test gaussian modeling
        # self.test_gaussian_modeling()
        
        # Test anomaly map
        # self.test_anomaly_map()
    
    def test_patch_extraction(self):
        """Test only patch extraction part"""
        # Your patch extraction code here
        pass
    
    def test_feature_embedding(self):
        """Test only feature embedding part"""
        # Your feature embedding code here
        pass
    
    def test_gaussian_modeling(self):
        """Test only gaussian modeling part"""
        # Your gaussian modeling code here
        pass
    
    def test_anomaly_map(self):
        """Test only anomaly map part"""
        # Your anomaly map code here
        pass

if __name__ == "__main__":
    # Run with ultra-fast quality
    # manim -pql debug_scene.py DebugScene
    pass 
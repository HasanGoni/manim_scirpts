from manim import *
import numpy as np

class ModularPaDiM(Scene):
    """Modular PaDiM animation for faster development"""
    
    def construct(self):
        """Main construction - comment out parts you don't need to test"""
        
        # Test only the parts you're working on
        # Uncomment the section you want to test:
        
        # self.test_introduction()
        # self.test_challenge()
        # self.test_patch_extraction()
        # self.test_feature_embedding()
        # self.test_gaussian_modeling()
        # self.test_inference()
        # self.test_epilogue()
        
        # Or run the full animation
        self.full_animation()
    
    def full_animation(self):
        """Complete animation - use for final testing"""
        self.test_introduction()
        self.test_challenge()
        self.test_patch_extraction()
        self.test_feature_embedding()
        self.test_gaussian_modeling()
        self.test_inference()
        self.test_epilogue()
    
    def test_introduction(self):
        """Test only the introduction part"""
        title = Text("PaDiM: Patch Distribution Modeling", font_size=36, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        self.clear()
    
    def test_challenge(self):
        """Test only the challenge part"""
        challenge = Text("Industrial Anomaly Detection Challenge", font_size=24, color=RED)
        self.play(Write(challenge))
        self.wait(1)
        self.play(FadeOut(challenge))
        self.clear()
    
    def test_patch_extraction(self):
        """Test only the patch extraction part"""
        # Your patch extraction code here
        patch_title = Text("Patch Extraction", font_size=24, color=GREEN)
        self.play(Write(patch_title))
        self.wait(1)
        self.play(FadeOut(patch_title))
        self.clear()
    
    def test_feature_embedding(self):
        """Test only the feature embedding part"""
        # Your feature embedding code here
        feature_title = Text("Feature Embedding", font_size=24, color=YELLOW)
        self.play(Write(feature_title))
        self.wait(1)
        self.play(FadeOut(feature_title))
        self.clear()
    
    def test_gaussian_modeling(self):
        """Test only the gaussian modeling part"""
        # Your gaussian modeling code here
        gaussian_title = Text("Gaussian Modeling", font_size=24, color=PURPLE)
        self.play(Write(gaussian_title))
        self.wait(1)
        self.play(FadeOut(gaussian_title))
        self.clear()
    
    def test_inference(self):
        """Test only the inference part"""
        # Your inference code here
        inference_title = Text("Inference Process", font_size=24, color=ORANGE)
        self.play(Write(inference_title))
        self.wait(1)
        self.play(FadeOut(inference_title))
        self.clear()
    
    def test_epilogue(self):
        """Test only the epilogue part"""
        # Your epilogue code here
        epilogue_title = Text("Epilogue", font_size=24, color=BLUE)
        self.play(Write(epilogue_title))
        self.wait(1)
        self.play(FadeOut(epilogue_title))
        self.clear()

if __name__ == "__main__":
    # Run with ultra-fast quality for development
    # manim -pql modular_padim.py ModularPaDiM
    pass 
from manim import *
import numpy as np

class SimpleTest(Scene):
    def construct(self):
        # Test basic shapes without complex text
        circle = Circle(radius=2, color=BLUE)
        square = Square(side_length=2, color=RED)
        
        self.play(Create(circle))
        self.wait(1)
        self.play(Transform(circle, square))
        self.wait(1)
        
        # Simple text that might work
        try:
            simple_text = Tex("PaDiM")
            self.play(Write(simple_text))
        except:
            # Fallback without text
            dot = Dot(color=YELLOW)
            self.play(Create(dot))
        
        self.wait(2) 
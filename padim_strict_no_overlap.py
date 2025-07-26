from manim import *
import numpy as np

# Animation Studio Color Palette
PRIMARY_COLOR = BLUE
SECONDARY_COLOR = YELLOW  
ACCENT_COLOR = RED
SUCCESS_COLOR = GREEN
NORMAL_COLOR = "#2E8B57"  # Sea Green for normal samples
ANOMALY_COLOR = "#DC143C"  # Crimson for anomalies

class PaDiMStrictNoOverlap(Scene):
    """
    PaDiM Animation - STRICT NO OVERLAP RULE
    
    What it shows: Complete PaDiM methodology with ZERO overlapping animations
    Duration: ~5-6 minutes
    Key concepts: Challenge → Solution → Patch Extraction → Implementation Details
    Cleanup: ONE animation at a time, complete scene separation
    """
    
    def construct(self):
        """Complete PaDiM story with STRICT no-overlap rules"""
        self.introduction()
        self.the_challenge()
        self.solution_overview()
        self.patch_extraction_process()
        self.feature_extraction_implementation()
        self.dimensionality_reduction_implementation()
        self.gaussian_modeling_implementation()
        self.mahalanobis_implementation()
        self.complete_pipeline()
        self.inference_demo()
        self.finale()
    
    def strict_clear(self):
        """STRICT clearing - remove ALL objects and clear completely"""
        if self.mobjects:
            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        self.wait(0.5)  # Brief pause between scenes
    
    def introduction(self):
        """Opening scene with PaDiM overview - NO OVERLAP"""
        # Title 1
        title1 = Text("PaDiM", font_size=56, color=PRIMARY_COLOR, weight=BOLD)
        self.play(Write(title1), run_time=2)
        self.wait(1)
        self.play(FadeOut(title1), run_time=1)
        self.clear()
        
        # Title 2
        title2 = Text("Patch Distribution Modeling", font_size=32, color=SECONDARY_COLOR)
        self.play(Write(title2), run_time=2)
        self.wait(1)
        self.play(FadeOut(title2), run_time=1)
        self.clear()
        
        # Title 3
        title3 = Text("Solving Industrial Anomaly Detection", font_size=24, color=WHITE)
        self.play(Write(title3), run_time=2)
        self.wait(2)
        self.play(FadeOut(title3), run_time=1)
        self.strict_clear()
    
    def the_challenge(self):
        """Scene 1: The Industrial Anomaly Detection Challenge - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("The Challenge", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Challenge title
        challenge_title = Text("Industrial Anomaly Detection Challenge", font_size=24, color=ACCENT_COLOR, weight=BOLD)
        challenge_title.to_edge(UP, buff=0.5)
        self.play(Write(challenge_title), run_time=1)
        self.wait(1)
        
        # Challenge 1
        challenge1 = Text("❌ Limited Training Data", font_size=18, color=ACCENT_COLOR, weight=BOLD)
        challenge1.shift(LEFT * 3 + UP * 1)
        self.play(Write(challenge1), run_time=1)
        self.wait(1)
        
        # Challenge 1 details
        detail1a = Text("• Only normal samples available", font_size=14, color=WHITE)
        detail1a.shift(LEFT * 3 + UP * 0.5)
        self.play(Write(detail1a), run_time=1)
        self.wait(0.5)
        
        detail1b = Text("• Anomalies are rare and unknown", font_size=14, color=WHITE)
        detail1b.shift(LEFT * 3 + UP * 0.2)
        self.play(Write(detail1b), run_time=1)
        self.wait(1)
        
        # Challenge 2
        challenge2 = Text("❌ Need for Precise Localization", font_size=18, color=ACCENT_COLOR, weight=BOLD)
        challenge2.shift(LEFT * 3 + DOWN * 0.2)
        self.play(Write(challenge2), run_time=1)
        self.wait(1)
        
        # Challenge 2 details
        detail2a = Text("• Must identify exact defect location", font_size=14, color=WHITE)
        detail2a.shift(LEFT * 3 + DOWN * 0.5)
        self.play(Write(detail2a), run_time=1)
        self.wait(0.5)
        
        detail2b = Text("• Pixel-level accuracy required", font_size=14, color=WHITE)
        detail2b.shift(LEFT * 3 + DOWN * 0.8)
        self.play(Write(detail2b), run_time=1)
        self.wait(1)
        
        # Challenge 3
        challenge3 = Text("❌ Real-time Processing", font_size=18, color=ACCENT_COLOR, weight=BOLD)
        challenge3.shift(LEFT * 3 + DOWN * 1.5)
        self.play(Write(challenge3), run_time=1)
        self.wait(1)
        
        # Challenge 3 details
        detail3a = Text("• Fast inference for production lines", font_size=14, color=WHITE)
        detail3a.shift(LEFT * 3 + DOWN * 1.8)
        self.play(Write(detail3a), run_time=1)
        self.wait(0.5)
        
        detail3b = Text("• No complex training needed", font_size=14, color=WHITE)
        detail3b.shift(LEFT * 3 + DOWN * 2.1)
        self.play(Write(detail3b), run_time=1)
        self.wait(2)
        
        # Clear left side
        self.play(FadeOut(challenge1), FadeOut(detail1a), FadeOut(detail1b), 
                  FadeOut(challenge2), FadeOut(detail2a), FadeOut(detail2b),
                  FadeOut(challenge3), FadeOut(detail3a), FadeOut(detail3b), run_time=1)
        self.clear()
        
        # Right side - Manufacturing scenario
        scenario_title = Text("🏭 Manufacturing Scenario:", font_size=16, color=SECONDARY_COLOR, weight=BOLD)
        scenario_title.shift(RIGHT * 2.5 + UP * 1.5)
        self.play(Write(scenario_title), run_time=1)
        self.wait(1)
        
        scenario1 = Text("• Quality control on production line", font_size=12, color=WHITE)
        scenario1.shift(RIGHT * 2.5 + UP * 1)
        self.play(Write(scenario1), run_time=1)
        self.wait(0.5)
        
        scenario2 = Text("• Detect scratches, dents, color variations", font_size=12, color=WHITE)
        scenario2.shift(RIGHT * 2.5 + UP * 0.7)
        self.play(Write(scenario2), run_time=1)
        self.wait(0.5)
        
        scenario3 = Text("• Must work with minimal normal data", font_size=12, color=WHITE)
        scenario3.shift(RIGHT * 2.5 + UP * 0.4)
        self.play(Write(scenario3), run_time=1)
        self.wait(0.5)
        
        scenario4 = Text("• Real-time decision making", font_size=12, color=WHITE)
        scenario4.shift(RIGHT * 2.5 + UP * 0.1)
        self.play(Write(scenario4), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(challenge_title), FadeOut(scenario_title), 
                  FadeOut(scenario1), FadeOut(scenario2), FadeOut(scenario3), FadeOut(scenario4), run_time=1)
        self.strict_clear()
    
    def solution_overview(self):
        """Scene 2: PaDiM Solution Overview - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("PaDiM Solution", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Solution title
        solution_title = Text("PaDiM: Patch Distribution Modeling", font_size=24, color=PRIMARY_COLOR, weight=BOLD)
        solution_title.to_edge(UP, buff=0.5)
        self.play(Write(solution_title), run_time=1)
        self.wait(1)
        
        # Key insight
        insight_title = Text("💡 Key Insight:", font_size=18, color=PRIMARY_COLOR, weight=BOLD)
        insight_title.shift(LEFT * 3 + UP * 1.5)
        self.play(Write(insight_title), run_time=1)
        self.wait(1)
        
        insight1 = Text("Model normal patch distributions", font_size=16, color=WHITE)
        insight1.shift(LEFT * 3 + UP * 1.2)
        self.play(Write(insight1), run_time=1)
        self.wait(0.5)
        
        insight2 = Text("instead of learning anomalies", font_size=16, color=WHITE)
        insight2.shift(LEFT * 3 + UP * 0.9)
        self.play(Write(insight2), run_time=1)
        self.wait(1)
        
        # Advantages
        advantages_title = Text("✅ Advantages:", font_size=18, color=SUCCESS_COLOR, weight=BOLD)
        advantages_title.shift(LEFT * 3 + UP * 0.3)
        self.play(Write(advantages_title), run_time=1)
        self.wait(1)
        
        advantage1 = Text("• Only needs normal samples", font_size=14, color=SUCCESS_COLOR)
        advantage1.shift(LEFT * 3 + UP * 0)
        self.play(Write(advantage1), run_time=1)
        self.wait(0.5)
        
        advantage2 = Text("• Precise pixel-level localization", font_size=14, color=SUCCESS_COLOR)
        advantage2.shift(LEFT * 3 + DOWN * 0.3)
        self.play(Write(advantage2), run_time=1)
        self.wait(0.5)
        
        advantage3 = Text("• Fast training & inference", font_size=14, color=SUCCESS_COLOR)
        advantage3.shift(LEFT * 3 + DOWN * 0.6)
        self.play(Write(advantage3), run_time=1)
        self.wait(0.5)
        
        advantage4 = Text("• Works with any pre-trained CNN", font_size=14, color=SUCCESS_COLOR)
        advantage4.shift(LEFT * 3 + DOWN * 0.9)
        self.play(Write(advantage4), run_time=1)
        self.wait(2)
        
        # Clear left side
        self.play(FadeOut(insight_title), FadeOut(insight1), FadeOut(insight2),
                  FadeOut(advantages_title), FadeOut(advantage1), FadeOut(advantage2),
                  FadeOut(advantage3), FadeOut(advantage4), run_time=1)
        self.clear()
        
        # Right side - Core idea
        core_title = Text("🎯 Core Idea:", font_size=16, color=SECONDARY_COLOR, weight=BOLD)
        core_title.shift(RIGHT * 2.5 + UP * 1.5)
        self.play(Write(core_title), run_time=1)
        self.wait(1)
        
        step1 = Text("1. Extract patches from image", font_size=12, color=WHITE)
        step1.shift(RIGHT * 2.5 + UP * 1.2)
        self.play(Write(step1), run_time=1)
        self.wait(0.5)
        
        step2 = Text("2. Get features from CNN layers", font_size=12, color=WHITE)
        step2.shift(RIGHT * 2.5 + UP * 0.9)
        self.play(Write(step2), run_time=1)
        self.wait(0.5)
        
        step3 = Text("3. Model normal distribution per patch", font_size=12, color=WHITE)
        step3.shift(RIGHT * 2.5 + UP * 0.6)
        self.play(Write(step3), run_time=1)
        self.wait(0.5)
        
        step4 = Text("4. Detect anomalies via distance scoring", font_size=12, color=WHITE)
        step4.shift(RIGHT * 2.5 + UP * 0.3)
        self.play(Write(step4), run_time=1)
        self.wait(1)
        
        # Implementation details
        impl_title = Text("🔬 Implementation Details:", font_size=14, color=ACCENT_COLOR, weight=BOLD)
        impl_title.shift(RIGHT * 2.5 + DOWN * 0.3)
        self.play(Write(impl_title), run_time=1)
        self.wait(1)
        
        impl1 = Text("• Multi-layer feature concatenation", font_size=12, color=ACCENT_COLOR)
        impl1.shift(RIGHT * 2.5 + DOWN * 0.6)
        self.play(Write(impl1), run_time=1)
        self.wait(0.5)
        
        impl2 = Text("• Dimensionality reduction", font_size=12, color=ACCENT_COLOR)
        impl2.shift(RIGHT * 2.5 + DOWN * 0.9)
        self.play(Write(impl2), run_time=1)
        self.wait(0.5)
        
        impl3 = Text("• Gaussian parameter fitting", font_size=12, color=ACCENT_COLOR)
        impl3.shift(RIGHT * 2.5 + DOWN * 1.2)
        self.play(Write(impl3), run_time=1)
        self.wait(0.5)
        
        impl4 = Text("• Mahalanobis distance computation", font_size=12, color=ACCENT_COLOR)
        impl4.shift(RIGHT * 2.5 + DOWN * 1.5)
        self.play(Write(impl4), run_time=1)
        self.wait(3)
        
        # Clear everything
        self.play(FadeOut(solution_title), FadeOut(core_title), FadeOut(step1), FadeOut(step2),
                  FadeOut(step3), FadeOut(step4), FadeOut(impl_title), FadeOut(impl1),
                  FadeOut(impl2), FadeOut(impl3), FadeOut(impl4), run_time=1)
        self.strict_clear()
    
    def patch_extraction_process(self):
        """Scene 3: Patch Extraction Process - Foundation - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Foundation: Patch Extraction Process", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Input image
        input_img = Square(side_length=3, color=SECONDARY_COLOR, fill_opacity=0.3)
        input_img.shift(UP * 1)
        self.play(Create(input_img), run_time=2)
        self.wait(1)
        
        # Input label
        input_label = Text("Input Image (224×224)", font_size=18, color=SECONDARY_COLOR, weight=BOLD)
        input_label.next_to(input_img, UP, buff=0.3)
        self.play(Write(input_label), run_time=1)
        self.wait(1)
        
        # Patch extraction title
        patch_title = Text("Patch Extraction Process", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        patch_title.to_edge(UP, buff=0.5)
        self.play(Write(patch_title), run_time=1)
        self.wait(1)
        
        # Create patches one by one - NO OVERLAP
        patches = VGroup()
        patch_size = 0.4
        for i in range(4):
            for j in range(4):
                patch = Square(side_length=patch_size, color=PRIMARY_COLOR, stroke_width=2)
                patch.move_to(input_img.get_center() + RIGHT * (j-1.5) * 0.5 + UP * (1.5-i) * 0.5)
                patches.add(patch)
                self.play(Create(patch), run_time=0.2)
        
        self.wait(1)
        
        # Patch explanation
        patch_text = Text("Extract overlapping patches\n(each patch = feature vector)", 
                         font_size=14, color=PRIMARY_COLOR)
        patch_text.next_to(input_img, DOWN, buff=0.5)
        self.play(Write(patch_text), run_time=2)
        self.wait(2)
        
        # Clear patches and text
        self.play(FadeOut(patches), FadeOut(patch_text), run_time=1)
        self.clear()
        
        # Foundation explanation
        foundation_title = Text("🎯 Foundation for Implementation:", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        foundation_title.shift(DOWN * 1)
        self.play(Write(foundation_title), run_time=1)
        self.wait(1)
        
        foundation1 = Text("• Each patch position (i,j) gets its own", font_size=12, color=WHITE)
        foundation1.shift(DOWN * 1.3)
        self.play(Write(foundation1), run_time=1)
        self.wait(0.5)
        
        foundation2 = Text("• Feature vector from CNN layers", font_size=12, color=WHITE)
        foundation2.shift(DOWN * 1.6)
        self.play(Write(foundation2), run_time=1)
        self.wait(0.5)
        
        foundation3 = Text("• Normal distribution modeling", font_size=12, color=WHITE)
        foundation3.shift(DOWN * 1.9)
        self.play(Write(foundation3), run_time=1)
        self.wait(0.5)
        
        foundation4 = Text("• Anomaly scoring via distance", font_size=12, color=WHITE)
        foundation4.shift(DOWN * 2.2)
        self.play(Write(foundation4), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(input_img), FadeOut(input_label), FadeOut(patch_title),
                  FadeOut(foundation_title), FadeOut(foundation1), FadeOut(foundation2),
                  FadeOut(foundation3), FadeOut(foundation4), run_time=1)
        self.strict_clear() 
    
    def feature_extraction_implementation(self):
        """Scene 4: Feature Extraction Implementation - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Implementation Detail 1: Feature Extraction", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Challenge connection
        challenge_connection = Text("Solving: Need for rich feature representation", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        self.wait(1)
        
        # Single patch focus
        single_patch = Square(side_length=1.5, color=SECONDARY_COLOR, fill_opacity=0.5)
        single_patch.shift(LEFT * 4)
        self.play(Create(single_patch), run_time=2)
        self.wait(1)
        
        # Patch label
        patch_label = Text("Single Patch (i,j)", font_size=16, color=SECONDARY_COLOR, weight=BOLD)
        patch_label.next_to(single_patch, UP, buff=0.3)
        self.play(Write(patch_label), run_time=1)
        self.wait(1)
        
        # CNN processing arrow
        cnn_arrow = Arrow(single_patch.get_right(), RIGHT * 1, color=WHITE, stroke_width=3)
        self.play(Create(cnn_arrow), run_time=1.5)
        self.wait(0.5)
        
        # CNN text
        cnn_text = Text("CNN Processing", font_size=14, color=PRIMARY_COLOR, weight=BOLD)
        cnn_text.next_to(cnn_arrow, UP, buff=0.2)
        self.play(Write(cnn_text), run_time=1)
        self.wait(1)
        
        # Layer 1 feature
        feature1 = Rectangle(width=0.3, height=2, color=BLUE, fill_opacity=0.6)
        feature1.shift(RIGHT * 0.7)
        self.play(Create(feature1), run_time=1)
        self.wait(0.5)
        
        # Layer 1 label
        label1 = VGroup(
            Text("Layer1", font_size=12, color=BLUE, weight=BOLD),
            Text("64D", font_size=10, color=BLUE),
            Text("Edges, textures", font_size=8, color=BLUE)
        ).arrange(DOWN, buff=0.1)
        label1.next_to(feature1, DOWN, buff=0.2)
        self.play(Write(label1), run_time=1)
        self.wait(1)
        
        # Layer 2 feature
        feature2 = Rectangle(width=0.3, height=2, color=GREEN, fill_opacity=0.6)
        feature2.shift(RIGHT * 1.7)
        self.play(Create(feature2), run_time=1)
        self.wait(0.5)
        
        # Layer 2 label
        label2 = VGroup(
            Text("Layer2", font_size=12, color=GREEN, weight=BOLD),
            Text("128D", font_size=10, color=GREEN),
            Text("Simple patterns", font_size=8, color=GREEN)
        ).arrange(DOWN, buff=0.1)
        label2.next_to(feature2, DOWN, buff=0.2)
        self.play(Write(label2), run_time=1)
        self.wait(1)
        
        # Layer 3 feature
        feature3 = Rectangle(width=0.3, height=2, color=PURPLE, fill_opacity=0.6)
        feature3.shift(RIGHT * 2.5)
        self.play(Create(feature3), run_time=1)
        self.wait(0.5)
        
        # Layer 3 label
        label3 = VGroup(
            Text("Layer3", font_size=12, color=PURPLE, weight=BOLD),
            Text("256D", font_size=10, color=PURPLE),
            Text("Complex features", font_size=8, color=PURPLE)
        ).arrange(DOWN, buff=0.1)
        label3.next_to(feature3, DOWN, buff=0.2)
        self.play(Write(label3), run_time=1)
        self.wait(1)
        
        # Concatenation arrow
        concat_arrow = Arrow(feature3.get_right(), RIGHT * 3.9, color=ACCENT_COLOR, stroke_width=4)
        self.play(Create(concat_arrow), run_time=1)
        self.wait(0.5)
        
        # Concatenation text
        concat_text = Text("Concatenate", font_size=14, color=ACCENT_COLOR, weight=BOLD)
        concat_text.next_to(concat_arrow, UP*0.005, buff=0.2)
        self.play(Write(concat_text), run_time=1)
        self.wait(1)
        
        # Final concatenated vector
        final_vector = Rectangle(width=0.5, height=3, color=ACCENT_COLOR, fill_opacity=0.7)
        final_vector.shift(RIGHT * 4.5)
        self.play(Create(final_vector), run_time=2)
        self.wait(1)
        
        # Final label
        final_label = VGroup(
            Text("Concat", font_size=12, color=ACCENT_COLOR, weight=BOLD),
            Text("448D", font_size=10, color=ACCENT_COLOR),
            Text("(64+128+256)", font_size=8, color=ACCENT_COLOR)
        ).arrange(DOWN, buff=0.1)
        final_label.next_to(final_vector, DOWN, buff=0.2)
        self.play(Write(final_label), run_time=1)
        self.wait(1)
        
        # Solution note
        solution_note = Text("✅ Solves Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD)
        solution_note.shift(DOWN * 2)
        self.play(Write(solution_note), run_time=1)
        self.wait(1)
        
        solution1 = Text("• Rich feature representation", font_size=12, color=SUCCESS_COLOR)
        solution1.shift(DOWN * 2.3)
        self.play(Write(solution1), run_time=1)
        self.wait(0.5)
        
        solution2 = Text("• Captures multiple semantic levels", font_size=12, color=SUCCESS_COLOR)
        solution2.shift(DOWN * 2.6)
        self.play(Write(solution2), run_time=1)
        self.wait(0.5)
        
        solution3 = Text("• Enables precise localization", font_size=12, color=SUCCESS_COLOR)
        solution3.shift(DOWN * 2.9)
        self.play(Write(solution3), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(challenge_connection), FadeOut(single_patch), FadeOut(patch_label),
                  FadeOut(cnn_arrow), FadeOut(cnn_text), FadeOut(feature1), FadeOut(label1),
                  FadeOut(feature2), FadeOut(label2), FadeOut(feature3), FadeOut(label3),
                  FadeOut(concat_arrow), FadeOut(concat_text), FadeOut(final_vector), FadeOut(final_label),
                  FadeOut(solution_note), FadeOut(solution1), FadeOut(solution2), FadeOut(solution3), run_time=1)
        self.strict_clear()
    
    def dimensionality_reduction_implementation(self):
        """Scene 5: Dimensionality Reduction Implementation - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Implementation Detail 2: Dimensionality Reduction", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Challenge connection
        challenge_connection = Text("Solving: Real-time processing requirement", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        self.wait(1)
        
        # High-dimensional features
        high_dim_vector = Rectangle(width=0.4, height=4, color=ACCENT_COLOR, fill_opacity=0.6)
        high_dim_vector.shift(LEFT * 3)
        self.play(Create(high_dim_vector), run_time=2)
        self.wait(1)
        
        # High-dim label
        high_dim_label = VGroup(
            Text("High-Dim Features", font_size=14, color=ACCENT_COLOR, weight=BOLD),
            Text("448 dimensions", font_size=12, color=ACCENT_COLOR),
            Text("(potentially redundant)", font_size=10, color=GRAY)
        ).arrange(DOWN, buff=0.1)
        high_dim_label.next_to(high_dim_vector, DOWN, buff=0.3)
        self.play(Write(high_dim_label), run_time=1)
        self.wait(1)
        
        # Challenge problem
        challenge_problem = Text("❌ Challenge:", font_size=14, color=ACCENT_COLOR, weight=BOLD)
        challenge_problem.shift(RIGHT * 2.5 + UP * 1.5)
        self.play(Write(challenge_problem), run_time=1)
        self.wait(1)
        
        problem1 = Text("• Slow computation", font_size=12, color=ACCENT_COLOR)
        problem1.shift(RIGHT * 2.5 + UP * 1.2)
        self.play(Write(problem1), run_time=1)
        self.wait(0.5)
        
        problem2 = Text("• Memory intensive", font_size=12, color=ACCENT_COLOR)
        problem2.shift(RIGHT * 2.5 + UP * 0.9)
        self.play(Write(problem2), run_time=1)
        self.wait(0.5)
        
        problem3 = Text("• Overfitting risk", font_size=12, color=ACCENT_COLOR)
        problem3.shift(RIGHT * 2.5 + UP * 0.6)
        self.play(Write(problem3), run_time=1)
        self.wait(0.5)
        
        problem4 = Text("• Not suitable for real-time", font_size=12, color=ACCENT_COLOR)
        problem4.shift(RIGHT * 2.5 + UP * 0.3)
        self.play(Write(problem4), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(high_dim_vector), FadeOut(high_dim_label), FadeOut(challenge_problem),
                  FadeOut(problem1), FadeOut(problem2), FadeOut(problem3), FadeOut(problem4), run_time=1)
        self.clear()
        
        # Solution title
        solution_title = Text("Solution: Random Dimensionality Reduction", font_size=18, color=PRIMARY_COLOR, weight=BOLD)
        solution_title.to_edge(UP, buff=0.5)
        self.play(Write(solution_title), run_time=1)
        self.wait(1)
        
        # Process arrow
        arrow = Arrow(LEFT * 3, RIGHT * 2, color=PRIMARY_COLOR, stroke_width=3)
        self.play(Create(arrow), run_time=1)
        self.wait(1)
        
        # Process text
        process_text = VGroup(
            Text("Random Selection", font_size=16, color=PRIMARY_COLOR, weight=BOLD),
            Text("Paper Recommendations:", font_size=12, color=WHITE),
            Text("• ResNet18: 100 features", font_size=10, color=YELLOW),
            Text("• WideResNet50: 550 features", font_size=10, color=YELLOW)
        ).arrange(DOWN, buff=0.2)
        process_text.next_to(arrow, UP, buff=0.3)
        self.play(Write(process_text), run_time=2)
        self.wait(1)
        
        # Reduced vector
        reduced_vector = Rectangle(width=0.3, height=2, color=SUCCESS_COLOR, fill_opacity=0.7)
        reduced_vector.shift(RIGHT * 3.5)
        self.play(Create(reduced_vector), run_time=2)
        self.wait(1)
        
        # Reduced label
        reduced_label = VGroup(
            Text("Reduced Features", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("100 dimensions", font_size=12, color=SUCCESS_COLOR),
            Text("(essential info)", font_size=10, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        reduced_label.next_to(reduced_vector, DOWN, buff=0.3)
        self.play(Write(reduced_label), run_time=1)
        self.wait(1)
        
        # Solution benefits
        solution_benefits = Text("✅ Solves Real-time Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD)
        solution_benefits.shift(DOWN * 2)
        self.play(Write(solution_benefits), run_time=1)
        self.wait(1)
        
        benefit1 = Text("• Faster computation", font_size=12, color=SUCCESS_COLOR)
        benefit1.shift(DOWN * 2.3)
        self.play(Write(benefit1), run_time=1)
        self.wait(0.5)
        
        benefit2 = Text("• Less memory usage", font_size=12, color=SUCCESS_COLOR)
        benefit2.shift(DOWN * 2.6)
        self.play(Write(benefit2), run_time=1)
        self.wait(0.5)
        
        benefit3 = Text("• Reduced overfitting", font_size=12, color=SUCCESS_COLOR)
        benefit3.shift(DOWN * 2.9)
        self.play(Write(benefit3), run_time=1)
        self.wait(0.5)
        
        benefit4 = Text("• Suitable for production", font_size=12, color=SUCCESS_COLOR)
        benefit4.shift(DOWN * 3.2)
        self.play(Write(benefit4), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(challenge_connection), FadeOut(solution_title), FadeOut(arrow), FadeOut(process_text),
                  FadeOut(reduced_vector), FadeOut(reduced_label), FadeOut(solution_benefits),
                  FadeOut(benefit1), FadeOut(benefit2), FadeOut(benefit3), FadeOut(benefit4), run_time=1)
        self.strict_clear()
    
    def gaussian_modeling_implementation(self):
        """Scene 6: Gaussian Distribution Modeling Implementation - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Implementation Detail 3: Gaussian Distribution Modeling", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Challenge connection
        challenge_connection = Text("Solving: Limited training data (only normal samples)", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        self.wait(1)
        
        # Training title
        training_title = Text("Training Phase: Collect Normal Embeddings", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        training_title.next_to(challenge_connection, DOWN, buff=0.5)
        self.play(Write(training_title), run_time=1)
        self.wait(1)
        
        # Training images
        training_images = VGroup()
        for i in range(4):
            img = Square(side_length=1, color=SECONDARY_COLOR, fill_opacity=0.4)
            img.shift(LEFT * 4 + RIGHT * i * 1.2)
            training_images.add(img)
        
        self.play(AnimationGroup(*[Create(img) for img in training_images], lag_ratio=0.2), run_time=2)
        self.wait(1)
        
        # Image labels
        for i, img in enumerate(training_images):
            label = Text(f"Normal {i+1}", font_size=10, color=SECONDARY_COLOR)
            label.next_to(img, UP, buff=0.1)
            self.play(Write(label), run_time=0.5)
        
        self.wait(1)
        
        # Key insight
        key_insight = Text("💡 Key Insight:", font_size=16, color=PRIMARY_COLOR, weight=BOLD)
        key_insight.shift(RIGHT * 2.5 + UP * 1.5)
        self.play(Write(key_insight), run_time=1)
        self.wait(1)
        
        insight1 = Text("Model what's NORMAL", font_size=14, color=PRIMARY_COLOR)
        insight1.shift(RIGHT * 2.5 + UP * 1.2)
        self.play(Write(insight1), run_time=1)
        self.wait(0.5)
        
        insight2 = Text("instead of learning anomalies", font_size=14, color=PRIMARY_COLOR)
        insight2.shift(RIGHT * 2.5 + UP * 0.9)
        self.play(Write(insight2), run_time=1)
        self.wait(1)
        
        insight3 = Text("✅ Solves: Limited training data", font_size=12, color=SUCCESS_COLOR, weight=BOLD)
        insight3.shift(RIGHT * 2.5 + UP * 0.6)
        self.play(Write(insight3), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(training_images), FadeOut(key_insight), FadeOut(insight1), FadeOut(insight2), FadeOut(insight3), run_time=1)
        self.clear()
        
        # Patch position concept
        patch_concept = Text("Each patch position (i,j) across all training images", font_size=14, color=WHITE)
        patch_concept.shift(DOWN * 1.5)
        self.play(Write(patch_concept), run_time=2)
        self.wait(1)
        
        # Clear and show Gaussian fitting
        self.play(FadeOut(patch_concept), run_time=1)
        self.clear()
        
        # Gaussian fitting title
        gaussian_title = Text("Per-Patch Gaussian Distribution Fitting", font_size=18, color=PRIMARY_COLOR, weight=BOLD)
        gaussian_title.next_to(training_title, DOWN, buff=0.5)
        self.play(Write(gaussian_title), run_time=1)
        self.wait(1)
        
        # Show data points for one patch position
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=2,
            axis_config={"color": WHITE, "stroke_width": 2}
        )
        axes.shift(LEFT * 3)
        self.play(Create(axes), run_time=1)
        self.wait(1)
        
        # Add sample data points
        np.random.seed(42)
        points = VGroup()
        for _ in range(15):
            x, y = np.random.multivariate_normal([0, 0], [[0.5, 0.2], [0.2, 0.3]])
            if -2.5 < x < 2.5 and -1.5 < y < 1.5:
                point = Dot(axes.coords_to_point(x, y), color=NORMAL_COLOR, radius=0.05)
                points.add(point)
        
        self.play(AnimationGroup(*[Create(point) for point in points], lag_ratio=0.1), run_time=2)
        self.wait(1)
        
        # Data label
        data_label = Text("Training embeddings\\nfor patch position (i,j)", font_size=12, color=NORMAL_COLOR)
        data_label.next_to(axes, DOWN, buff=0.3)
        self.play(Write(data_label), run_time=1)
        self.wait(1)
        
        # Show fitted Gaussian
        gaussian_ellipse = Ellipse(
            width=2, 
            height=1.5, 
            color=PRIMARY_COLOR, 
            fill_opacity=0.3
        )
        gaussian_ellipse.move_to(axes.get_center())
        self.play(Create(gaussian_ellipse), run_time=2)
        self.wait(1)
        
        # Gaussian label
        gaussian_label = VGroup(
            Text("Fitted Multivariate", font_size=12, color=PRIMARY_COLOR, weight=BOLD),
            Text("Gaussian Distribution", font_size=12, color=PRIMARY_COLOR, weight=BOLD),
            Text("μ(i,j), Σ(i,j)", font_size=10, color=PRIMARY_COLOR)
        ).arrange(DOWN, buff=0.1)
        gaussian_label.shift(RIGHT * 2.5)
        self.play(Write(gaussian_label), run_time=2)
        self.wait(1)
        
        # Solution note
        solution_note = Text("✅ Solves Limited Data Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD)
        solution_note.shift(DOWN * 2)
        self.play(Write(solution_note), run_time=1)
        self.wait(1)
        
        solution1 = Text("• Only needs normal samples", font_size=12, color=SUCCESS_COLOR)
        solution1.shift(DOWN * 2.3)
        self.play(Write(solution1), run_time=1)
        self.wait(0.5)
        
        solution2 = Text("• No anomaly examples required", font_size=12, color=SUCCESS_COLOR)
        solution2.shift(DOWN * 2.6)
        self.play(Write(solution2), run_time=1)
        self.wait(0.5)
        
        solution3 = Text("• Learns normal distribution", font_size=12, color=SUCCESS_COLOR)
        solution3.shift(DOWN * 2.9)
        self.play(Write(solution3), run_time=1)
        self.wait(0.5)
        
        solution4 = Text("• Detects deviations from normal", font_size=12, color=SUCCESS_COLOR)
        solution4.shift(DOWN * 3.2)
        self.play(Write(solution4), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(challenge_connection), FadeOut(training_title), FadeOut(gaussian_title),
                  FadeOut(axes), FadeOut(points), FadeOut(data_label), FadeOut(gaussian_ellipse),
                  FadeOut(gaussian_label), FadeOut(solution_note), FadeOut(solution1),
                  FadeOut(solution2), FadeOut(solution3), FadeOut(solution4), run_time=1)
        self.strict_clear()
        
        # Complete Gaussian map
        complete_title = Text("Complete Gaussian Parameter Map", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        complete_title.to_edge(UP, buff=0.5)
        self.play(Write(complete_title), run_time=1)
        self.wait(1)
        
        # Create grid of Gaussians
        gaussian_grid = VGroup()
        for i in range(4):
            for j in range(4):
                gauss_circle = Circle(radius=0.15, color=PRIMARY_COLOR, fill_opacity=0.6)
                gauss_circle.shift(LEFT * 2 + RIGHT * j * 0.5 + UP * 1 + DOWN * i * 0.5)
                gaussian_grid.add(gauss_circle)
                self.play(Create(gauss_circle), run_time=0.1)
        
        self.wait(1)
        
        # Grid label
        grid_label = VGroup(
            Text("Each position has its own", font_size=14, color=WHITE),
            Text("Gaussian distribution", font_size=14, color=WHITE),
            Text("Parameters: μ(i,j) and Σ(i,j)", font_size=12, color=PRIMARY_COLOR)
        ).arrange(DOWN, buff=0.2)
        grid_label.shift(RIGHT * 2.5)
        self.play(Write(grid_label), run_time=2)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(complete_title), FadeOut(gaussian_grid), FadeOut(grid_label), run_time=1)
        self.strict_clear() 
    
    def mahalanobis_implementation(self):
        """Scene 7: Mahalanobis Distance Implementation - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Implementation Detail 4: Mahalanobis Distance Computation", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Challenge connection
        challenge_connection = Text("Solving: Need for precise pixel-level localization", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        self.wait(1)
        
        # Title
        title = Text("Anomaly Scoring with Mahalanobis Distance", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        title.next_to(challenge_connection, DOWN, buff=0.5)
        self.play(Write(title), run_time=1)
        self.wait(1)
        
        # Formula
        formula_text = Text("M(x) = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]", font_size=18, color=ACCENT_COLOR)
        self.play(Write(formula_text), run_time=2)
        self.wait(1)
        
        # Formula explanation
        formula_explanation = Text("x: test embedding, μ: mean, Σ: covariance", font_size=12, color=WHITE)
        formula_explanation.next_to(formula_text, DOWN, buff=0.3)
        self.play(Write(formula_explanation), run_time=1)
        self.wait(1)
        
        # Localization explanation
        localization_title = Text("🎯 Enables Precise Localization:", font_size=14, color=PRIMARY_COLOR, weight=BOLD)
        localization_title.shift(LEFT * 2.5 + UP * 1)
        self.play(Write(localization_title), run_time=1)
        self.wait(1)
        
        loc1 = Text("• Each patch gets its own score", font_size=12, color=WHITE)
        loc1.shift(LEFT * 2.5 + UP * 0.7)
        self.play(Write(loc1), run_time=1)
        self.wait(0.5)
        
        loc2 = Text("• Pixel-level anomaly detection", font_size=12, color=WHITE)
        loc2.shift(LEFT * 2.5 + UP * 0.4)
        self.play(Write(loc2), run_time=1)
        self.wait(0.5)
        
        loc3 = Text("• Exact defect location identification", font_size=12, color=WHITE)
        loc3.shift(LEFT * 2.5 + UP * 0.1)
        self.play(Write(loc3), run_time=1)
        self.wait(0.5)
        
        loc4 = Text("• No global image-level decisions", font_size=12, color=WHITE)
        loc4.shift(LEFT * 2.5 + DOWN * 0.2)
        self.play(Write(loc4), run_time=1)
        self.wait(1)
        
        # Clear left side
        self.play(FadeOut(localization_title), FadeOut(loc1), FadeOut(loc2), FadeOut(loc3), FadeOut(loc4), run_time=1)
        self.clear()
        
        # Visual representation - Normal case
        normal_title = Text("Normal Case", font_size=14, color=NORMAL_COLOR, weight=BOLD)
        normal_title.shift(LEFT * 3 + UP * 1.5)
        self.play(Write(normal_title), run_time=1)
        self.wait(1)
        
        normal_axes = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=2, y_length=2)
        normal_axes.shift(LEFT * 3 + UP * 0.5)
        self.play(Create(normal_axes), run_time=1)
        self.wait(1)
        
        normal_gaussian = Circle(radius=0.5, color=NORMAL_COLOR, fill_opacity=0.3)
        normal_gaussian.move_to(normal_axes.get_center())
        self.play(Create(normal_gaussian), run_time=1)
        self.wait(1)
        
        normal_point = Dot(normal_axes.coords_to_point(0.2, 0.1), color=NORMAL_COLOR, radius=0.08)
        self.play(Create(normal_point), run_time=1)
        self.wait(1)
        
        normal_label = Text("Normal\\n(Low Distance)", font_size=12, color=NORMAL_COLOR)
        normal_label.next_to(normal_axes, DOWN, buff=0.3)
        self.play(Write(normal_label), run_time=1)
        self.wait(1)
        
        # Anomaly case
        anomaly_title = Text("Anomaly Case", font_size=14, color=ANOMALY_COLOR, weight=BOLD)
        anomaly_title.shift(RIGHT * 3 + UP * 1.5)
        self.play(Write(anomaly_title), run_time=1)
        self.wait(1)
        
        anomaly_axes = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=2, y_length=2)
        anomaly_axes.shift(RIGHT * 3 + UP * 0.5)
        self.play(Create(anomaly_axes), run_time=1)
        self.wait(1)
        
        anomaly_gaussian = Circle(radius=0.5, color=ANOMALY_COLOR, fill_opacity=0.3)
        anomaly_gaussian.move_to(anomaly_axes.get_center())
        self.play(Create(anomaly_gaussian), run_time=1)
        self.wait(1)
        
        anomaly_point = Dot(anomaly_axes.coords_to_point(1.5, 1.2), color=ANOMALY_COLOR, radius=0.08)
        self.play(Create(anomaly_point), run_time=1)
        self.wait(1)
        
        anomaly_label = Text("Anomaly\\n(High Distance)", font_size=12, color=ANOMALY_COLOR)
        anomaly_label.next_to(anomaly_axes, DOWN, buff=0.3)
        self.play(Write(anomaly_label), run_time=1)
        self.wait(1)
        
        # Distance lines
        normal_line = Line(normal_gaussian.get_center(), normal_point.get_center(), color=NORMAL_COLOR, stroke_width=3)
        self.play(Create(normal_line), run_time=1)
        self.wait(0.5)
        
        anomaly_line = Line(anomaly_gaussian.get_center(), anomaly_point.get_center(), color=ANOMALY_COLOR, stroke_width=3)
        self.play(Create(anomaly_line), run_time=1)
        self.wait(1)
        
        # Solution note
        solution_note = Text("✅ Solves Localization Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD)
        solution_note.shift(DOWN * 2)
        self.play(Write(solution_note), run_time=1)
        self.wait(1)
        
        solution1 = Text("• Precise pixel-level detection", font_size=12, color=SUCCESS_COLOR)
        solution1.shift(DOWN * 2.3)
        self.play(Write(solution1), run_time=1)
        self.wait(0.5)
        
        solution2 = Text("• Exact defect boundaries", font_size=12, color=SUCCESS_COLOR)
        solution2.shift(DOWN * 2.6)
        self.play(Write(solution2), run_time=1)
        self.wait(0.5)
        
        solution3 = Text("• Quantitative anomaly scores", font_size=12, color=SUCCESS_COLOR)
        solution3.shift(DOWN * 2.9)
        self.play(Write(solution3), run_time=1)
        self.wait(0.5)
        
        solution4 = Text("• Interpretable results", font_size=12, color=SUCCESS_COLOR)
        solution4.shift(DOWN * 3.2)
        self.play(Write(solution4), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(challenge_connection), FadeOut(title), FadeOut(formula_text), FadeOut(formula_explanation),
                  FadeOut(normal_title), FadeOut(normal_axes), FadeOut(normal_gaussian), FadeOut(normal_point),
                  FadeOut(normal_label), FadeOut(normal_line), FadeOut(anomaly_title), FadeOut(anomaly_axes),
                  FadeOut(anomaly_gaussian), FadeOut(anomaly_point), FadeOut(anomaly_label), FadeOut(anomaly_line),
                  FadeOut(solution_note), FadeOut(solution1), FadeOut(solution2), FadeOut(solution3), FadeOut(solution4), run_time=1)
        self.strict_clear()
    
    def complete_pipeline(self):
        """Scene 8: Complete training pipeline - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Complete Training Pipeline", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Challenge connection
        challenge_connection = Text("Solving: All industrial challenges with one unified approach", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        self.wait(1)
        
        # Pipeline steps
        steps = [
            ("1. Extract patches", SECONDARY_COLOR),
            ("2. Get CNN features", BLUE),
            ("3. Concatenate layers", GREEN),
            ("4. Reduce dimensions", PURPLE),
            ("5. Fit Gaussians", PRIMARY_COLOR)
        ]
        
        pipeline = VGroup()
        for i, (step, color) in enumerate(steps):
            step_box = Rectangle(width=2.5, height=0.8, color=color, fill_opacity=0.3)
            step_box.shift(UP * 2 + DOWN * i * 1)
            self.play(Create(step_box), run_time=1)
            self.wait(0.5)
            
            step_text = Text(step, font_size=12, color=color, weight=BOLD)
            step_text.move_to(step_box.get_center())
            self.play(Write(step_text), run_time=1)
            self.wait(0.5)
            
            pipeline.add(VGroup(step_box, step_text))
        
        # Add arrows between steps
        arrows = VGroup()
        for i in range(len(steps) - 1):
            arrow = Arrow(pipeline[i].get_bottom(), pipeline[i+1].get_top(), color=WHITE, stroke_width=2)
            arrows.add(arrow)
            self.play(Create(arrow), run_time=1)
            self.wait(0.5)
        
        # Timing information
        timing_title = Text("Training Time:", font_size=14, color=WHITE, weight=BOLD)
        timing_title.shift(RIGHT * 3 + UP * 1.5)
        self.play(Write(timing_title), run_time=1)
        self.wait(1)
        
        timing1 = Text("• Fast: No gradient computation", font_size=12, color=SUCCESS_COLOR)
        timing1.shift(RIGHT * 3 + UP * 1.2)
        self.play(Write(timing1), run_time=1)
        self.wait(0.5)
        
        timing2 = Text("• Only forward passes needed", font_size=12, color=SUCCESS_COLOR)
        timing2.shift(RIGHT * 3 + UP * 0.9)
        self.play(Write(timing2), run_time=1)
        self.wait(0.5)
        
        timing3 = Text("• Gaussian fitting is efficient", font_size=12, color=SUCCESS_COLOR)
        timing3.shift(RIGHT * 3 + UP * 0.6)
        self.play(Write(timing3), run_time=1)
        self.wait(1)
        
        # All solutions
        all_solutions_title = Text("✅ Solves ALL Challenges:", font_size=14, color=SUCCESS_COLOR, weight=BOLD)
        all_solutions_title.shift(DOWN * 2)
        self.play(Write(all_solutions_title), run_time=1)
        self.wait(1)
        
        all_sol1 = Text("• Limited data → Multi-layer features + Gaussian modeling", font_size=12, color=SUCCESS_COLOR)
        all_sol1.shift(DOWN * 2.3)
        self.play(Write(all_sol1), run_time=1)
        self.wait(0.5)
        
        all_sol2 = Text("• Precise localization → Per-patch Mahalanobis scoring", font_size=12, color=SUCCESS_COLOR)
        all_sol2.shift(DOWN * 2.6)
        self.play(Write(all_sol2), run_time=1)
        self.wait(0.5)
        
        all_sol3 = Text("• Real-time processing → Dimensionality reduction", font_size=12, color=SUCCESS_COLOR)
        all_sol3.shift(DOWN * 2.9)
        self.play(Write(all_sol3), run_time=1)
        self.wait(3)
        
        # Clear everything
        self.play(FadeOut(challenge_connection), FadeOut(pipeline), FadeOut(arrows),
                  FadeOut(timing_title), FadeOut(timing1), FadeOut(timing2), FadeOut(timing3),
                  FadeOut(all_solutions_title), FadeOut(all_sol1), FadeOut(all_sol2), FadeOut(all_sol3), run_time=1)
        self.strict_clear()
    
    def inference_demo(self):
        """Scene 9: Inference process demo - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Inference Process", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Challenge connection
        challenge_connection = Text("Solving: Real-time production line deployment", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        self.wait(1)
        
        # Test image
        test_img = Square(side_length=2, color=ACCENT_COLOR, fill_opacity=0.4)
        test_img.shift(LEFT * 4)
        self.play(Create(test_img), run_time=1)
        self.wait(1)
        
        # Test label
        test_label = Text("Test Image", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        test_label.next_to(test_img, UP, buff=0.3)
        self.play(Write(test_label), run_time=1)
        self.wait(1)
        
        # Processing arrow
        arrow1 = Arrow(test_img.get_right(), RIGHT * 1.5, color=WHITE, stroke_width=2)
        self.play(Create(arrow1), run_time=1)
        self.wait(0.5)
        
        # Processing text
        processing = Text("Same feature\\nextraction process", font_size=12, color=WHITE)
        processing.next_to(arrow1, UP, buff=0.2)
        self.play(Write(processing), run_time=1)
        self.wait(1)
        
        # Features
        features = Rectangle(width=0.4, height=2, color=PRIMARY_COLOR, fill_opacity=0.6)
        features.shift(RIGHT * 0.5)
        self.play(Create(features), run_time=1)
        self.wait(1)
        
        # Distance arrow
        arrow2 = Arrow(features.get_right(), RIGHT * 2, color=WHITE, stroke_width=2)
        self.play(Create(arrow2), run_time=1)
        self.wait(0.5)
        
        # Distance text
        distance_comp = Text("Mahalanobis\\ndistance", font_size=12, color=WHITE)
        distance_comp.next_to(arrow2, UP, buff=0.2)
        self.play(Write(distance_comp), run_time=1)
        self.wait(1)
        
        # Anomaly map
        anomaly_map = VGroup()
        for i in range(6):
            for j in range(6):
                intensity = 0.3 + 0.4 * np.random.random()
                if intensity > 0.6:
                    color = ANOMALY_COLOR
                elif intensity > 0.4:
                    color = YELLOW
                else:
                    color = NORMAL_COLOR
                pixel = Square(side_length=0.15, color=color, fill_opacity=0.8, stroke_width=0)
                pixel.shift(RIGHT * 3.5 + RIGHT * j * 0.15 + UP * 0.5 + DOWN * i * 0.15)
                anomaly_map.add(pixel)
                self.play(Create(pixel), run_time=0.05)
        
        self.wait(1)
        
        # Map label
        map_label = Text("Anomaly Map", font_size=14, color=ACCENT_COLOR, weight=BOLD)
        map_label.next_to(anomaly_map, DOWN, buff=0.3)
        self.play(Write(map_label), run_time=1)
        self.wait(1)
        
        # Real-time solution
        real_time_title = Text("✅ Solves Real-time Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD)
        real_time_title.shift(DOWN * 2)
        self.play(Write(real_time_title), run_time=1)
        self.wait(1)
        
        rt1 = Text("• Fast inference for production", font_size=12, color=SUCCESS_COLOR)
        rt1.shift(DOWN * 2.3)
        self.play(Write(rt1), run_time=1)
        self.wait(0.5)
        
        rt2 = Text("• Immediate defect detection", font_size=12, color=SUCCESS_COLOR)
        rt2.shift(DOWN * 2.6)
        self.play(Write(rt2), run_time=1)
        self.wait(0.5)
        
        rt3 = Text("• Precise localization output", font_size=12, color=SUCCESS_COLOR)
        rt3.shift(DOWN * 2.9)
        self.play(Write(rt3), run_time=1)
        self.wait(0.5)
        
        rt4 = Text("• Ready for deployment", font_size=12, color=SUCCESS_COLOR)
        rt4.shift(DOWN * 3.2)
        self.play(Write(rt4), run_time=1)
        self.wait(2)
        
        # Clear everything
        self.play(FadeOut(challenge_connection), FadeOut(test_img), FadeOut(test_label),
                  FadeOut(arrow1), FadeOut(processing), FadeOut(features), FadeOut(arrow2),
                  FadeOut(distance_comp), FadeOut(anomaly_map), FadeOut(map_label),
                  FadeOut(real_time_title), FadeOut(rt1), FadeOut(rt2), FadeOut(rt3), FadeOut(rt4), run_time=1)
        self.strict_clear()
    
    def finale(self):
        """Final scene with key takeaways - NO OVERLAP"""
        # Chapter title
        chapter_title = Text("Key Takeaways", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
        
        # Final message
        final_title = Text("PaDiM: Complete Industrial Solution", font_size=32, color=PRIMARY_COLOR, weight=BOLD)
        self.play(Write(final_title), run_time=2)
        self.wait(2)
        self.play(FadeOut(final_title), run_time=1)
        self.clear()
        
        # Challenge → Solution
        challenge_solution_title = Text("🎯 Challenge → Solution:", font_size=18, color=ACCENT_COLOR, weight=BOLD)
        self.play(Write(challenge_solution_title), run_time=1)
        self.wait(1)
        
        cs1 = Text("• Limited data → Multi-layer features + Gaussian modeling", font_size=14, color=SECONDARY_COLOR)
        cs1.shift(DOWN * 0.3)
        self.play(Write(cs1), run_time=1)
        self.wait(0.5)
        
        cs2 = Text("• Precise localization → Per-patch Mahalanobis scoring", font_size=14, color=SECONDARY_COLOR)
        cs2.shift(DOWN * 0.6)
        self.play(Write(cs2), run_time=1)
        self.wait(0.5)
        
        cs3 = Text("• Real-time processing → Dimensionality reduction", font_size=14, color=SECONDARY_COLOR)
        cs3.shift(DOWN * 0.9)
        self.play(Write(cs3), run_time=1)
        self.wait(2)
        
        # Clear challenge-solution
        self.play(FadeOut(challenge_solution_title), FadeOut(cs1), FadeOut(cs2), FadeOut(cs3), run_time=1)
        self.clear()
        
        # Implementation details
        impl_title = Text("✨ Implementation Details:", font_size=16, color=PRIMARY_COLOR, weight=BOLD)
        self.play(Write(impl_title), run_time=1)
        self.wait(1)
        
        impl1 = Text("• Layer concatenation: [layer1, layer2, layer3]", font_size=12, color=WHITE)
        impl1.shift(DOWN * 0.3)
        self.play(Write(impl1), run_time=1)
        self.wait(0.5)
        
        impl2 = Text("• Random selection: 100-550 features", font_size=12, color=WHITE)
        impl2.shift(DOWN * 0.6)
        self.play(Write(impl2), run_time=1)
        self.wait(0.5)
        
        impl3 = Text("• Gaussian fitting: μ(i,j), Σ(i,j) per patch", font_size=12, color=WHITE)
        impl3.shift(DOWN * 0.9)
        self.play(Write(impl3), run_time=1)
        self.wait(0.5)
        
        impl4 = Text("• Distance scoring: M(x) = √[(x-μ)ᵀΣ⁻¹(x-μ)]", font_size=12, color=WHITE)
        impl4.shift(DOWN * 1.2)
        self.play(Write(impl4), run_time=1)
        self.wait(2)
        
        # Clear implementation
        self.play(FadeOut(impl_title), FadeOut(impl1), FadeOut(impl2), FadeOut(impl3), FadeOut(impl4), run_time=1)
        self.clear()
        
        # Final result
        final_result = Text("🚀 Result: Production-ready anomaly detection!", font_size=18, color=SUCCESS_COLOR, weight=BOLD)
        self.play(Write(final_result), run_time=2)
        self.wait(3) 
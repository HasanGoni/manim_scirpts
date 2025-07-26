from manim import *
import numpy as np

# Animation Studio Color Palette
PRIMARY_COLOR = BLUE
SECONDARY_COLOR = YELLOW  
ACCENT_COLOR = RED
SUCCESS_COLOR = GREEN
NORMAL_COLOR = "#2E8B57"  # Sea Green for normal samples
ANOMALY_COLOR = "#DC143C"  # Crimson for anomalies

# Spatial Organization Constants
LEFT_ZONE = LEFT * 3
RIGHT_ZONE = RIGHT * 3
TITLE_ZONE = UP * 3
WORK_ZONE = ORIGIN

class PaDiMFinal(Scene):
    """
    Final PaDiM Animation - Strictly Following Cursor Rules
    
    What it shows: Complete PaDiM methodology with proper patch extraction focus
    Duration: ~4-5 minutes
    Key concepts: Challenge → Solution → Patch Extraction → Implementation Details
    Cleanup: Proper scene transitions with self.clear() - NO OVERLAP
    """
    
    def construct(self):
        """Complete PaDiM story with strict no-overlap rules"""
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
    
    def chapter_transition(self, title):
        """Clean transition between major topics - Stage Director Rule"""
        if self.mobjects:
            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        
        chapter_title = Text(title, font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        chapter_title.to_edge(UP, buff=0.5)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)
        self.clear()
    
    def introduction(self):
        """Opening scene with PaDiM overview"""
        title = VGroup(
            Text("PaDiM", font_size=56, color=PRIMARY_COLOR, weight=BOLD),
            Text("Patch Distribution Modeling", font_size=32, color=SECONDARY_COLOR),
            Text("Solving Industrial Anomaly Detection", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.5)
        
        self.play(Write(title, lag_ratio=0.3, run_time=3))
        self.wait(2)
        
        # Clear for next scene - Stage Director Rule
        self.play(FadeOut(title), run_time=1.5)
        self.clear()
    
    def the_challenge(self):
        """Scene 1: The Industrial Anomaly Detection Challenge"""
        self.chapter_transition("The Challenge")
        
        # Challenge title
        challenge_title = Text("Industrial Anomaly Detection Challenge", font_size=24, color=ACCENT_COLOR, weight=BOLD)
        challenge_title.to_edge(UP, buff=0.5)
        self.play(Write(challenge_title), run_time=1)
        
        # Key challenges - LEFT SIDE
        challenges = VGroup(
            Text("❌ Limited Training Data", font_size=18, color=ACCENT_COLOR, weight=BOLD),
            Text("• Only normal samples available", font_size=14, color=WHITE),
            Text("• Anomalies are rare and unknown", font_size=14, color=WHITE),
            Text("", font_size=8),  # spacer
            Text("❌ Need for Precise Localization", font_size=18, color=ACCENT_COLOR, weight=BOLD),
            Text("• Must identify exact defect location", font_size=14, color=WHITE),
            Text("• Pixel-level accuracy required", font_size=14, color=WHITE),
            Text("", font_size=8),  # spacer
            Text("❌ Real-time Processing", font_size=18, color=ACCENT_COLOR, weight=BOLD),
            Text("• Fast inference for production lines", font_size=14, color=WHITE),
            Text("• No complex training needed", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        challenges.shift(LEFT * 2)
        
        self.play(Write(challenges), run_time=3)
        self.wait(2)
        
        # Show example industrial scenario - RIGHT SIDE
        scenario = VGroup(
            Text("🏭 Manufacturing Scenario:", font_size=16, color=SECONDARY_COLOR, weight=BOLD),
            Text("• Quality control on production line", font_size=12, color=WHITE),
            Text("• Detect scratches, dents, color variations", font_size=12, color=WHITE),
            Text("• Must work with minimal normal data", font_size=12, color=WHITE),
            Text("• Real-time decision making", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        scenario.shift(RIGHT * 2.5)
        
        self.play(Write(scenario), run_time=2)
        self.wait(2)
        
        # Clear for solution - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def solution_overview(self):
        """Scene 2: PaDiM Solution Overview"""
        self.chapter_transition("PaDiM Solution")
        
        # Solution approach
        solution_title = Text("PaDiM: Patch Distribution Modeling", font_size=24, color=PRIMARY_COLOR, weight=BOLD)
        solution_title.to_edge(UP, buff=0.5)
        self.play(Write(solution_title), run_time=1)
        
        # Key insights - LEFT SIDE
        insights = VGroup(
            Text("💡 Key Insight:", font_size=18, color=PRIMARY_COLOR, weight=BOLD),
            Text("Model normal patch distributions", font_size=16, color=WHITE),
            Text("instead of learning anomalies", font_size=16, color=WHITE),
            Text("", font_size=8),  # spacer
            Text("✅ Advantages:", font_size=18, color=SUCCESS_COLOR, weight=BOLD),
            Text("• Only needs normal samples", font_size=14, color=SUCCESS_COLOR),
            Text("• Precise pixel-level localization", font_size=14, color=SUCCESS_COLOR),
            Text("• Fast training & inference", font_size=14, color=SUCCESS_COLOR),
            Text("• Works with any pre-trained CNN", font_size=14, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        insights.shift(LEFT * 2)
        
        self.play(Write(insights), run_time=3)
        
        # Show the core idea visually - RIGHT SIDE
        core_idea = VGroup(
            Text("🎯 Core Idea:", font_size=16, color=SECONDARY_COLOR, weight=BOLD),
            Text("1. Extract patches from image", font_size=12, color=WHITE),
            Text("2. Get features from CNN layers", font_size=12, color=WHITE),
            Text("3. Model normal distribution per patch", font_size=12, color=WHITE),
            Text("4. Detect anomalies via distance scoring", font_size=12, color=WHITE),
            Text("", font_size=8),  # spacer
            Text("🔬 Implementation Details:", font_size=14, color=ACCENT_COLOR, weight=BOLD),
            Text("• Multi-layer feature concatenation", font_size=12, color=ACCENT_COLOR),
            Text("• Dimensionality reduction", font_size=12, color=ACCENT_COLOR),
            Text("• Gaussian parameter fitting", font_size=12, color=ACCENT_COLOR),
            Text("• Mahalanobis distance computation", font_size=12, color=ACCENT_COLOR)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        core_idea.shift(RIGHT * 2.5)
        
        self.play(Write(core_idea), run_time=3)
        self.wait(3)
        
        # Clear for detailed implementation - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def patch_extraction_process(self):
        """Scene 3: Patch Extraction Process - Foundation for Implementation"""
        self.chapter_transition("Foundation: Patch Extraction Process")
        
        # Show input image first
        input_img = Square(side_length=3, color=SECONDARY_COLOR, fill_opacity=0.3)
        input_label = Text("Input Image (224×224)", font_size=18, color=SECONDARY_COLOR, weight=BOLD)
        input_label.next_to(input_img, UP, buff=0.3)
        
        input_group = VGroup(input_img, input_label)
        input_group.shift(UP * 1)
        
        self.play(Create(input_group), run_time=2)
        self.wait(1)
        
        # Show patch extraction process
        patch_extraction_title = Text("Patch Extraction Process", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        patch_extraction_title.to_edge(UP, buff=0.5)
        self.play(Write(patch_extraction_title), run_time=1)
        
        # Create grid of patches
        patches = VGroup()
        patch_size = 0.4
        for i in range(4):
            for j in range(4):
                patch = Square(side_length=patch_size, color=PRIMARY_COLOR, stroke_width=2)
                patch.move_to(input_img.get_center() + RIGHT * (j-1.5) * 0.5 + UP * (1.5-i) * 0.5)
                patches.add(patch)
        
        patch_text = Text("Extract overlapping patches\\n(each patch = feature vector)", 
                         font_size=14, color=PRIMARY_COLOR)
        patch_text.next_to(input_img, DOWN, buff=0.5)
        
        self.play(AnimationGroup(*[Create(patch) for patch in patches], lag_ratio=0.1), run_time=2)
        self.play(Write(patch_text), run_time=1)
        self.wait(2)
        
        # Show how patches become the foundation
        foundation_note = VGroup(
            Text("🎯 Foundation for Implementation:", font_size=16, color=ACCENT_COLOR, weight=BOLD),
            Text("• Each patch position (i,j) gets its own", font_size=12, color=WHITE),
            Text("• Feature vector from CNN layers", font_size=12, color=WHITE),
            Text("• Normal distribution modeling", font_size=12, color=WHITE),
            Text("• Anomaly scoring via distance", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        foundation_note.shift(DOWN * 2)
        
        self.play(Write(foundation_note), run_time=2)
        self.wait(2)
        
        # Clear for implementation details - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def feature_extraction_implementation(self):
        """Scene 4: Feature Extraction Implementation Detail"""
        self.chapter_transition("Implementation Detail 1: Feature Extraction")
        
        # Connect to challenge
        challenge_connection = Text("Solving: Need for rich feature representation", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        
        # Show single patch focus
        single_patch = Square(side_length=1.5, color=SECONDARY_COLOR, fill_opacity=0.5)
        patch_label = Text("Single Patch (i,j)", font_size=16, color=SECONDARY_COLOR, weight=BOLD)
        patch_label.next_to(single_patch, UP, buff=0.3)
        
        patch_group = VGroup(single_patch, patch_label)
        patch_group.shift(LEFT * 4)
        
        self.play(Create(patch_group), run_time=2)
        self.wait(1)
        
        # Show CNN processing
        cnn_arrow = Arrow(patch_group.get_right(), RIGHT * 1, color=WHITE, stroke_width=3)
        cnn_text = Text("CNN Processing", font_size=14, color=PRIMARY_COLOR, weight=BOLD)
        cnn_text.next_to(cnn_arrow, UP, buff=0.2)
        
        self.play(Create(cnn_arrow), Write(cnn_text), run_time=1.5)
        
        # Show feature vectors from different layers
        feature_vectors = VGroup()
        layer_info = [
            ("Layer1", 64, BLUE, "Edges, textures"),
            ("Layer2", 128, GREEN, "Simple patterns"), 
            ("Layer3", 256, PURPLE, "Complex features")
        ]
        
        for i, (layer_name, dim, color, description) in enumerate(layer_info):
            # Feature vector representation
            feature_rect = Rectangle(width=0.3, height=2, color=color, fill_opacity=0.6)
            feature_label = VGroup(
                Text(layer_name, font_size=12, color=color, weight=BOLD),
                Text(f"{dim}D", font_size=10, color=color),
                Text(description, font_size=8, color=color)
            ).arrange(DOWN, buff=0.1)
            feature_label.next_to(feature_rect, DOWN, buff=0.2)
            
            feature_group = VGroup(feature_rect, feature_label)
            feature_group.shift(RIGHT * 0.5 + RIGHT * i * 1.2)
            feature_vectors.add(feature_group)
        
        # Animate feature extraction
        for i, feature in enumerate(feature_vectors):
            arrow = Arrow(cnn_arrow.get_end(), feature.get_left(), color=WHITE, stroke_width=2)
            self.play(Create(arrow), Create(feature), run_time=1)
            self.play(FadeOut(arrow), run_time=0.3)
        
        self.wait(1)
        
        # Show concatenation
        concat_arrow = Arrow(feature_vectors.get_right(), RIGHT * 2, color=ACCENT_COLOR, stroke_width=4)
        concat_text = Text("Concatenate", font_size=14, color=ACCENT_COLOR, weight=BOLD)
        concat_text.next_to(concat_arrow, UP, buff=0.2)
        
        # Final concatenated vector
        final_vector = Rectangle(width=0.5, height=3, color=ACCENT_COLOR, fill_opacity=0.7)
        final_label = VGroup(
            Text("Concat", font_size=12, color=ACCENT_COLOR, weight=BOLD),
            Text("448D", font_size=10, color=ACCENT_COLOR),
            Text("(64+128+256)", font_size=8, color=ACCENT_COLOR)
        ).arrange(DOWN, buff=0.1)
        final_label.next_to(final_vector, DOWN, buff=0.2)
        
        final_group = VGroup(final_vector, final_label)
        final_group.shift(RIGHT * 3)
        
        self.play(Create(concat_arrow), Write(concat_text), run_time=1)
        self.play(Create(final_group), run_time=2)
        
        # Show how this solves the challenge
        solution_note = VGroup(
            Text("✅ Solves Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("• Rich feature representation", font_size=12, color=SUCCESS_COLOR),
            Text("• Captures multiple semantic levels", font_size=12, color=SUCCESS_COLOR),
            Text("• Enables precise localization", font_size=12, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        solution_note.shift(DOWN * 2)
        
        self.play(Write(solution_note), run_time=2)
        self.wait(2)
        
        # Clear for next scene - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear() 
    
    def dimensionality_reduction_implementation(self):
        """Scene 5: Dimensionality Reduction Implementation Detail"""
        self.chapter_transition("Implementation Detail 2: Dimensionality Reduction")
        
        # Connect to challenge
        challenge_connection = Text("Solving: Real-time processing requirement", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        
        # Show the problem: high-dimensional features
        high_dim_vector = Rectangle(width=0.4, height=4, color=ACCENT_COLOR, fill_opacity=0.6)
        high_dim_label = VGroup(
            Text("High-Dim Features", font_size=14, color=ACCENT_COLOR, weight=BOLD),
            Text("448 dimensions", font_size=12, color=ACCENT_COLOR),
            Text("(potentially redundant)", font_size=10, color=GRAY)
        ).arrange(DOWN, buff=0.1)
        high_dim_label.next_to(high_dim_vector, DOWN, buff=0.3)
        
        high_dim_group = VGroup(high_dim_vector, high_dim_label)
        high_dim_group.shift(LEFT * 3)
        
        self.play(Create(high_dim_group), run_time=2)
        self.wait(1)
        
        # Show the challenge this creates
        challenge_problem = VGroup(
            Text("❌ Challenge:", font_size=14, color=ACCENT_COLOR, weight=BOLD),
            Text("• Slow computation", font_size=12, color=ACCENT_COLOR),
            Text("• Memory intensive", font_size=12, color=ACCENT_COLOR),
            Text("• Overfitting risk", font_size=12, color=ACCENT_COLOR),
            Text("• Not suitable for real-time", font_size=12, color=ACCENT_COLOR)
        ).arrange(DOWN, buff=0.1)
        challenge_problem.shift(RIGHT * 2.5)
        
        self.play(Write(challenge_problem), run_time=2)
        self.wait(1)
        
        # Clear and show solution
        self.play(FadeOut(high_dim_group), FadeOut(challenge_problem), run_time=1)
        self.clear()
        
        # Show random selection process
        solution_title = Text("Solution: Random Dimensionality Reduction", font_size=18, color=PRIMARY_COLOR, weight=BOLD)
        solution_title.to_edge(UP, buff=0.5)
        self.play(Write(solution_title), run_time=1)
        
        arrow = Arrow(LEFT * 3, RIGHT * 2, color=PRIMARY_COLOR, stroke_width=3)
        process_text = VGroup(
            Text("Random Selection", font_size=16, color=PRIMARY_COLOR, weight=BOLD),
            Text("Paper Recommendations:", font_size=12, color=WHITE),
            Text("• ResNet18: 100 features", font_size=10, color=YELLOW),
            Text("• WideResNet50: 550 features", font_size=10, color=YELLOW)
        ).arrange(DOWN, buff=0.2)
        process_text.next_to(arrow, UP, buff=0.3)
        
        self.play(Create(arrow), Write(process_text), run_time=2)
        self.wait(1)
        
        # Show reduced dimension vector
        reduced_vector = Rectangle(width=0.3, height=2, color=SUCCESS_COLOR, fill_opacity=0.7)
        reduced_label = VGroup(
            Text("Reduced Features", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("100 dimensions", font_size=12, color=SUCCESS_COLOR),
            Text("(essential info)", font_size=10, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        reduced_label.next_to(reduced_vector, DOWN, buff=0.3)
        
        reduced_group = VGroup(reduced_vector, reduced_label)
        reduced_group.shift(RIGHT * 3.5)
        
        self.play(Create(reduced_group), run_time=2)
        self.wait(1)
        
        # Show how this solves the challenge
        solution_benefits = VGroup(
            Text("✅ Solves Real-time Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("• Faster computation", font_size=12, color=SUCCESS_COLOR),
            Text("• Less memory usage", font_size=12, color=SUCCESS_COLOR),
            Text("• Reduced overfitting", font_size=12, color=SUCCESS_COLOR),
            Text("• Suitable for production", font_size=12, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        solution_benefits.shift(DOWN * 2)
        
        self.play(Write(solution_benefits), run_time=2)
        
        self.wait(2)
        
        # Clear for next scene - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def gaussian_modeling_implementation(self):
        """Scene 6: Gaussian Distribution Modeling Implementation Detail"""
        self.chapter_transition("Implementation Detail 3: Gaussian Distribution Modeling")
        
        # Connect to challenge
        challenge_connection = Text("Solving: Limited training data (only normal samples)", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        
        # Step 1: Show training data collection
        training_title = Text("Training Phase: Collect Normal Embeddings", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        training_title.next_to(challenge_connection, DOWN, buff=0.5)
        self.play(Write(training_title), run_time=1)
        
        # Show multiple training images
        training_images = VGroup()
        for i in range(4):
            img = Square(side_length=1, color=SECONDARY_COLOR, fill_opacity=0.4)
            label = Text(f"Normal {i+1}", font_size=10, color=SECONDARY_COLOR)
            label.next_to(img, UP, buff=0.1)
            img_group = VGroup(img, label)
            img_group.shift(LEFT * 4 + RIGHT * i * 1.2)
            training_images.add(img_group)
        
        self.play(AnimationGroup(*[Create(img) for img in training_images], lag_ratio=0.2), run_time=2)
        self.wait(1)
        
        # Show the key insight
        key_insight = VGroup(
            Text("💡 Key Insight:", font_size=16, color=PRIMARY_COLOR, weight=BOLD),
            Text("Model what's NORMAL", font_size=14, color=PRIMARY_COLOR),
            Text("instead of learning anomalies", font_size=14, color=PRIMARY_COLOR),
            Text("", font_size=8),  # spacer
            Text("✅ Solves: Limited training data", font_size=12, color=SUCCESS_COLOR, weight=BOLD)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        key_insight.shift(RIGHT * 2.5)
        
        self.play(Write(key_insight), run_time=2)
        self.wait(2)
        
        # Show patch position concept
        arrow_down = Arrow(training_images.get_bottom(), DOWN * 1.5, color=WHITE, stroke_width=2)
        patch_concept = Text("Each patch position (i,j) across all training images", 
                           font_size=14, color=WHITE)
        patch_concept.next_to(arrow_down, DOWN, buff=0.3)
        
        self.play(Create(arrow_down), Write(patch_concept), run_time=2)
        self.wait(1)
        
        # Clear and show Gaussian fitting
        self.play(FadeOut(training_images), FadeOut(arrow_down), FadeOut(patch_concept), FadeOut(key_insight), run_time=1)
        
        # Step 2: Show Gaussian fitting process
        gaussian_title = Text("Per-Patch Gaussian Distribution Fitting", font_size=18, color=PRIMARY_COLOR, weight=BOLD)
        gaussian_title.next_to(training_title, DOWN, buff=0.5)
        self.play(Write(gaussian_title), run_time=1)
        
        # Show data points for one patch position
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=2,
            axis_config={"color": WHITE, "stroke_width": 2}
        )
        axes.shift(LEFT * 3)
        
        # Add sample data points
        np.random.seed(42)
        points = VGroup()
        for _ in range(15):
            x, y = np.random.multivariate_normal([0, 0], [[0.5, 0.2], [0.2, 0.3]])
            if -2.5 < x < 2.5 and -1.5 < y < 1.5:
                point = Dot(axes.coords_to_point(x, y), color=NORMAL_COLOR, radius=0.05)
                points.add(point)
        
        data_label = Text("Training embeddings\\nfor patch position (i,j)", 
                         font_size=12, color=NORMAL_COLOR)
        data_label.next_to(axes, DOWN, buff=0.3)
        
        self.play(Create(axes), run_time=1)
        self.play(AnimationGroup(*[Create(point) for point in points], lag_ratio=0.1), run_time=2)
        self.play(Write(data_label), run_time=1)
        
        # Show fitted Gaussian
        gaussian_ellipse = Ellipse(
            width=2, 
            height=1.5, 
            color=PRIMARY_COLOR, 
            fill_opacity=0.3
        )
        gaussian_ellipse.move_to(axes.get_center())
        
        gaussian_label = VGroup(
            Text("Fitted Multivariate", font_size=12, color=PRIMARY_COLOR, weight=BOLD),
            Text("Gaussian Distribution", font_size=12, color=PRIMARY_COLOR, weight=BOLD),
            Text("μ(i,j), Σ(i,j)", font_size=10, color=PRIMARY_COLOR)
        ).arrange(DOWN, buff=0.1)
        gaussian_label.shift(RIGHT * 2.5)
        
        self.play(Create(gaussian_ellipse), Write(gaussian_label), run_time=2)
        
        # Show how this solves the challenge
        solution_note = VGroup(
            Text("✅ Solves Limited Data Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("• Only needs normal samples", font_size=12, color=SUCCESS_COLOR),
            Text("• No anomaly examples required", font_size=12, color=SUCCESS_COLOR),
            Text("• Learns normal distribution", font_size=12, color=SUCCESS_COLOR),
            Text("• Detects deviations from normal", font_size=12, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        solution_note.shift(DOWN * 2)
        
        self.play(Write(solution_note), run_time=2)
        self.wait(2)
        
        # Clear and show the complete concept
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        
        # Step 3: Show the complete Gaussian map
        complete_title = Text("Complete Gaussian Parameter Map", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        complete_title.to_edge(UP, buff=0.5)
        self.play(Write(complete_title), run_time=1)
        
        # Create grid of Gaussians
        gaussian_grid = VGroup()
        for i in range(4):
            for j in range(4):
                # Small Gaussian representation
                gauss_circle = Circle(radius=0.15, color=PRIMARY_COLOR, fill_opacity=0.6)
                gauss_circle.shift(LEFT * 2 + RIGHT * j * 0.5 + UP * 1 + DOWN * i * 0.5)
                gaussian_grid.add(gauss_circle)
        
        grid_label = VGroup(
            Text("Each position has its own", font_size=14, color=WHITE),
            Text("Gaussian distribution", font_size=14, color=WHITE),
            Text("Parameters: μ(i,j) and Σ(i,j)", font_size=12, color=PRIMARY_COLOR)
        ).arrange(DOWN, buff=0.2)
        grid_label.shift(RIGHT * 2.5)
        
        self.play(AnimationGroup(*[Create(gauss) for gauss in gaussian_grid], lag_ratio=0.05), run_time=2)
        self.play(Write(grid_label), run_time=2)
        
        self.wait(2)
        
        # Clear for next scene - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def mahalanobis_implementation(self):
        """Scene 7: Mahalanobis Distance Implementation Detail"""
        self.chapter_transition("Implementation Detail 4: Mahalanobis Distance Computation")
        
        # Connect to challenge
        challenge_connection = Text("Solving: Need for precise pixel-level localization", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        
        # Show the formula and concept
        title = Text("Anomaly Scoring with Mahalanobis Distance", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        title.next_to(challenge_connection, DOWN, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Mathematical formula (simplified for animation)
        formula_text = VGroup(
            Text("M(x) = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]", font_size=18, color=ACCENT_COLOR),
            Text("x: test embedding, μ: mean, Σ: covariance", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.3)
        
        self.play(Write(formula_text), run_time=2)
        self.wait(1)
        
        # Show how this enables precise localization
        localization_explanation = VGroup(
            Text("🎯 Enables Precise Localization:", font_size=14, color=PRIMARY_COLOR, weight=BOLD),
            Text("• Each patch gets its own score", font_size=12, color=WHITE),
            Text("• Pixel-level anomaly detection", font_size=12, color=WHITE),
            Text("• Exact defect location identification", font_size=12, color=WHITE),
            Text("• No global image-level decisions", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.1)
        localization_explanation.shift(LEFT * 2.5)
        
        self.play(Write(localization_explanation), run_time=2)
        self.wait(1)
        
        # Visual representation
        # Normal case
        normal_demo = VGroup()
        normal_axes = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=2, y_length=2)
        normal_gaussian = Circle(radius=0.5, color=NORMAL_COLOR, fill_opacity=0.3)
        normal_point = Dot(normal_axes.coords_to_point(0.2, 0.1), color=NORMAL_COLOR, radius=0.08)
        normal_label = Text("Normal\\n(Low Distance)", font_size=12, color=NORMAL_COLOR)
        normal_label.next_to(normal_axes, DOWN, buff=0.3)
        
        normal_demo.add(normal_axes, normal_gaussian, normal_point, normal_label)
        normal_demo.shift(LEFT * 3 + DOWN * 1)
        
        # Anomaly case  
        anomaly_demo = VGroup()
        anomaly_axes = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=2, y_length=2)
        anomaly_gaussian = Circle(radius=0.5, color=ANOMALY_COLOR, fill_opacity=0.3)
        anomaly_point = Dot(anomaly_axes.coords_to_point(1.5, 1.2), color=ANOMALY_COLOR, radius=0.08)
        anomaly_label = Text("Anomaly\\n(High Distance)", font_size=12, color=ANOMALY_COLOR)
        anomaly_label.next_to(anomaly_axes, DOWN, buff=0.3)
        
        anomaly_demo.add(anomaly_axes, anomaly_gaussian, anomaly_point, anomaly_label)
        anomaly_demo.shift(RIGHT * 3 + DOWN * 1)
        
        self.play(Create(normal_demo), Create(anomaly_demo), run_time=3)
        
        # Add distance lines
        normal_line = Line(normal_gaussian.get_center(), normal_point.get_center(), 
                          color=NORMAL_COLOR, stroke_width=3)
        anomaly_line = Line(anomaly_gaussian.get_center(), anomaly_point.get_center(), 
                           color=ANOMALY_COLOR, stroke_width=3)
        
        self.play(Create(normal_line), Create(anomaly_line), run_time=2)
        
        # Show how this solves the localization challenge
        solution_note = VGroup(
            Text("✅ Solves Localization Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("• Precise pixel-level detection", font_size=12, color=SUCCESS_COLOR),
            Text("• Exact defect boundaries", font_size=12, color=SUCCESS_COLOR),
            Text("• Quantitative anomaly scores", font_size=12, color=SUCCESS_COLOR),
            Text("• Interpretable results", font_size=12, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        solution_note.shift(DOWN * 2)
        
        self.play(Write(solution_note), run_time=2)
        self.wait(2)
        
        # Clear for next scene - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def complete_pipeline(self):
        """Scene 8: Complete training pipeline"""
        self.chapter_transition("Complete Training Pipeline")
        
        # Connect to all challenges
        challenge_connection = Text("Solving: All industrial challenges with one unified approach", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        
        # Create pipeline steps
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
            step_text = Text(step, font_size=12, color=color, weight=BOLD)
            step_text.move_to(step_box.get_center())
            
            step_group = VGroup(step_box, step_text)
            step_group.shift(UP * 2 + DOWN * i * 1)
            pipeline.add(step_group)
        
        # Add arrows between steps
        arrows = VGroup()
        for i in range(len(steps) - 1):
            arrow = Arrow(pipeline[i].get_bottom(), pipeline[i+1].get_top(), 
                         color=WHITE, stroke_width=2)
            arrows.add(arrow)
        
        self.play(AnimationGroup(*[Create(step) for step in pipeline], lag_ratio=0.3), run_time=3)
        self.play(AnimationGroup(*[Create(arrow) for arrow in arrows], lag_ratio=0.2), run_time=2)
        
        # Add timing information
        timing_info = VGroup(
            Text("Training Time:", font_size=14, color=WHITE, weight=BOLD),
            Text("• Fast: No gradient computation", font_size=12, color=SUCCESS_COLOR),
            Text("• Only forward passes needed", font_size=12, color=SUCCESS_COLOR),
            Text("• Gaussian fitting is efficient", font_size=12, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        timing_info.shift(RIGHT * 3)
        
        self.play(Write(timing_info), run_time=2)
        
        # Show how this solves all challenges
        all_solutions = VGroup(
            Text("✅ Solves ALL Challenges:", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("• Limited data → Multi-layer features + Gaussian modeling", font_size=12, color=SUCCESS_COLOR),
            Text("• Precise localization → Per-patch Mahalanobis scoring", font_size=12, color=SUCCESS_COLOR),
            Text("• Real-time processing → Dimensionality reduction", font_size=12, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        all_solutions.shift(DOWN * 2)
        
        self.play(Write(all_solutions), run_time=2)
        self.wait(3)
        
        # Clear for next scene - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def inference_demo(self):
        """Scene 9: Inference process demo"""
        self.chapter_transition("Inference Process")
        
        # Connect to challenge
        challenge_connection = Text("Solving: Real-time production line deployment", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        challenge_connection.to_edge(UP, buff=0.5)
        self.play(Write(challenge_connection), run_time=1)
        
        # Show test image processing
        test_img = Square(side_length=2, color=ACCENT_COLOR, fill_opacity=0.4)
        test_label = Text("Test Image", font_size=16, color=ACCENT_COLOR, weight=BOLD)
        test_label.next_to(test_img, UP, buff=0.3)
        test_group = VGroup(test_img, test_label)
        test_group.shift(LEFT * 4)
        
        self.play(Create(test_group), run_time=1)
        
        # Processing steps
        arrow1 = Arrow(test_group.get_right(), RIGHT * 1.5, color=WHITE, stroke_width=2)
        processing = Text("Same feature\\nextraction process", font_size=12, color=WHITE)
        processing.next_to(arrow1, UP, buff=0.2)
        
        # Features
        features = Rectangle(width=0.4, height=2, color=PRIMARY_COLOR, fill_opacity=0.6)
        features.shift(RIGHT * 0.5)
        
        self.play(Create(arrow1), Write(processing), Create(features), run_time=2)
        
        # Distance computation
        arrow2 = Arrow(features.get_right(), RIGHT * 2, color=WHITE, stroke_width=2)
        distance_comp = Text("Mahalanobis\\ndistance", font_size=12, color=WHITE)
        distance_comp.next_to(arrow2, UP, buff=0.2)
        
        # Anomaly map
        anomaly_map = VGroup()
        for i in range(6):
            for j in range(6):
                # Simulate anomaly scores with some random pattern
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
        
        map_label = Text("Anomaly Map", font_size=14, color=ACCENT_COLOR, weight=BOLD)
        map_label.next_to(anomaly_map, DOWN, buff=0.3)
        
        self.play(Create(arrow2), Write(distance_comp), run_time=1)
        self.play(AnimationGroup(*[Create(pixel) for pixel in anomaly_map], lag_ratio=0.01), run_time=2)
        self.play(Write(map_label), run_time=1)
        
        # Show how this solves the real-time challenge
        real_time_solution = VGroup(
            Text("✅ Solves Real-time Challenge:", font_size=14, color=SUCCESS_COLOR, weight=BOLD),
            Text("• Fast inference for production", font_size=12, color=SUCCESS_COLOR),
            Text("• Immediate defect detection", font_size=12, color=SUCCESS_COLOR),
            Text("• Precise localization output", font_size=12, color=SUCCESS_COLOR),
            Text("• Ready for deployment", font_size=12, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.1)
        real_time_solution.shift(DOWN * 2)
        
        self.play(Write(real_time_solution), run_time=2)
        self.wait(2)
        
        # Clear for finale - Stage Director Rule
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def finale(self):
        """Final scene with key takeaways"""
        self.chapter_transition("Key Takeaways")
        
        # Final message connecting back to challenges
        final_message = VGroup(
            Text("PaDiM: Complete Industrial Solution", font_size=32, color=PRIMARY_COLOR, weight=BOLD),
            Text("", font_size=8),  # spacer
            Text("🎯 Challenge → Solution:", font_size=18, color=ACCENT_COLOR, weight=BOLD),
            Text("• Limited data → Multi-layer features + Gaussian modeling", font_size=14, color=SECONDARY_COLOR),
            Text("• Precise localization → Per-patch Mahalanobis scoring", font_size=14, color=SECONDARY_COLOR),
            Text("• Real-time processing → Dimensionality reduction", font_size=14, color=SECONDARY_COLOR),
            Text("", font_size=8),  # spacer
            Text("✨ Implementation Details:", font_size=16, color=PRIMARY_COLOR, weight=BOLD),
            Text("• Layer concatenation: [layer1, layer2, layer3]", font_size=12, color=WHITE),
            Text("• Random selection: 100-550 features", font_size=12, color=WHITE),
            Text("• Gaussian fitting: μ(i,j), Σ(i,j) per patch", font_size=12, color=WHITE),
            Text("• Distance scoring: M(x) = √[(x-μ)ᵀΣ⁻¹(x-μ)]", font_size=12, color=WHITE),
            Text("", font_size=8),  # spacer
            Text("🚀 Result: Production-ready anomaly detection!", font_size=18, color=SUCCESS_COLOR, weight=BOLD)
        ).arrange(DOWN, buff=0.3)
        
        self.play(Write(final_message, lag_ratio=0.2), run_time=4)
        self.wait(3) 
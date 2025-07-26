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

class PaDiMStory(Scene):
    """
    The Epic Tale of PaDiM: Patch Distribution Modeling for Anomaly Detection
    
    What it shows: Complete methodology of PaDiM from patch extraction to anomaly scoring
    Duration: ~3-4 minutes
    Key concepts: Patch extraction, feature embedding, Gaussian modeling, Mahalanobis distance
    Cleanup: Proper scene transitions with self.clear()
    """
    
    def construct(self):
        # 🎬 Act I: The Grand Introduction
        self.prologue()
        
        # 🎭 Act II: The Methodology Unfolds
        self.show_patch_extraction()
        self.show_feature_embedding() 
        self.show_gaussian_modeling()
        self.show_anomaly_detection()
        
        # 🎪 Act III: The Grand Finale
        self.show_complete_pipeline()
        self.epilogue()
    
    def prologue(self):
        """
        Opening scene: Introduce PaDiM with cinematic flair
        """
        # Title sequence with dramatic reveal
        title = Text("PaDiM", font_size=72, color=PRIMARY_COLOR, weight=BOLD)
        subtitle = Text("Patch Distribution Modeling", font_size=36, color=SECONDARY_COLOR)
        tagline = Text("for Anomaly Detection", font_size=24, color=WHITE)
        
        # Arrange title elements
        title.to_edge(UP, buff=1)
        subtitle.next_to(title, DOWN, buff=0.3)
        tagline.next_to(subtitle, DOWN, buff=0.2)
        
        # Dramatic entrance
        self.play(
            Write(title, run_time=2),
            lag_ratio=0.3
        )
        self.play(
            FadeIn(subtitle, shift=UP*0.5, run_time=1.5),
            FadeIn(tagline, shift=UP*0.3, run_time=1.5),
            lag_ratio=0.2
        )
        self.wait(2)
        
        # The challenge text
        challenge = VGroup(
            Text("The Challenge:", font_size=32, color=ACCENT_COLOR, weight=BOLD),
            Text("Detect anomalies without knowing what they look like", font_size=24),
            Text("Using only normal samples for training", font_size=24, color=SUCCESS_COLOR)
        ).arrange(DOWN, buff=0.4)
        challenge.next_to(tagline, DOWN, buff=1)
        
        self.play(Write(challenge[0], run_time=1))
        self.play(
            Write(challenge[1], run_time=1.5),
            Write(challenge[2], run_time=1.5),
            lag_ratio=0.3
        )
        self.wait(2)
        
        # Chapter transition
        self.chapter_transition("Chapter 1: Patch Extraction")
    
    def show_patch_extraction(self):
        """
        Scene 1: Demonstrate patch extraction process
        """
        # Input image representation
        image_rect = Rectangle(height=3, width=4, color=PRIMARY_COLOR, fill_opacity=0.3)
        image_label = Text("Input Image", font_size=24, color=PRIMARY_COLOR)
        image_label.next_to(image_rect, UP, buff=0.3)
        
        image_group = VGroup(image_rect, image_label)
        image_group.to_edge(LEFT, buff=1)
        
        self.play(Create(image_rect), Write(image_label), run_time=1.5)
        
        # Show grid overlay for patches
        grid_lines = VGroup()
        patch_size = 0.6
        for i in range(6):  # 5x7 grid approximately
            h_line = Line(
                image_rect.get_left() + UP * (1.5 - i * patch_size),
                image_rect.get_right() + UP * (1.5 - i * patch_size),
                color=YELLOW, stroke_width=2
            )
            grid_lines.add(h_line)
        
        for j in range(8):
            v_line = Line(
                image_rect.get_bottom() + RIGHT * (-2 + j * patch_size),
                image_rect.get_top() + RIGHT * (-2 + j * patch_size),
                color=YELLOW, stroke_width=2
            )
            grid_lines.add(v_line)
        
        self.play(Create(grid_lines, lag_ratio=0.1, run_time=2))
        
        # Highlight individual patches
        patch_coords = [(i, j) for i in range(5) for j in range(7)]
        sample_patches = []
        
        for idx, (i, j) in enumerate(patch_coords[:6]):  # Show first 6 patches
            patch = Rectangle(
                height=patch_size-0.05, width=patch_size-0.05,
                color=ACCENT_COLOR, fill_opacity=0.7
            )
            patch.move_to(
                image_rect.get_center() + 
                RIGHT * (-1.8 + j * patch_size) + 
                UP * (1.2 - i * patch_size)
            )
            sample_patches.append(patch)
        
        # Progressive patch highlighting
        self.play(
            *[Create(patch) for patch in sample_patches[:3]],
            lag_ratio=0.2, run_time=1.5
        )
        self.play(
            *[Create(patch) for patch in sample_patches[3:]],
            lag_ratio=0.2, run_time=1.5
        )
        
        # Extract patches to the right
        explanation = VGroup(
            Text("Patch Extraction:", font_size=28, color=SECONDARY_COLOR, weight=BOLD),
            Text("• Divide image into overlapping patches", font_size=20),
            Text("• Each patch captures local features", font_size=20),
            Text("• Provides spatial context", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        explanation.to_edge(RIGHT, buff=1)
        
        self.play(Write(explanation, lag_ratio=0.3, run_time=2))
        self.wait(2)
        
        # Fade out current scene elements
        self.play(
            *[FadeOut(mob) for mob in [image_group, grid_lines, *sample_patches, explanation]],
            run_time=1.5
        )
        self.clear()
    
    def show_feature_embedding(self):
        """
        Scene 2: Feature embedding through pre-trained CNNs with detailed explanation
        """
        scene_title = Text("Feature Embedding Process", font_size=32, color=PRIMARY_COLOR, weight=BOLD)
        scene_title.to_edge(UP, buff=0.5)
        self.play(Write(scene_title), run_time=1)
        self.wait(1)
        
        # Step 1: Show single patch first
        self.chapter_transition("Step 1: Single Patch Processing")
        
        # Single patch visualization
        patch = Square(side_length=1.5, color=SECONDARY_COLOR, fill_opacity=0.5)
        patch_label = Text("Single Patch\n(e.g., 3x3 pixels)", font_size=18, color=SECONDARY_COLOR)
        patch_label.next_to(patch, UP, buff=0.3)
        
        patch_group = VGroup(patch, patch_label)
        patch_group.shift(LEFT * 4)
        
        self.play(Create(patch), Write(patch_label), run_time=1.5)
        self.wait(1)
        
        # Step 2: Multi-layer CNN explanation
        cnn_explanation = VGroup(
            Text("Pre-trained CNN (ResNet/WideResNet)", font_size=20, color=PRIMARY_COLOR, weight=BOLD),
            Text("Extracts features from multiple layers:", font_size=16),
            Text("• Early layers: Low-level features (edges, textures)", font_size=14, color=BLUE),
            Text("• Middle layers: Mid-level features (shapes, patterns)", font_size=14, color=GREEN), 
            Text("• Later layers: High-level features (objects, semantics)", font_size=14, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        cnn_explanation.to_edge(RIGHT, buff=1)
        
        self.play(Write(cnn_explanation, lag_ratio=0.3, run_time=3))
        self.wait(2)
        
        # Clear for next step
        self.play(FadeOut(patch_group), FadeOut(cnn_explanation), run_time=1)
        self.clear()
        
        # Step 3: Layer selection strategy
        self.chapter_transition("Step 2: Strategic Layer Selection")
        
        # CNN layers visualization
        layers = VGroup()
        layer_info = [
            ("Layer 1", BLUE, "64 channels", "Edge detection"),
            ("Layer 10", GREEN, "256 channels", "Pattern recognition"), 
            ("Layer 17", YELLOW, "512 channels", "Object parts")
        ]
        
        for i, (name, color, channels, purpose) in enumerate(layer_info):
            layer_rect = Rectangle(height=1.5, width=1.2, color=color, fill_opacity=0.6)
            layer_name = Text(name, font_size=14, color=WHITE, weight=BOLD)
            layer_name.move_to(layer_rect.get_center())
            
            layer_details = VGroup(
                Text(channels, font_size=12, color=color),
                Text(purpose, font_size=10, color=WHITE)
            ).arrange(DOWN, buff=0.1)
            layer_details.next_to(layer_rect, DOWN, buff=0.2)
            
            layer_group = VGroup(layer_rect, layer_name, layer_details)
            layer_group.shift(LEFT * 3 + RIGHT * i * 2.5)
            layers.add(layer_group)
        
        self.play(Create(layers, lag_ratio=0.4, run_time=2))
        
        # Selection explanation
        selection_text = VGroup(
            Text("Why Multiple Layers?", font_size=20, color=ACCENT_COLOR, weight=BOLD),
            Text("• Different layers capture different levels of abstraction", font_size=14),
            Text("• Combines low-level texture with high-level semantics", font_size=14),
            Text("• Creates rich, multi-scale feature representation", font_size=14, color=SUCCESS_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        selection_text.to_edge(DOWN, buff=0.5)
        
        self.play(Write(selection_text, lag_ratio=0.3, run_time=2))
        self.wait(2)
        
        # Clear for next step
        self.play(FadeOut(layers), FadeOut(selection_text), run_time=1)
        self.clear()
        
        # Step 4: Feature concatenation and dimensionality
        self.chapter_transition("Step 3: Feature Concatenation")
        
        # Show feature vectors from different layers
        feature_vectors = VGroup()
        vector_sizes = [64, 256, 512]
        vector_colors = [BLUE, GREEN, YELLOW]
        
        for i, (size, color) in enumerate(zip(vector_sizes, vector_colors)):
            vector = VGroup()
            for j in range(min(8, size//32)):  # Visual representation
                feat_rect = Rectangle(height=0.3, width=0.4, color=color, fill_opacity=0.8)
                feat_rect.shift(UP * (3 - i * 2) + RIGHT * j * 0.5)
                vector.add(feat_rect)
            
            size_label = Text(f"{size}D", font_size=16, color=color, weight=BOLD)
            size_label.next_to(vector, LEFT, buff=0.5)
            
            feature_vectors.add(VGroup(vector, size_label))
        
        feature_vectors.shift(LEFT * 3)
        self.play(Create(feature_vectors, lag_ratio=0.3, run_time=2))
        
        # Concatenation arrow and result
        concat_arrow = Arrow(LEFT * 1, RIGHT * 1, color=WHITE, stroke_width=4)
        concat_label = Text("Concatenate", font_size=16, color=WHITE)
        concat_label.next_to(concat_arrow, UP, buff=0.2)
        
        # Final concatenated vector
        final_vector = VGroup()
        total_size = sum(vector_sizes)
        for i in range(12):  # Visual representation of 832D vector
            feat_rect = Rectangle(height=0.3, width=0.4, color=PRIMARY_COLOR, fill_opacity=0.8)
            feat_rect.shift(UP * 1 + RIGHT * (2.5 + i * 0.5))
            final_vector.add(feat_rect)
        
        final_label = Text(f"{total_size}D Vector", font_size=18, color=PRIMARY_COLOR, weight=BOLD)
        final_label.next_to(final_vector, UP, buff=0.3)
        
        self.play(
            Create(concat_arrow),
            Write(concat_label),
            run_time=1
        )
        self.play(
            Create(final_vector),
            Write(final_label),
            run_time=1.5
        )
        self.wait(2)
        
        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def show_gaussian_modeling(self):
        """
        Scene 3: Detailed Gaussian distribution modeling process
        """
        scene_title = Text("Gaussian Distribution Modeling", font_size=32, color=PRIMARY_COLOR, weight=BOLD)
        scene_title.to_edge(UP, buff=0.5)
        self.play(Write(scene_title), run_time=1)
        self.wait(1)
        
        # Clear and start with collection phase
        self.chapter_transition("Step 1: Feature Collection from Normal Images")
        
        # Show multiple normal training images
        training_images = VGroup()
        for i in range(6):
            img_rect = Rectangle(height=0.8, width=0.8, color=SUCCESS_COLOR, fill_opacity=0.3)
            img_label = Text(f"Normal {i+1}", font_size=10, color=SUCCESS_COLOR)
            img_label.next_to(img_rect, DOWN, buff=0.1)
            
            img_group = VGroup(img_rect, img_label)
            img_group.shift(LEFT * 4 + UP * (1 - i%3) + RIGHT * (i//3) * 1.2)
            training_images.add(img_group)
        
        self.play(Create(training_images, lag_ratio=0.2, run_time=2))
        
        # Arrow pointing to feature collection
        arrow1 = Arrow(LEFT * 2, ORIGIN, color=WHITE, stroke_width=3)
        extraction_text = Text("Extract features from\neach patch position", font_size=14, color=WHITE)
        extraction_text.next_to(arrow1, UP, buff=0.2)
        
        self.play(Create(arrow1), Write(extraction_text), run_time=1)
        
        # Feature collection visualization  
        feature_collection = VGroup()
        for i in range(4):
            for j in range(3):
                feature_point = Dot(radius=0.06, color=NORMAL_COLOR)
                # Create slight variations to show distribution
                x_offset = np.random.normal(0, 0.3)
                y_offset = np.random.normal(0, 0.2)
                feature_point.shift(RIGHT * 2 + UP * (1 - i*0.5) + RIGHT * j*0.4 + RIGHT * x_offset + UP * y_offset)
                feature_collection.add(feature_point)
        
        self.play(Create(feature_collection, lag_ratio=0.05, run_time=1.5))
        self.wait(1)
        
        # Clear for distribution fitting
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        
        # Step 2: Distribution fitting
        self.chapter_transition("Step 2: Fitting Gaussian Distribution")
        
        # Show feature cloud again but more organized
        np.random.seed(42)  # Reproducible randomness
        feature_cloud = VGroup()
        
        for i in range(25):
            point = Dot(radius=0.08, color=NORMAL_COLOR)
            x = np.random.normal(0, 1) * 1.2
            y = np.random.normal(0, 0.8) * 1.0
            point.move_to([x, y, 0])
            feature_cloud.add(point)
        
        feature_cloud.shift(LEFT * 2.5)
        
        # Add coordinate system
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE, "stroke_width": 2}
        )
        axes.shift(LEFT * 2.5)
        
        self.play(Create(axes), run_time=1)
        self.play(Create(feature_cloud, lag_ratio=0.05, run_time=2))
        
        # Statistical calculations explanation with proper LaTeX
        stats_explanation = VGroup(
            Text("Statistical Parameter Estimation:", font_size=18, color=PRIMARY_COLOR, weight=BOLD),
            Text("", font_size=8),  # Spacer
            Text("1. Mean (μ): Center of the distribution", font_size=14, color=SECONDARY_COLOR),
            MathTex(r"\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i", font_size=24, color=SECONDARY_COLOR),
            Text("", font_size=6),  # Spacer
            Text("2. Covariance (Σ): Shape and orientation", font_size=14, color=SECONDARY_COLOR),
            MathTex(r"\boldsymbol{\Sigma} = \frac{1}{N-1} \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T", 
                   font_size=20, color=SECONDARY_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        stats_explanation.to_edge(RIGHT, buff=0.5)
        
        self.play(Write(stats_explanation, lag_ratio=0.2, run_time=4))
        self.wait(2)
        
        # Show fitted distribution
        distribution_ellipse = Ellipse(
            width=2.8, height=1.8, color=PRIMARY_COLOR, 
            fill_opacity=0.15, stroke_width=3
        )
        distribution_ellipse.shift(LEFT * 2.5)
        
        # Mean point
        mean_dot = Dot([0, 0, 0], color=ACCENT_COLOR, radius=0.12)
        mean_dot.shift(LEFT * 2.5)
        mean_label = MathTex(r"\boldsymbol{\mu}", font_size=24, color=ACCENT_COLOR)
        mean_label.next_to(mean_dot, UP, buff=0.2)
        
        self.play(
            Create(distribution_ellipse),
            Create(mean_dot),
            Write(mean_label),
            run_time=2
        )
        self.wait(2)
        
        # Clear for position-specific explanation
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        
        # Step 3: Position-specific modeling
        self.chapter_transition("Step 3: Position-Specific Modeling")
        
        # Grid of positions
        position_grid = VGroup()
        for i in range(3):
            for j in range(4):
                pos_rect = Rectangle(height=0.6, width=0.6, color=PRIMARY_COLOR, fill_opacity=0.2)
                pos_rect.shift(LEFT * 2 + UP * (1 - i*0.8) + RIGHT * j*0.8)
                
                # Small distribution ellipse for each position
                mini_ellipse = Ellipse(width=0.4, height=0.25, color=SUCCESS_COLOR, fill_opacity=0.3)
                mini_ellipse.move_to(pos_rect.get_center())
                
                position_grid.add(VGroup(pos_rect, mini_ellipse))
        
        self.play(Create(position_grid, lag_ratio=0.1, run_time=2))
        
        # Explanation
        position_explanation = VGroup(
            Text("Key Insight: Position Matters!", font_size=20, color=ACCENT_COLOR, weight=BOLD),
            Text("", font_size=8),  # Spacer
            Text("• Each spatial position has its own distribution", font_size=16),
            Text("• Different positions have different normal patterns", font_size=16),
            Text("• This captures spatial context effectively", font_size=16, color=SUCCESS_COLOR),
            Text("", font_size=8),  # Spacer
            Text("Total: H×W position-specific Gaussians", font_size=16, color=PRIMARY_COLOR, weight=BOLD)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        position_explanation.to_edge(RIGHT, buff=0.5)
        
        self.play(Write(position_explanation, lag_ratio=0.2, run_time=3))
        self.wait(3)
        
        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def show_anomaly_detection(self):
        """
        Scene 4: Anomaly detection using Mahalanobis distance
        """
        scene_title = Text("Anomaly Detection", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        scene_title.to_edge(UP, buff=0.5)
        self.play(Write(scene_title), run_time=1)
        
        test_subtitle = Text("Test Phase: Detecting Anomalies", font_size=24, color=ACCENT_COLOR)
        test_subtitle.next_to(scene_title, DOWN, buff=0.5)
        self.play(Write(test_subtitle), run_time=1)
        
        # Show the learned distribution from previous scene
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": WHITE, "stroke_width": 2}
        )
        axes.shift(LEFT * 3)
        
        # Normal distribution ellipse
        normal_ellipse = Ellipse(
            width=2.5, height=1.8, color=SUCCESS_COLOR, 
            fill_opacity=0.3, stroke_width=3
        )
        normal_ellipse.shift(LEFT * 3)
        
        normal_label = Text("Learned Normal\nDistribution", font_size=16, color=SUCCESS_COLOR)
        normal_label.next_to(normal_ellipse, UP, buff=0.3)
        
        self.play(
            Create(axes),
            Create(normal_ellipse),
            Write(normal_label),
            run_time=1.5
        )
        
        # Test points - normal and anomalous
        normal_point = Dot([0.5, 0.3, 0], color=NORMAL_COLOR, radius=0.12)
        normal_point.shift(LEFT * 3)
        
        anomaly_point = Dot([2.2, 1.5, 0], color=ANOMALY_COLOR, radius=0.12)
        anomaly_point.shift(LEFT * 3)
        
        self.play(
            Create(normal_point),
            Create(anomaly_point),
            run_time=1
        )
        
        # Mahalanobis distance formula
        formula = VGroup(
            Text("Mahalanobis Distance:", font_size=24, color=PRIMARY_COLOR, weight=BOLD),
            MathTex(r"M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}", 
                   font_size=32, color=SECONDARY_COLOR)
        ).arrange(DOWN, buff=0.5)
        formula.to_edge(RIGHT, buff=0.5)
        
        self.play(Write(formula, lag_ratio=0.3, run_time=2))
        
        # Show distance vectors
        center = axes.get_center()
        
        normal_arrow = Arrow(
            center, normal_point.get_center(),
            color=NORMAL_COLOR, stroke_width=4
        )
        normal_distance = Text("Low Distance\n(Normal)", font_size=14, color=NORMAL_COLOR)
        normal_distance.next_to(normal_point, DOWN, buff=0.2)
        
        anomaly_arrow = Arrow(
            center, anomaly_point.get_center(),
            color=ANOMALY_COLOR, stroke_width=4
        )
        anomaly_distance = Text("High Distance\n(Anomaly!)", font_size=14, color=ANOMALY_COLOR)
        anomaly_distance.next_to(anomaly_point, UP, buff=0.2)
        
        self.play(
            Create(normal_arrow),
            Write(normal_distance),
            run_time=1
        )
        self.play(
            Create(anomaly_arrow), 
            Write(anomaly_distance),
            run_time=1
        )
        
        # Threshold visualization
        threshold_circle = Circle(radius=1.8, color=YELLOW, stroke_width=3, stroke_opacity=0.8)
        threshold_circle.shift(LEFT * 3)
        threshold_label = Text("Threshold", font_size=16, color=YELLOW)
        threshold_label.next_to(threshold_circle, RIGHT, buff=0.3)
        
        self.play(
            Create(threshold_circle),
            Write(threshold_label),
            run_time=1.5
        )
        
        # Decision rule
        decision = VGroup(
            Text("Decision Rule:", font_size=20, color=ACCENT_COLOR, weight=BOLD),
            Text("if M(x) > threshold: ANOMALY", font_size=16, color=ANOMALY_COLOR),
            Text("else: NORMAL", font_size=16, color=NORMAL_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        decision.to_edge(DOWN, buff=0.5)
        
        self.play(Write(decision, lag_ratio=0.3, run_time=2))
        self.wait(2)
        
        # Cleanup
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def show_complete_pipeline(self):
        """
        Scene 5: Complete PaDiM pipeline with NO overlapping animations
        """
        pipeline_title = Text("Complete PaDiM Pipeline", font_size=32, color=PRIMARY_COLOR, weight=BOLD)
        pipeline_title.to_edge(UP, buff=0.3)
        self.play(Write(pipeline_title), run_time=1)
        self.wait(1)
        
        # Clear and show Phase 1: Training
        self.chapter_transition("Phase 1: Training on Normal Data")
        
        # Training pipeline steps
        train_steps = VGroup()
        train_step_texts = [
            "Normal\nImages",
            "Patch\nExtraction", 
            "Multi-layer\nCNN Features",
            "Gaussian\nModeling"
        ]
        train_colors = [SUCCESS_COLOR, YELLOW, GREEN, PURPLE]
        
        for i, (text, color) in enumerate(zip(train_step_texts, train_colors)):
            step_box = Rectangle(height=1.2, width=1.8, color=color, fill_opacity=0.3)
            step_label = Text(text, font_size=14, color=color, weight=BOLD)
            step_label.move_to(step_box.get_center())
            
            step = VGroup(step_box, step_label)
            step.shift(LEFT * 4 + RIGHT * i * 2.2)
            train_steps.add(step)
        
        # Create training pipeline flow
        self.play(Create(train_steps[0]), run_time=0.8)
        
        for i in range(1, len(train_steps)):
            arrow = Arrow(
                train_steps[i-1].get_right(),
                train_steps[i].get_left(),
                color=WHITE, stroke_width=3
            )
            
            self.play(
                Create(arrow),
                Create(train_steps[i]),
                run_time=0.8
            )
        
        self.wait(2)
        
        # Clear and show Phase 2: Testing
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        
        self.chapter_transition("Phase 2: Testing for Anomalies")
        
        # Testing pipeline steps
        test_steps = VGroup()
        test_step_texts = [
            "Test\nImage",
            "Patch\nExtraction", 
            "Multi-layer\nCNN Features",
            "Anomaly\nScoring"
        ]
        test_colors = [BLUE, YELLOW, GREEN, RED]
        
        for i, (text, color) in enumerate(zip(test_step_texts, test_colors)):
            step_box = Rectangle(height=1.2, width=1.8, color=color, fill_opacity=0.3)
            step_label = Text(text, font_size=14, color=color, weight=BOLD)
            step_label.move_to(step_box.get_center())
            
            step = VGroup(step_box, step_label)
            step.shift(LEFT * 4 + RIGHT * i * 2.2)
            test_steps.add(step)
        
        # Create testing pipeline flow
        self.play(Create(test_steps[0]), run_time=0.8)
        
        for i in range(1, len(test_steps)):
            arrow = Arrow(
                test_steps[i-1].get_right(),
                test_steps[i].get_left(),
                color=WHITE, stroke_width=3
            )
            
            self.play(
                Create(arrow),
                Create(test_steps[i]),
                run_time=0.8
            )
        
        self.wait(2)
        
        # Clear and show key advantages
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        
        self.chapter_transition("Why PaDiM Works So Well")
        
        # Key advantages with clear spacing
        advantages = VGroup(
            Text("🎯 Key Advantages of PaDiM:", font_size=24, color=SUCCESS_COLOR, weight=BOLD),
            Text("", font_size=12),  # Spacer
            Text("✓ No anomaly samples needed for training", font_size=18, color=WHITE),
            Text("✓ Position-aware feature modeling", font_size=18, color=WHITE),
            Text("✓ Multi-scale CNN feature integration", font_size=18, color=WHITE),
            Text("✓ Efficient inference with pre-trained networks", font_size=18, color=WHITE),
            Text("✓ State-of-the-art performance on MVTec AD", font_size=18, color=WHITE),
            Text("", font_size=12),  # Spacer
            Text("Result: Robust, practical anomaly detection!", font_size=20, color=PRIMARY_COLOR, weight=BOLD)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(Write(advantages, lag_ratio=0.3, run_time=4))
        self.wait(3)
        
        # Cleanup
        self.play(FadeOut(advantages), run_time=1.5)
        self.clear()
    
    def epilogue(self):
        """
        Closing scene with key takeaways
        """
        # Scene 1: Final title card
        final_title = VGroup(
            Text("PaDiM", font_size=48, color=PRIMARY_COLOR, weight=BOLD),
            Text("Elegant. Effective. Efficient.", font_size=24, color=SECONDARY_COLOR),
            Text("The Art of Anomaly Detection", font_size=20, color=WHITE)
        ).arrange(DOWN, buff=0.4)
        
        self.play(Write(final_title, lag_ratio=0.4, run_time=3))
        self.wait(2)
        
        # Clear the stage before showing summary (The Stage Director rule)
        self.play(FadeOut(final_title), run_time=1.5)
        self.clear()
        
        # Scene 2: Key insight summary
        summary = VGroup(
            Text("🔑 Core Innovation:", font_size=20, color=ACCENT_COLOR, weight=BOLD),
            Text("Position-specific Gaussian modeling", font_size=18, color=PRIMARY_COLOR),
            Text("of patch-level feature distributions", font_size=18, color=PRIMARY_COLOR),
            Text("", font_size=10),  # Spacer
            Text("💡 Result:", font_size=20, color=SUCCESS_COLOR, weight=BOLD),
            Text("Robust anomaly detection without", font_size=18),
            Text("requiring anomalous training samples", font_size=18)
        ).arrange(DOWN, buff=0.2)
        
        self.play(Write(summary, lag_ratio=0.2, run_time=3))
        self.wait(3)
        
        # Final fade out
        self.play(FadeOut(summary), run_time=2)
        self.clear()
    
    def chapter_transition(self, title):
        """Clean transition between major topics"""
        # Only play fadeout if there are mobjects to fade out
        if self.mobjects:
            self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
        self.clear()
        
        chapter_title = Text(title, font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        self.play(Write(chapter_title), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(chapter_title), run_time=1)


# 🎬 Alternative scene for technical deep-dive
class PaDiMTechnicalDeepDive(Scene):
    """
    Technical deep-dive into PaDiM mathematics and implementation details
    
    What it shows: Mathematical foundations and algorithmic details
    Duration: ~2-3 minutes  
    Key concepts: Multivariate Gaussian, covariance matrix, feature selection
    Cleanup: Detailed mathematical explanations
    """
    
    def construct(self):
        self.show_mathematical_foundation()
        self.show_feature_selection_strategy()
        self.show_computational_efficiency()
    
    def show_mathematical_foundation(self):
        """Deep dive into the mathematical foundations"""
        title = Text("Mathematical Foundation", font_size=32, color=PRIMARY_COLOR, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Multivariate Gaussian formulation
        mvg_title = Text("Multivariate Gaussian Distribution", font_size=24, color=SECONDARY_COLOR)
        mvg_title.next_to(title, DOWN, buff=0.8)
        
        # The probability density function
        pdf_formula = MathTex(
            r"p(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)",
            font_size=24, color=WHITE
        )
        pdf_formula.next_to(mvg_title, DOWN, buff=0.5)
        
        self.play(Write(mvg_title), run_time=1)
        self.play(Write(pdf_formula), run_time=3)
        
        # Parameter explanations
        params = VGroup(
            MathTex(r"\mathbf{x} \in \mathbb{R}^d", color=BLUE),
            Text(": Feature vector at spatial position", font_size=16),
            MathTex(r"\boldsymbol{\mu} \in \mathbb{R}^d", color=GREEN),
            Text(": Mean vector (learned from normal samples)", font_size=16),
            MathTex(r"\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}", color=YELLOW),
            Text(": Covariance matrix (captures feature correlations)", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        params.to_edge(DOWN, buff=1)
        
        self.play(Write(params, lag_ratio=0.3, run_time=3))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def show_feature_selection_strategy(self):
        """Explain the feature selection and dimensionality strategy"""
        title = Text("Feature Selection Strategy", font_size=32, color=PRIMARY_COLOR, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Random feature selection visualization
        feature_matrix = VGroup()
        for i in range(8):
            row = VGroup()
            for j in range(12):
                cell = Square(side_length=0.3, stroke_width=1)
                if j in [1, 3, 5, 7, 9, 11]:  # Selected features
                    cell.set_fill(SUCCESS_COLOR, opacity=0.7)
                    cell.set_stroke(SUCCESS_COLOR, width=2)
                else:
                    cell.set_fill(GRAY, opacity=0.3)
                    cell.set_stroke(GRAY, width=1)
                row.add(cell)
            row.arrange(RIGHT, buff=0.05)
            feature_matrix.add(row)
        feature_matrix.arrange(DOWN, buff=0.05)
        
        matrix_label = Text("Feature Selection Matrix", font_size=20, color=SECONDARY_COLOR)
        matrix_label.next_to(feature_matrix, UP, buff=0.3)
        
        self.play(
            Write(matrix_label),
            Create(feature_matrix, lag_ratio=0.05, run_time=2)
        )
        
        # Explanation
        explanation = VGroup(
            Text("🎯 Random Feature Selection:", font_size=18, color=ACCENT_COLOR, weight=BOLD),
            Text("• Reduces computational complexity", font_size=14),
            Text("• Maintains discriminative power", font_size=14),
            Text("• Prevents overfitting", font_size=14),
            Text("", font_size=8),  # Spacer
            MathTex(r"d_{reduced} = \min(d_{original}, 100)", font_size=16, color=PRIMARY_COLOR),
            Text("Typical reduction: 2048 → 100 dimensions", font_size=14, color=SECONDARY_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        explanation.next_to(feature_matrix, RIGHT, buff=1)
        
        self.play(Write(explanation, lag_ratio=0.3, run_time=3))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()
    
    def show_computational_efficiency(self):
        """Show computational efficiency aspects"""
        title = Text("Computational Efficiency", font_size=32, color=PRIMARY_COLOR, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        
        # Training vs Inference comparison
        comparison = VGroup()
        
        # Training phase
        train_title = Text("Training Phase", font_size=24, color=SUCCESS_COLOR, weight=BOLD)
        train_steps = VGroup(
            Text("1. Extract patches from normal images", font_size=14),
            Text("2. Compute CNN features", font_size=14),
            Text("3. Calculate μ and Σ per position", font_size=14),
            Text("4. Store distribution parameters", font_size=14)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        train_box = SurroundingRectangle(train_steps, color=SUCCESS_COLOR, buff=0.3)
        
        train_group = VGroup(train_title, train_box, train_steps)
        train_group.to_edge(LEFT, buff=1)
        
        # Inference phase  
        infer_title = Text("Inference Phase", font_size=24, color=ACCENT_COLOR, weight=BOLD)
        infer_steps = VGroup(
            Text("1. Extract patches from test image", font_size=14),
            Text("2. Compute CNN features", font_size=14),
            Text("3. Calculate Mahalanobis distance", font_size=14),
            Text("4. Apply threshold for detection", font_size=14)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        infer_box = SurroundingRectangle(infer_steps, color=ACCENT_COLOR, buff=0.3)
        
        infer_group = VGroup(infer_title, infer_box, infer_steps)
        infer_group.to_edge(RIGHT, buff=1)
        
        self.play(
            Write(train_title),
            Create(train_box),
            Write(train_steps),
            run_time=2
        )
        self.play(
            Write(infer_title),
            Create(infer_box), 
            Write(infer_steps),
            run_time=2
        )
        
        # Performance metrics
        perf_metrics = VGroup(
            Text("⚡ Performance Highlights:", font_size=20, color=PRIMARY_COLOR, weight=BOLD),
            Text("• Fast inference: ~ms per image", font_size=16),
            Text("• Memory efficient: O(H×W×d) storage", font_size=16),
            Text("• Scalable: Linear with image size", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        perf_metrics.to_edge(DOWN, buff=1)
        
        self.play(Write(perf_metrics, lag_ratio=0.3, run_time=2))
        self.wait(3)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)
        self.clear()


# 🎪 Demo scene for quick overview
class PaDiMQuickDemo(Scene):
    """
    Quick 60-second demo of PaDiM core concept
    
    What it shows: Essential PaDiM concept in minimal time
    Duration: ~1 minute
    Key concepts: Core workflow only
    Cleanup: Rapid transitions
    """
    
    def construct(self):
        # Quick title
        title = Text("PaDiM in 60 Seconds", font_size=36, color=PRIMARY_COLOR, weight=BOLD)
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(title), run_time=0.5)
        
        # Core concept in 4 steps
        steps = [
            ("1. Extract Patches", "Divide images into local regions"),
            ("2. Get Features", "Use pre-trained CNN embeddings"),
            ("3. Learn Normal", "Model Gaussian distribution"),
            ("4. Detect Anomalies", "Calculate Mahalanobis distance")
        ]
        
        for i, (step_title, step_desc) in enumerate(steps):
            step_text = VGroup(
                Text(step_title, font_size=28, color=PRIMARY_COLOR, weight=BOLD),
                Text(step_desc, font_size=18, color=SECONDARY_COLOR)
            ).arrange(DOWN, buff=0.3)
            
            self.play(Write(step_text), run_time=1)
            self.wait(0.8)
            if i < len(steps) - 1:
                self.play(FadeOut(step_text), run_time=0.5)
        
        # Final message
        final = Text("Simple. Effective. Elegant.", font_size=24, color=SUCCESS_COLOR)
        final.next_to(step_text, DOWN, buff=1)
        self.play(Write(final), run_time=1)
        self.wait(2)


if __name__ == "__main__":
    # You can render different scenes based on your needs:
    
    # Full story (recommended):
    #manim -pql padim_animation.py PaDiMStory
    
    # Technical deep-dive:
    # manim -pql padim_animation.py PaDiMTechnicalDeepDive
    
    # Quick demo:
    # manim -pql padim_animation.py PaDiMQuickDemo
    
    pass 
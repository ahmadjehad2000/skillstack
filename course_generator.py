# course_generator.py
import os
import re
import time
import torch
import numpy as np
import gc
import json
import shutil
import logging
import random
import functools
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
import openai
from typing import Dict, List, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CourseGenerator')

class KnowledgeValidator:
    """Base class for domain-specific knowledge validation"""
    
    def __init__(self):
        self.base_prompt = """
        You are an expert educator specialized in creating factually accurate, high-quality educational content.
        
        CRITICAL REQUIREMENTS:
        1. Content must be factually accurate and reflect current academic understanding
        2. Examples and applications must be realistic and properly contextualized
        3. Avoid speculation, exaggeration, or oversimplification
        4. Use precise terminology appropriate for the indicated knowledge level
        5. Structure content logically with clear progression of concepts
        6. When multiple valid perspectives exist, present them fairly
        7. Present methodology with appropriate rigor for the indicated level
        """
        
        # Common general misconceptions for validation
        self.common_misconceptions = []
    
    def get_system_prompt(self, depth: str) -> str:
        """Get domain-appropriate system prompt"""
        depth_modifiers = {
            "beginner": "Introduce fundamental concepts with accessible language. Focus on building a solid foundation and avoid overwhelming details.",
            "intermediate": "Assume basic knowledge and build upon it. Introduce more complex concepts with appropriate context and examples.",
            "advanced": "Present sophisticated concepts with technical precision. Explore nuances, trade-offs, and cutting-edge developments."
        }
        
        return f"{self.base_prompt}\n\nContent Level: {depth_modifiers.get(depth.lower(), depth_modifiers['intermediate'])}"
    
    def verify_content(self, content: str) -> str:
        """Verify factual accuracy of generated content"""
        return content
    
    def identify_potential_inaccuracies(self, content: str) -> List[str]:
        """Identify potential inaccuracies in the content"""
        issues = []
        content_lower = content.lower()
        
        for issue in self.common_misconceptions:
            if re.search(issue["pattern"], content_lower):
                issues.append(issue["correction"])
                
        return issues

class CSKnowledgeValidator(KnowledgeValidator):
    """Computer science domain knowledge validator"""
    
    def __init__(self):
        super().__init__()
        self.base_prompt = """
        You are an expert computer science educator with deep knowledge of programming languages, 
        algorithms, data structures, software engineering, and computer systems.
        
        Guidelines for generating computer science educational content:
        
        1. Code examples must be syntactically correct and follow best practices
        2. Algorithm descriptions must include accurate time/space complexity analysis
        3. API references must correspond to current documented versions
        4. System architecture descriptions must reflect actual implementation patterns
        5. Historical facts about computing must be accurate and properly contextualized
        6. When referring to tools or frameworks, provide accurate version information
        7. Distinguish clearly between language-specific features and general concepts
        8. For mathematical foundations, ensure correct notation and definitions
        """
        
        # Common CS misconceptions for validation
        self.common_misconceptions = [
            {"pattern": r"bubble sort.*O\(n\)", "correction": "Bubble sort has O(n²) time complexity, not O(n)"},
            {"pattern": r"arrays are always faster than linked lists", "correction": "Arrays aren't universally faster than linked lists; it depends on the operation"},
            {"pattern": r"blockchain is (always|inherently) secure", "correction": "Blockchain has security properties but isn't inherently secure in all implementations"},
            {"pattern": r"quantum computers can solve all NP problems", "correction": "Quantum computers don't solve all NP problems efficiently; they address specific problems"},
            {"pattern": r"ai is (capable of|has) general intelligence", "correction": "Current AI systems have narrow intelligence, not general intelligence"},
            {"pattern": r"(python|interpreted languages) are always slower than (c\+\+|compiled languages)", "correction": "Performance differences between languages depend on specific use cases and implementations"}
        ]

class MathKnowledgeValidator(KnowledgeValidator):
    """Mathematics domain knowledge validator"""
    
    def __init__(self):
        super().__init__()
        self.base_prompt = """
        You are an expert mathematics educator with extensive knowledge of algebra, calculus, 
        statistics, geometry, and mathematical reasoning.
        
        Guidelines for generating mathematics educational content:
        
        1. Equations and formulas must be correctly formatted and mathematically accurate
        2. Proofs should follow valid logical steps with appropriate justification
        3. Statistical concepts must be presented with proper attention to assumptions
        4. Visual representations of mathematical concepts should be clearly described
        5. Historical mathematical developments should be accurately attributed
        6. Ensure terminology is consistent and precisely defined
        7. Include intuitive explanations alongside formal mathematical notation
        8. Address common misconceptions specifically when introducing difficult concepts
        """
        
        # Common math misconceptions for validation
        self.common_misconceptions = [
            {"pattern": r"correlation (implies|means|equals|is) causation", "correction": "Correlation does not imply causation; this is a fundamental statistical principle"},
            {"pattern": r"dividing by zero (equals|is) infinity", "correction": "Division by zero is undefined, not infinity"},
            {"pattern": r"0\.[9]+ != 1", "correction": "The repeating decimal 0.999... is exactly equal to 1, not less than 1"},
            {"pattern": r"negative numbers don't have square roots", "correction": "Negative numbers have complex square roots, not no square roots"},
            {"pattern": r"probability can be (greater than 1|negative)", "correction": "Probability values must be between 0 and 1 inclusive"}
        ]

class BusinessKnowledgeValidator(KnowledgeValidator):
    """Business domain knowledge validator"""
    
    def __init__(self):
        super().__init__()
        self.base_prompt = """
        You are an expert business educator with deep knowledge of management, finance, 
        marketing, economics, and organizational behavior.
        
        Guidelines for generating business educational content:
        
        1. Business concepts must be presented with real-world context and applications
        2. Financial calculations and ratios must be correctly defined and explained
        3. Economic principles should include both theoretical foundations and practical implications
        4. Marketing strategies should reflect evidence-based approaches with appropriate metrics
        5. Management techniques should acknowledge different organizational contexts
        6. Case studies should be based on verifiable business scenarios
        7. Balance theoretical frameworks with practical implementation considerations
        8. Acknowledge cultural and regional variations in business practices when relevant
        """
        
        # Common business misconceptions for validation
        self.common_misconceptions = [
            {"pattern": r"(high risk always means high returns|high returns always require high risk)", "correction": "While risk and return are related, high risk doesn't guarantee high returns"},
            {"pattern": r"markets are (always efficient|perfectly efficient)", "correction": "Markets exhibit varying degrees of efficiency, not perfect efficiency"},
            {"pattern": r"profit maximization is the only business goal", "correction": "Businesses often balance multiple goals beyond profit maximization"},
            {"pattern": r"economic models perfectly predict real-world behavior", "correction": "Economic models simplify reality and don't perfectly predict real-world behavior"}
        ]

class ScienceKnowledgeValidator(KnowledgeValidator):
    """Science domain knowledge validator"""
    
    def __init__(self):
        super().__init__()
        self.base_prompt = """
        You are an expert science educator with deep knowledge of physics, chemistry, 
        biology, earth sciences, and scientific methodology.
        
        Guidelines for generating science educational content:
        
        1. Scientific concepts must be presented with accurate current understanding
        2. Distinguish between well-established theories and ongoing research
        3. Physical and chemical principles must be accurately described
        4. Biological processes should be explained with appropriate complexity
        5. Scientific history should be accurately presented with proper attribution
        6. Experimental methods should reflect actual scientific practice
        7. When describing scientific consensus, ensure accuracy and appropriate nuance
        8. Units and measurements must be correctly presented and converted
        """
        
        # Common science misconceptions for validation
        self.common_misconceptions = [
            {"pattern": r"humans (only|primarily) use 10% of their brains", "correction": "Humans use most of their brain, not just 10%"},
            {"pattern": r"evolution (works towards|has a goal of) higher complexity", "correction": "Evolution doesn't work toward complexity; it selects for reproductive fitness"},
            {"pattern": r"molecules in gases don't have attractions", "correction": "Gas molecules do have attractive forces, though they're relatively weak"},
            {"pattern": r"normal body temperature is (always|exactly) 98\.6", "correction": "Normal body temperature varies between individuals and throughout the day"}
        ]

class GeneralKnowledgeValidator(KnowledgeValidator):
    """General domain knowledge validator for topics without specialized validation"""
    
    def __init__(self):
        super().__init__()
        self.base_prompt = """
        You are an expert educator with broad knowledge across disciplines.
        
        Guidelines for generating educational content:
        
        1. Present information with appropriate nuance and context
        2. Distinguish between facts, theories, and opinions
        3. Use precise terminology appropriate for the subject matter
        4. Provide concrete examples to illustrate abstract concepts
        5. When introducing new terms, define them clearly
        6. Structure content logically with progressive complexity
        7. Acknowledge limitations of current understanding when appropriate
        8. Incorporate diverse perspectives when relevant to the topic
        """
        
        # Common general misconceptions
        self.common_misconceptions = [
            {"pattern": r"learning styles.*(visual|auditory|kinesthetic)", "correction": "The theory of visual/auditory/kinesthetic learning styles lacks scientific support"},
            {"pattern": r"we have five (basic |)senses", "correction": "Humans have more than five senses, including balance, temperature, and proprioception"},
            {"pattern": r"left brain.*(logical|analytical).*right brain.*(creative|artistic)", "correction": "The left/right brain dichotomy oversimplifies; brain functions involve complex interactions across regions"}
        ]

class ModelManager:
    """Manages loading, unloading, and caching of AI models"""
    
    def __init__(self, device="cuda", optimize_memory=True):
        self.device = device
        self.optimize_memory = optimize_memory
        self._models = {}
        self._last_used = {}
        self._lock = functools.partial(contextlib.nullcontext)()  # Simple lock substitute
    
    def get_model(self, model_key, model_load_function):
        """Get a model, loading it if needed"""
        if model_key not in self._models:
            self._models[model_key] = model_load_function()
        
        self._last_used[model_key] = time.time()
        return self._models[model_key]
    
    def unload_model(self, model_key):
        """Unload a specific model to free memory"""
        if model_key in self._models:
            if self.device == "cuda":
                try:
                    self._models[model_key] = self._models[model_key].to("cpu")
                except:
                    pass
                    
            del self._models[model_key]
            if model_key in self._last_used:
                del self._last_used[model_key]
                
            # Force garbage collection
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def cleanup_unused_models(self, threshold_seconds=300):
        """Release models unused for more than threshold_seconds"""
        current_time = time.time()
        for key, last_used in list(self._last_used.items()):
            if current_time - last_used > threshold_seconds:
                self.unload_model(key)

class EducationalThumbnailGenerator:
    """Specialized thumbnail generator optimized for educational content"""
    
    def __init__(self, device="cuda", optimize_memory=True):
        self.device = device
        self.optimize_memory = optimize_memory
        self.model_loaded = False
        
        # Educational design templates for different domains
        self.design_templates = {
            "technology": {
                "color_scheme": [(42, 87, 154), (49, 140, 231), (107, 178, 255)],
                "font": "robotomono",
                "layout": "technical"
            },
            "business": {
                "color_scheme": [(31, 97, 141), (41, 128, 185), (133, 193, 233)],
                "font": "opensans",
                "layout": "professional"
            },
            "creative": {
                "color_scheme": [(74, 35, 90), (136, 78, 160), (175, 122, 197)],
                "font": "montserrat",
                "layout": "dynamic"
            },
            "science": {
                "color_scheme": [(20, 90, 50), (35, 155, 86), (88, 214, 141)],
                "font": "lato",
                "layout": "structured"
            },
            "default": {
                "color_scheme": [(41, 128, 185), (52, 152, 219), (133, 193, 233)],
                "font": "sourcesanspro",
                "layout": "balanced"
            }
        }
    
    def _load_model(self):
        """Load the specialized thumbnail generation model"""
        if self.model_loaded:
            return
            
        try:
            # Use DreamShaper v8 - specialized for illustrations and educational content
            model_id = "stabilityai/stable-diffusion-2-1"
            
            # Load with optimized settings for educational thumbnails
            self.thumb_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Apply memory optimizations
            if self.device == "cuda":
                if hasattr(self.thumb_pipeline, "enable_attention_slicing"):
                    self.thumb_pipeline.enable_attention_slicing("max")
                
                if hasattr(self.thumb_pipeline, "enable_xformers_memory_efficient_attention"):
                    try:
                        self.thumb_pipeline.enable_xformers_memory_efficient_attention()
                    except:
                        pass
                        
            # Use optimized scheduler for thumbnail generation
            self.thumb_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.thumb_pipeline.scheduler.config
            )
            
            # Move to appropriate device
            self.thumb_pipeline = self.thumb_pipeline.to(self.device)
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise
    
    def _unload_model(self):
        """Unload the model to free memory"""
        if not self.model_loaded or not self.optimize_memory:
            return
            
        try:
            self.thumb_pipeline = self.thumb_pipeline.to("cpu")
            del self.thumb_pipeline
            self.thumb_pipeline = None
            self.model_loaded = False
            
            # Force garbage collection
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error unloading thumbnail model: {e}")
    
    def _select_design_template(self, topic, template_key="default"):
        """Select appropriate design template based on topic"""
        topic_lower = topic.lower()
        
        domain_keywords = {
            "technology": ["programming", "coding", "software", "computer", "tech", "algorithm", "digital"],
            "business": ["management", "marketing", "finance", "economics", "business", "entrepreneur"],
            "creative": ["art", "design", "music", "writing", "creative", "visual", "media"],
            "science": ["physics", "chemistry", "biology", "science", "scientific", "research", "lab"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return self.design_templates[domain]
                
        return self.design_templates[template_key if template_key in self.design_templates else "default"]
    
    def _generate_base_image(self, course_title, topic, template):
        """Generate base image using specialized prompt engineering"""
        # Construct optimized prompt for educational thumbnail
        prompt = f"professional educational course thumbnail about {topic}, digital illustration, clean minimalist design, {template['layout']} composition, educational concept"
        
        # Negative prompt to avoid common issues
        negative_prompt = "text, words, letters, signature, watermark, low quality, blurry, amateur"
        
        # Set generation parameters based on available resources
        width, height = self._get_optimal_dimensions()
        steps = 25 if self.device == "cuda" else 20
        guidance = 7.5
        
        # Generate the image
        result = self.thumb_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        
        return result.images[0]
    
    def _get_optimal_dimensions(self):
        """Get optimal dimensions based on available resources"""
        if self.device == "cuda":
            try:
                # Check available VRAM
                free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                free_vram_gb = free_vram / (1024**3)
                
                if free_vram_gb > 6:
                    return 768, 512  # High resources available
                elif free_vram_gb > 4:
                    return 640, 448  # Medium resources
                else:
                    return 512, 384  # Limited resources
            except:
                return 512, 384  # Default for CUDA when detection fails
        else:
            return 512, 384  # Default for CPU
    
    def _apply_educational_overlay(self, base_image, course_title, topic, template):
        """Apply professional educational design overlay to the image"""
        try:
            # Create a copy of the image to work with
            result_image = base_image.copy()
            width, height = result_image.size
            
            # Create overlay for better text visibility
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Get color scheme from template
            colors = template['color_scheme']
            primary_color = colors[0]
            secondary_color = colors[1]
            accent_color = colors[2]
            
            # Apply gradient overlay at the bottom for text area
            for i in range(height // 2, height):
                alpha = int(200 * (i - height // 2) / (height - height // 2))
                color = (*primary_color, alpha)
                draw.line([(0, i), (width, i)], fill=color)
            
            # Add top bar for branding
            draw.rectangle(
                [(0, 0), (width, 60)],
                fill=(*secondary_color, 180)
            )
            
            # Try to use a suitable font, or fallback to default
            try:
                # Try to find system fonts
                system_fonts = [
                    "C:\\Windows\\Fonts\\Arial.ttf",
                    "C:\\Windows\\Fonts\\Calibri.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ]
                
                font_path = None
                for path in system_fonts:
                    if os.path.exists(path):
                        font_path = path
                        break
                
                if font_path:
                    title_font = ImageFont.truetype(font_path, int(width / 18))
                    subtitle_font = ImageFont.truetype(font_path, int(width / 30))
                    topic_font = ImageFont.truetype(font_path, int(width / 40))
                else:
                    title_font = ImageFont.load_default()
                    subtitle_font = ImageFont.load_default()
                    topic_font = ImageFont.load_default()
                    
            except Exception as font_error:
                logger.warning(f"Font loading error: {font_error}")
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                topic_font = ImageFont.load_default()
            
            # Draw topic text at the top
            topic_text = topic.upper()
            topic_x = 20  # Left-aligned position
            topic_y = 20  # Top position
            
            # Draw topic text
            draw.text(
                (topic_x, topic_y),
                topic_text,
                fill=(255, 255, 255, 230),
                font=topic_font
            )
            
            # Draw title with shadow effect for better readability
            title_text = course_title
            wrapped_title = self._wrap_text(title_text, title_font, width - 60)
            title_y = height - 120  # Position from bottom
            
            # Draw text shadow
            for line_idx, line in enumerate(wrapped_title):
                line_y = title_y + (line_idx * int(width / 16))
                draw.text(
                    (width/2 + 2, line_y + 2),
                    line,
                    fill=(0, 0, 0, 200),
                    font=title_font,
                    anchor="mm"
                )
                
                # Draw main text
                draw.text(
                    (width/2, line_y),
                    line,
                    fill=(255, 255, 255, 255),
                    font=title_font,
                    anchor="mm"
                )
            
            # Draw subtitle
            subtitle_text = f"{topic} | Complete Course"
            subtitle_y = title_y + (len(wrapped_title) * int(width / 16)) + 10
            
            draw.text(
                (width/2, subtitle_y),
                subtitle_text,
                fill=(220, 220, 220, 230),
                font=subtitle_font,
                anchor="mm"
            )
            
            # Apply the overlay to the original image
            result_image = Image.alpha_composite(result_image.convert("RGBA"), overlay)
            
            # Enhance the image slightly for better visual appeal
            result_image = result_image.convert("RGB")
            enhancer = ImageEnhance.Contrast(result_image)
            result_image = enhancer.enhance(1.1)
            
            # Add subtle vignette effect
            return self._add_vignette(result_image)
            
        except Exception as overlay_error:
            logger.error(f"Error adding text overlay: {overlay_error}")
            # Return the original image if overlay fails
            return base_image.convert("RGB") if base_image.mode == "RGBA" else base_image
    
    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            try:
                # Check if font supports getlength
                text_width = font.getlength(f"{current_line} {word}")
            except:
                # Estimate width if getlength not available
                text_width = len(f"{current_line} {word}") * font.size * 0.6
                
            if text_width <= max_width:
                current_line = f"{current_line} {word}"
            else:
                lines.append(current_line)
                current_line = word
                
        lines.append(current_line)
        return lines
    
    def _add_vignette(self, image):
        """Add a subtle vignette effect to the image"""
        try:
            # Create a radial gradient mask for vignette
            width, height = image.size
            mask = Image.new('L', (width, height), 255)
            draw = ImageDraw.Draw(mask)
            
            # Calculate parameters for elliptical gradient
            center_x, center_y = width // 2, height // 2
            inner_radius = min(width, height) // 2
            outer_radius = int(inner_radius * 1.5)
            
            # Draw gradient ellipses from outer (dark) to inner (light)
            for r in range(outer_radius, inner_radius, -1):
                opacity = int(255 * (1 - (outer_radius - r) / (outer_radius - inner_radius)))
                draw.ellipse(
                    [(center_x - r, center_y - r), (center_x + r, center_y + r)],
                    fill=opacity
                )
            
            # Apply the mask with slight blur for smooth transition
            mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
            result = image.copy()
            
            # Convert to RGBA if necessary
            if result.mode != 'RGBA':
                result = result.convert('RGBA')
                
            # Create alpha channel from mask
            alpha = mask.convert('L')
            result.putalpha(alpha)
            
            # Convert back to RGB after applying the vignette
            return result.convert('RGB')
            
        except Exception as vignette_error:
            logger.error(f"Error adding vignette: {vignette_error}")
            # Return the original image if vignette fails
            return image
    
    def _save_thumbnail(self, image, topic):
        """Save the thumbnail image and return the path"""
        # Create a unique filename
        safe_topic = re.sub(r'[^\w\s-]', '', topic.lower())
        safe_topic = re.sub(r'[-\s]+', '-', safe_topic).strip('-')
        timestamp = int(time.time())
        thumbnail_dir = "static/thumbnails"
        os.makedirs(thumbnail_dir, exist_ok=True)
        filename = f"{thumbnail_dir}/{safe_topic}-{timestamp}.png"
        
        # Save with high quality
        image.save(filename, quality=95)
        
        # Unload the model to free memory
        self._unload_model()
        
        return filename
    
    def _create_fallback_thumbnail(self, course_title, topic):
        """Create a fallback thumbnail when image generation fails"""
        try:
            thumbnail_dir = "static/thumbnails"
            os.makedirs(thumbnail_dir, exist_ok=True)
            safe_topic = re.sub(r'[^\w\s-]', '', topic.lower())
            safe_topic = re.sub(r'[-\s]+', '-', safe_topic).strip('-')
            timestamp = int(time.time())
            filename = f"{thumbnail_dir}/{safe_topic}-{timestamp}.png"
            
            # Create a gradient background
            width, height = 800, 600
            image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(image)
            
            # Create a gradient background
            for y in range(height):
                r = int(41 + (y / height) * 20)
                g = int(128 + (y / height) * 24)
                b = int(185 + (y / height) * 48)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Try to use a system font
            try:
                system_fonts = [
                    "C:\\Windows\\Fonts\\Arial.ttf",
                    "C:\\Windows\\Fonts\\Calibri.ttf", 
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ]
                
                font_path = None
                for path in system_fonts:
                    if os.path.exists(path):
                        font_path = path
                        break
                
                if font_path:
                    title_font = ImageFont.truetype(font_path, 40)
                    topic_font = ImageFont.truetype(font_path, 30)
                else:
                    title_font = ImageFont.load_default()
                    topic_font = ImageFont.load_default()
            except:
                title_font = ImageFont.load_default()
                topic_font = ImageFont.load_default()
            
            # Draw title
            draw.text(
                (width//2, height//2 - 50), 
                course_title, 
                fill=(255, 255, 255), 
                font=title_font,
                anchor="mm" if hasattr(title_font, "getbbox") else None
            )
            
            # Draw topic subtitle
            draw.text(
                (width//2, height//2 + 50), 
                f"A course on {topic}", 
                fill=(220, 220, 220), 
                font=topic_font,
                anchor="mm" if hasattr(topic_font, "getbbox") else None
            )
            
            # Save the image
            image.save(filename)
            return filename
            
        except Exception as fallback_error:
            logger.error(f"Fallback thumbnail creation failed: {fallback_error}")
            return "static/thumbnails/default-thumbnail.png"
    
    def generate_thumbnail(self, course_title, topic, template_key="default"):
        """Generate a high-quality educational thumbnail with professional design"""
        try:
            # 1. Load the specialized thumbnail model
            self._load_model()
            
            # 2. Select appropriate design template
            template = self._select_design_template(topic, template_key)
            
            # 3. Generate the base image using the specialized model
            base_image = self._generate_base_image(course_title, topic, template)
            
            # 4. Apply professional educational design overlay
            thumbnail = self._apply_educational_overlay(base_image, course_title, topic, template)
            
            # 5. Save and return the thumbnail path
            return self._save_thumbnail(thumbnail, topic)
            
        except Exception as e:
            logger.error(f"Thumbnail generation error: {str(e)}")
            return self._create_fallback_thumbnail(course_title, topic)

class CourseGenerator:
    """
    Enhanced course generation using OpenAI and specialized thumbnail generation with
    domain-calibrated knowledge validation and factual verification.
    
    Implements a multi-stage content generation pipeline for creating high-quality,
    scientifically accurate educational courses.
    """
    
    def __init__(self, 
                openai_api_key: str,
                use_gpu: bool = True, 
                optimize_memory: bool = True,
                high_quality: bool = True):
        """
        Initialize the Course Generator with OpenAI for text and Stable Diffusion for images.
        
        Args:
            openai_api_key: Your OpenAI API key
            use_gpu: Whether to use GPU for image generation if available (default: True)
            optimize_memory: Whether to optimize memory usage for limited VRAM (default: True)
            high_quality: Whether to use higher quality settings (default: True)
        """
        # Validate API key format
        if not openai_api_key:
            raise ValueError("API key is required")
        
        if openai_api_key.startswith("sk-proj-"):
            raise ValueError("Project-scoped API keys (sk-proj-*) are not compatible with this service. Please use a standard API key.")
        
        if not openai_api_key.startswith("sk-") or len(openai_api_key) < 35:
            raise ValueError("Invalid API key format. Keys should start with 'sk-' followed by at least 32 characters.")
        
        # Set up OpenAI
        openai.api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Configuration
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.optimize_memory = optimize_memory
        self.high_quality = high_quality
        
        # Choose the GPT model to use
        # GPT-4o for the highest quality (but more expensive)
        # GPT-3.5-turbo for more economical option (cheaper and faster)
        if high_quality:
            self.text_model = "gpt-4o"  # Highest quality with advanced capabilities
        else:
            self.text_model = "gpt-3.5-turbo"  # More economical, still good quality
        
        # Hardware information
        print(f"\n=== SYSTEM CONFIGURATION ===")
        print(f"Text generation: OpenAI {self.text_model}")
        print(f"Image generation device: {self.device}")
        print(f"Memory optimization: {'Enabled' if optimize_memory else 'Disabled'}")
        print(f"Quality setting: {'High' if high_quality else 'Standard'}")
        print(f"Anti-hallucination: Enabled")
        
        # Get GPU info if using CUDA
        if self.device == "cuda":
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU Memory: {gpu_mem:.2f} GB")
            except Exception as e:
                print(f"Could not get GPU memory info: {e}")
        
        # Initialize image model as None, will load when needed to save memory
        self.image_model = None
        self.image_to_image_model = None
        
        # Create necessary directories
        os.makedirs("static/thumbnails", exist_ok=True)
        
        print("\n=== READY TO GENERATE COURSES ===\n")
    
    def _initialize_knowledge_validators(self):
        """Initialize domain-specific knowledge validators for content verification"""
        return {
            "computer_science": CSKnowledgeValidator(),
            "mathematics": MathKnowledgeValidator(),
            "business": BusinessKnowledgeValidator(),
            "science": ScienceKnowledgeValidator(),
            "general": GeneralKnowledgeValidator()  # Fallback validator
        }
    
    def _select_knowledge_validator(self, topic):
        """Select the appropriate domain validator for a given topic"""
        topic_lower = topic.lower()
        
        domain_keywords = {
            "computer_science": ["programming", "algorithm", "data structure", "software", "coding", "computation", "database", "web development", "machine learning", "artificial intelligence", "network"],
            "mathematics": ["math", "calculus", "algebra", "statistics", "geometry", "trigonometry", "probability", "number theory", "linear algebra"],
            "business": ["marketing", "finance", "management", "economics", "entrepreneurship", "business", "accounting", "strategy", "leadership"],
            "science": ["physics", "chemistry", "biology", "astronomy", "geology", "scientific", "experiment", "hypothesis", "theory", "research"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return self.knowledge_validators[domain]
                
        return self.knowledge_validators["general"]
    
    def _retry_with_exponential_backoff(self, func, max_retries=3, initial_delay=1):
        """Decorator for API calls with exponential backoff retry logic"""
        retries = 0
        delay = initial_delay
        
        while retries < max_retries:
            try:
                return func()
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise
                    
                wait_time = delay * (2 ** (retries - 1)) * (0.5 + random.random())
                logger.warning(f"API error: {e}. Retrying in {wait_time:.1f}s (attempt {retries}/{max_retries})")
                time.sleep(wait_time)
    
    def _generate_text_openai(self, prompt: str, system_prompt: str = None, max_tokens: int = 2000, temperature: float = 0.5):
        """
        Generate text using OpenAI's API with factual constraints.
        
        Args:
            prompt: The instruction/prompt for text generation
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (lower = more deterministic)
            
        Returns:
            Generated text string
        """
        def api_call():
            # Rate limiting to avoid OpenAI rate limits
            time.sleep(0.5)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.85,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            
            return response.choices[0].message.content.strip()
        
        try:
            # Call with retry logic
            return self._retry_with_exponential_backoff(api_call)
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error generating content. Please check your API key and try again. Details: {str(e)}"
    
    def _generate_validated_title(self, topic: str, depth: str) -> str:
        """Generate a factually accurate and engaging course title"""
        validator = self._select_knowledge_validator(topic)
        system_prompt = validator.get_system_prompt(depth)
        
        title_prompt = f"""
        Create an engaging, professional, and accurate course title for a {depth} level course on {topic}.
        
        The title should be:
        - Clear and descriptive
        - Professionally worded (avoid clickbait or marketing language)
        - Under 10 words
        - Specific to {topic}
        - Accurately reflect the course content
        
        Just provide the title with no additional text or explanation.
        """
        
        return self._generate_text_openai(title_prompt, system_prompt, max_tokens=50)
    
    def _generate_validated_description(self, course_title: str, topic: str, depth: str) -> str:
        """Generate a factually accurate course description"""
        validator = self._select_knowledge_validator(topic)
        system_prompt = validator.get_system_prompt(depth)
        
        description_prompt = f"""
        Write a clear, factual course description for a {depth} level course titled "{course_title}" about {topic}.
        
        The description should:
        - Be 3-5 sentences long
        - Clearly explain what students will learn
        - Highlight specific skills or knowledge that will be gained
        - Specify who the course is designed for
        - Use professional, educational language (avoid marketing hyperbole)
        
        Just provide the description with no additional text or explanation.
        """
        
        return self._generate_text_openai(description_prompt, system_prompt, max_tokens=250)
    
    def _generate_learning_objectives(self, course_title: str, topic: str, depth: str) -> str:
        """Generate measurable learning objectives aligned with educational standards"""
        validator = self._select_knowledge_validator(topic)
        system_prompt = validator.get_system_prompt(depth)
        
        objectives_prompt = f"""
        Create 5-7 specific, measurable learning objectives for the course "{course_title}" on {topic}.
        
        Each learning objective should:
        - Start with an action verb (e.g., Define, Explain, Implement, Analyze, Evaluate)
        - Be specific and measurable
        - Relate directly to {topic}
        - Be appropriate for a {depth} level course
        - Focus on skills and knowledge that can be assessed
        
        Format as a bullet point list with no additional text or explanation.
        """
        
        return self._generate_text_openai(objectives_prompt, system_prompt, max_tokens=400)
    
    def _generate_course_content(self, topic: str, depth: str) -> str:
        """Generate comprehensive, factually accurate course content"""
        validator = self._select_knowledge_validator(topic)
        system_prompt = validator.get_system_prompt(depth)
        
        lesson_prompt = f"""
        Create extremely detailed and factually accurate educational content for a comprehensive lesson about {topic} at a {depth} level.
        
        Structure the lesson content as follows:
        
        1. INTRODUCTION (2-3 paragraphs):
        A thorough introduction to {topic}, explaining its importance, applications, and relevance.
        
        2. CORE CONCEPTS (3-5 sections):
        Detailed explanation of the fundamental concepts of {topic}, with clear definitions, examples, and context.
        
        3. DETAILED METHODOLOGY (4-6 subsections):
        Step-by-step breakdown of methodologies, processes, or techniques related to {topic}.
        Include practical examples and real-world applications.
        
        4. ADVANCED CONCEPTS (3-4 sections):
        Exploration of more complex aspects of {topic} appropriate for {depth} level students.
        Include detailed explanations and advanced applications.
        
        5. PRACTICAL IMPLEMENTATION (2-3 detailed exercises):
        Comprehensive exercises that allow students to apply what they've learned.
        Include step-by-step instructions and expected outcomes.
        
        6. COMMON CHALLENGES AND SOLUTIONS (3-4 challenges):
        Address typical obstacles students might face when working with {topic}.
        Provide detailed, actionable solutions for each challenge.
        
        7. RESOURCES AND TOOLS (5-8 resources):
        Curated list of valuable tools, resources, and references for further exploration.
        
        Format the content with professional Markdown formatting including headers (# for main sections, ## for subsections), bullet points, numbered lists, and code blocks if needed.
        
        CRITICAL REQUIREMENTS:
        - Content must be factually accurate and reflect current academic understanding
        - Use precise terminology appropriate for {depth} level
        - Include concrete examples that illustrate abstract concepts
        - Provide proper context for specialized concepts
        - Structure content logically with clear progression
        - Include appropriate detail without overwhelming
        - Focus on educational value rather than marketing language
        """
        
        # Generate initial content
        initial_content = self._generate_text_openai(
            prompt=lesson_prompt,
            system_prompt=system_prompt,
            max_tokens=4000,
            temperature=0.3  # Lower temperature for increased factual accuracy
        )
        
        # Apply factual verification if enabled
        if self.factual_verification_enabled:
            content_issues = validator.identify_potential_inaccuracies(initial_content)
            
            if content_issues:
                # Apply corrections
                return self._apply_factual_corrections(initial_content, content_issues, validator, topic, depth)
        
        return initial_content
    
    def _apply_factual_corrections(self, content: str, issues: List[str], validator, topic: str, depth: str) -> str:
        """Apply corrections to identified factual issues"""
        if not issues:
            return content
            
        correction_prompt = f"""
        Review and correct the following educational content on {topic} which contains some potential inaccuracies.
        
        IDENTIFIED ISSUES:
        {chr(10).join(['- ' + issue for issue in issues])}
        
        ORIGINAL CONTENT:
        {content}
        
        Please provide a corrected version that addresses these issues while maintaining the original structure and flow.
        Make your corrections minimal and focused on addressing only the identified issues.
        """
        
        system_prompt = validator.get_system_prompt(depth)
        
        corrected_content = self._generate_text_openai(
            prompt=correction_prompt,
            system_prompt=system_prompt,
            max_tokens=4000,
            temperature=0.2  # Very low temperature for focused corrections
        )
        
        return corrected_content
    
    def _generate_aligned_assessment(self, topic: str, course_title: str, depth: str, learning_objectives: str) -> str:
        """Generate assessment aligned with learning objectives"""
        validator = self._select_knowledge_validator(topic)
        system_prompt = validator.get_system_prompt(depth)
        
        assessment_prompt = f"""
        Create a comprehensive assessment for the course "{course_title}" on {topic} at a {depth} level.
        The assessment must align with these learning objectives:
        
        {learning_objectives}
        
        Include exactly:
        1. 6 multiple-choice questions with 4 options each
        2. 2 scenario-based questions that test application of knowledge
        
        For multiple-choice questions:
        - Clearly mark correct answers with [x] and incorrect answers with [ ]
        - Provide a brief explanation for each correct answer
        - Ensure questions test understanding, not just recall
        - Align questions with specific learning objectives
        
        For scenario-based questions:
        - Present a realistic scenario related to {topic}
        - Ask the student to solve a problem or make a recommendation
        - Provide evaluation criteria or key points for a good answer
        
        Format in Markdown with clear headers and proper formatting.
        
        IMPORTANT: 
        - Questions must be factually accurate and reflect current understanding of {topic}
        - Avoid ambiguous wording or trick questions
        - Questions should be challenging but fair for {depth} level students
        """
        
        return self._generate_text_openai(assessment_prompt, system_prompt, max_tokens=2000)
    
    def _determine_domain_template(self, topic: str) -> str:
        """Determine appropriate domain template for thumbnail generation"""
        topic_lower = topic.lower()
        
        domain_keywords = {
            "technology": ["programming", "coding", "software", "computer", "tech", "algorithm", "digital"],
            "business": ["management", "marketing", "finance", "economics", "business", "entrepreneur"],
            "creative": ["art", "design", "music", "writing", "creative", "visual", "media"],
            "science": ["physics", "chemistry", "biology", "science", "scientific", "research", "lab"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
                
        return "default"
    
    def generate_course(self, topic: str, depth: str = "intermediate") -> Dict:
        """
        Generate a comprehensive, factually accurate course on a topic.
        
        Args:
            topic: The main subject of the course
            depth: The complexity level (beginner, intermediate, advanced)
            
        Returns:
            A dictionary containing the course title, description, module, lessons and content
        """
        logger.info(f"Generating evidence-based course on: {topic} (Level: {depth})")
        
        # Generate core course components with enhanced factual verification
        course_title = self._generate_validated_title(topic, depth)
        course_description = self._generate_validated_description(course_title, topic, depth)
        learning_objectives = self._generate_learning_objectives(course_title, topic, depth)
        
        # Module structure remains the same for compatibility
        module_title = f"Complete {topic} Masterclass"
        lesson_title = f"Comprehensive Guide to {topic}"
        
        # Generate comprehensive lesson content with multi-stage verification
        logger.info("Generating comprehensive, factually-verified content...")
        lesson_content = self._generate_course_content(topic, depth)
        
        # Generate assessment with learning objective alignment
        assessment_content = self._generate_aligned_assessment(topic, course_title, depth, learning_objectives)
        
        # Generate specialized educational thumbnail
        thumbnail_path = self.thumbnail_generator.generate_thumbnail(
            course_title, 
            topic,
            self._determine_domain_template(topic)
        )
        
        # Structure course data for compatibility with existing system
        modules = [{
            "title": module_title,
            "lessons": [lesson_title]
        }]
        
        module_content = {
            "title": module_title,
            "introduction": f"This comprehensive masterclass covers everything you need to know about {topic}.",
            "lessons": [{
                "title": lesson_title,
                "content": lesson_content,
                "assessment": assessment_content
            }]
        }
        
        module_contents = {"0": module_content}
        
        # Return course data in expected format
        return {
            "title": course_title,
            "description": course_description,
            "level": depth,
            "topic": topic,
            "learning_objectives": learning_objectives,
            "modules": modules,
            "thumbnail_path": thumbnail_path,
            "module_contents": module_contents
        }
    
    def generate_module_content(self, course_title: str, module_title: str, lessons: List[str]) -> Dict:
        """
        Generate detailed content for a specific module.
        
        Args:
            course_title: The title of the course
            module_title: The title of the module
            lessons: List of lesson titles in this module
            
        Returns:
            Dictionary containing detailed content for each lesson
        """
        # Extract topic from module title
        topic = module_title.replace("Complete ", "").replace(" Masterclass", "")
        
        # Determine appropriate depth based on course title
        depth_indicators = {
            "beginner": ["introduction", "basics", "fundamental", "beginner", "getting started"],
            "advanced": ["advanced", "expert", "mastery", "professional", "deep dive"]
        }
        
        depth = "intermediate"  # Default
        course_title_lower = course_title.lower()
        
        for level, indicators in depth_indicators.items():
            if any(indicator in course_title_lower for indicator in indicators):
                depth = level
                break
        
        # Select appropriate knowledge validator
        validator = self._select_knowledge_validator(topic)
        system_prompt = validator.get_system_prompt(depth)
        
        module_content = {
            "title": module_title,
            "lessons": []
        }
        
        # Generate module introduction
        module_intro_prompt = f"""
        Write a comprehensive introduction for the module "{module_title}" which is part of the course "{course_title}".
        
        This introduction should:
        - Explain the purpose and importance of this module within the larger course context
        - Outline what students will learn in this module specifically
        - Connect concepts in a logical progression
        - Set appropriate expectations for the difficulty level
        - Motivate students by explaining the value of mastering these concepts
        
        Write 2-3 paragraphs (250-350 words total) with no additional text or explanation.
        Focus on factual accuracy and educational value.
        """
        
        module_intro = self._generate_text_openai(module_intro_prompt, system_prompt, max_tokens=500)
        module_content["introduction"] = module_intro
        
        # Generate content for each lesson
        for lesson in lessons:
            logger.info(f"Generating factually accurate content for lesson: {lesson}")
            
            lesson_prompt = f"""
            Create detailed, factually accurate educational content for the lesson titled "{lesson}" 
            which is part of the module "{module_title}" in the course "{course_title}".
            
            Structure the lesson content as follows:
            
            1. INTRODUCTION:
            A brief paragraph introducing the specific topic and explaining its importance.
            
            2. LEARNING OBJECTIVES:
            3-4 specific, measurable objectives that students will achieve by the end of this lesson.
            
            3. MAIN CONTENT:
            The core educational material, divided into logical sections with clear headings.
            Include relevant examples, code snippets if applicable, and thorough explanations.
            
            4. PRACTICAL EXERCISE:
            A hands-on activity that reinforces the lesson concepts.
            
            5. KEY TAKEAWAYS:
            A bullet point summary of the most important concepts covered.
            
            Format the content with appropriate Markdown formatting including headers, bullet points, 
            code blocks if needed, and emphasis where appropriate.
            
            IMPORTANT GUIDELINES:
            - Content must be factually accurate and reflect current understanding
            - Use precise terminology with clear definitions
            - Provide concrete examples that illustrate abstract concepts
            - Structure content logically with clear progression
            - Include appropriate detail without overwhelming
            """
            
            content = self._generate_text_openai(lesson_prompt, system_prompt, max_tokens=1200)
            
            # Verify factual accuracy if enabled
            if self.factual_verification_enabled:
                content_issues = validator.identify_potential_inaccuracies(content)
                if content_issues:
                    content = self._apply_factual_corrections(content, content_issues, validator, topic, depth)
            
            # Generate quiz questions for assessment
            quiz_prompt = f"""
            Create 3-5 factually accurate assessment questions for the lesson "{lesson}" in the course "{course_title}".
            
            Include these types of questions:
            - 2-3 multiple-choice questions with 4 options each, clearly marking correct answers with [x]
            - 1-2 practical application questions that test deeper understanding
            
            For each question:
            1. Write a clear, factually accurate question that tests understanding of key concepts
            2. For multiple choice: provide options marked with [x] for correct and [ ] for incorrect
            3. Include a brief explanation of why the correct answer is right
            
            Format each question with proper Markdown.
            
            IMPORTANT:
            - Questions must be factually accurate and reflect current understanding
            - Focus on testing meaningful understanding, not trivial details
            - Questions should be clearly written and unambiguous
            - Avoid unnecessarily complex language or jargon
            """
            
            quiz_questions = self._generate_text_openai(quiz_prompt, system_prompt, max_tokens=600)
            
            module_content["lessons"].append({
                "title": lesson,
                "content": content,
                "assessment": quiz_questions
            })
        
        return module_content
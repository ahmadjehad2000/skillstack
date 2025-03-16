import os
import re
import time
import torch
import numpy as np
import gc
import json
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    EulerDiscreteScheduler
)
import openai
from typing import Dict, List, Tuple, Optional

class CourseGenerator:
    """
    Advanced course generation using OpenAI and Stable Diffusion with
    robust anti-hallucination measures, content quality controls, and
    optimized CUDA memory management.
    
    This class generates comprehensive educational courses with high-quality
    content and custom thumbnails, preventing common hallucination issues
    while maximizing GPU utilization efficiency.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 use_gpu: bool = True, 
                 optimize_memory: bool = True,
                 high_quality: bool = True):
        """
        Initialize with advanced CUDA optimizations and memory management
        
        Args:
            openai_api_key: Your OpenAI API key
            use_gpu: Whether to use GPU acceleration (default: True)
            optimize_memory: Whether to use aggressive memory optimization (default: True)
            high_quality: Whether to use higher quality models (default: True)
        """
        # Set up OpenAI
        openai.api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Enhanced GPU configuration with improved diagnostics
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.optimize_memory = optimize_memory
        self.high_quality = high_quality
        
        # GPU diagnostics and configuration
        if self.device == "cuda":
            gpu_initialized = self._setup_gpu_environment()
            
            # Fall back to CPU if GPU setup fails
            if not gpu_initialized:
                print("Failed to initialize GPU environment, falling back to CPU")
                self.device = "cpu"
        
        # Choose the GPT model to use
        if high_quality:
            self.text_model = "gpt-4o"
        else:
            self.text_model = "gpt-3.5-turbo"
        
        # Hardware information and diagnostics
        print(f"\n=== SYSTEM CONFIGURATION ===")
        print(f"Text generation: OpenAI {self.text_model}")
        print(f"Device: {self.device}")
        print(f"Memory optimization: {'Enabled' if optimize_memory else 'Disabled'}")
        print(f"Quality setting: {'High' if high_quality else 'Standard'}")
        
        if self.device == "cuda":
            self._print_cuda_diagnostics()
        
        # Initialize models for lazy loading
        self.image_model = None
        self.image_to_image_model = None
        
        # Create necessary directories
        os.makedirs("static/thumbnails", exist_ok=True)
        
        print("\n=== READY TO GENERATE COURSES ===\n")
    
    def _setup_gpu_environment(self):
        """Initialize optimized CUDA environment for stable performance"""
        if self.device != "cuda":
            return False
            
        try:
            # Force garbage collection before GPU operations
            gc.collect()
            torch.cuda.empty_cache()
            
            # Get device properties for adaptive optimization
            device_properties = torch.cuda.get_device_properties(0)
            self.total_memory = device_properties.total_memory / (1024**3)  # GB
            available_memory = self._get_available_vram()
            
            # Configure memory allocation strategy based on available VRAM
            if self.total_memory < 6:  # Low VRAM GPUs (e.g., GTX 1050 Ti, 4GB)
                torch.cuda.set_per_process_memory_fraction(0.85)
                self.max_batch_size = 1
                self.precision = torch.float16
                self.enable_attention_slicing = "max"
                self.use_vae_slicing = True
                # Force deterministic algorithms for more consistent memory usage
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            elif self.total_memory < 12:  # Mid-range GPUs (e.g., RTX 3060, 12GB)
                torch.cuda.set_per_process_memory_fraction(0.9)
                self.max_batch_size = 2
                self.precision = torch.float16 
                self.enable_attention_slicing = True
                self.use_vae_slicing = False
                torch.backends.cudnn.benchmark = True
            else:  # High-end GPUs (RTX 3080+, 16GB+)
                self.max_batch_size = 4
                self.precision = torch.float16
                self.enable_attention_slicing = False
                self.use_vae_slicing = False
                torch.backends.cudnn.benchmark = True
                
            print(f"GPU: {device_properties.name}")
            print(f"VRAM: Total {self.total_memory:.2f} GB, Available: {available_memory:.2f} GB")
            print(f"CUDA Capability: {device_properties.major}.{device_properties.minor}")
            print(f"Configured batch size: {self.max_batch_size}")
            print(f"Precision: {self.precision}")
            print(f"Attention slicing: {self.enable_attention_slicing}")
            
            # Set optimal threading configuration
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            
            # Implement efficient CUDA streams
            self.cuda_stream = torch.cuda.Stream()
            
            return True
        except Exception as e:
            print(f"GPU initialization error: {str(e)}")
            return False
    
    def _print_cuda_diagnostics(self):
        """Print detailed CUDA diagnostics information"""
        try:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Memory usage statistics
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            
            print(f"CUDA Version: {cuda_version}")
            print(f"Device Count: {device_count}")
            print(f"Current Device: {current_device} ({device_name})")
            print(f"Memory Reserved: {reserved:.2f} GB")
            print(f"Memory Allocated: {allocated:.2f} GB")
            
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats()
                print(f"Active Allocations: {stats.get('active_bytes.all.current', 0) / (1024**3):.2f} GB")
        except Exception as e:
            print(f"Error getting CUDA diagnostics: {e}")
    
    def _get_available_vram(self):
        """Get available VRAM in GB with improved accuracy"""
        try:
            if self.device != "cuda":
                return 0
                
            # Force cleanup before checking
            torch.cuda.empty_cache()
            gc.collect()
            
            # Get memory statistics
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free = total - reserved
            
            return free / (1024**3)  # Convert to GB
        except Exception as e:
            print(f"Error checking available VRAM: {e}")
            return 1.0  # Safe default
    
    def _load_image_model(self):
        """Load the image generation model with optimized CUDA configuration"""
        if self.image_model is not None:
            return
                
        print("Loading image generation model with CUDA optimizations...")
        
        # Pre-loading CUDA optimization
        if self.device == "cuda":
            self._setup_gpu_environment()
            torch.cuda.empty_cache()
            gc.collect()
        
        # Select model based on available VRAM and quality settings
        if self.device == "cuda" and hasattr(self, 'total_memory'):
            if self.total_memory < 4.5:  # Extremely limited VRAM
                self.image_model_name = "runwayml/stable-diffusion-v1-5"
                print("Selected SD v1.5 for low VRAM compatibility")
            elif self.total_memory < 6 or not self.high_quality:
                self.image_model_name = "stabilityai/stable-diffusion-2-base"
                print("Selected SD 2.0 base for balanced performance")
            else:
                self.image_model_name = "stabilityai/stable-diffusion-2-1-base"
                print("Selected SD 2.1 for higher quality")
        else:
            # Default for CPU or unknown GPU memory
            self.image_model_name = "runwayml/stable-diffusion-v1-5"
        
        try:
            # Memory efficient pipeline initialization with CUDA optimizations
            with torch.no_grad():
                self.image_model = StableDiffusionPipeline.from_pretrained(
                    self.image_model_name,
                    torch_dtype=self.precision if self.device == "cuda" else torch.float32,
                    revision="fp16" if self.device == "cuda" else "main",
                    safety_checker=None,  # Disable safety checker to save memory
                    requires_safety_checker=False
                )
                
                # Apply specific CUDA optimizations
                if self.device == "cuda":
                    # Memory-efficient attention if needed
                    if hasattr(self, 'enable_attention_slicing') and self.enable_attention_slicing:
                        self.image_model.enable_attention_slicing(self.enable_attention_slicing)
                    
                    # Enable VAE slicing for extremely limited VRAM
                    if hasattr(self, 'use_vae_slicing') and self.use_vae_slicing:
                        if hasattr(self.image_model, "enable_vae_slicing"):
                            self.image_model.enable_vae_slicing()
                            print("VAE slicing enabled for low VRAM operation")
                    
                    # Enable xformers for substantial memory savings if available
                    try:
                        if hasattr(self, 'total_memory') and self.total_memory >= 4:
                            self.image_model.enable_xformers_memory_efficient_attention()
                            print("xformers memory efficient attention enabled")
                    except Exception as xf_error:
                        print(f"xformers optimization unavailable: {xf_error}")
                    
                    # Use the optimal scheduler based on VRAM
                    if self.high_quality and hasattr(self, 'total_memory') and self.total_memory >= 6:
                        self.image_model.scheduler = DPMSolverMultistepScheduler.from_config(
                            self.image_model.scheduler.config,
                            algorithm_type="dpmsolver++",
                            use_karras_sigmas=True
                        )
                        print("Using DPMSolver++ scheduler with Karras sigmas")
                    else:
                        self.image_model.scheduler = EulerDiscreteScheduler.from_config(
                            self.image_model.scheduler.config
                        )
                        print("Using memory-efficient Euler scheduler")
                    
                    # Move to CUDA with optimized CUDA stream
                    if hasattr(self, 'cuda_stream'):
                        with torch.cuda.stream(self.cuda_stream):
                            self.image_model = self.image_model.to(self.device)
                            torch.cuda.synchronize()
                    else:
                        self.image_model = self.image_model.to(self.device)
        
        except Exception as e:
            print(f"Error loading image model with CUDA: {e}")
            self.image_model = None
            
            # Try fallback to CPU if CUDA loading fails
            try:
                print("Attempting fallback to CPU model...")
                self.image_model = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",  # Use smallest model for CPU
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.device = "cpu"  # Force CPU for future operations
            except Exception as cpu_error:
                print(f"CPU fallback also failed: {cpu_error}")
                self.image_model = None
    
    def _unload_model(self, model_name):
        """Unload a model to free GPU memory with enhanced cleanup"""
        if not self.optimize_memory:
            return
            
        if model_name == "image" and self.image_model is not None:
            try:
                print(f"Unloading image model to free GPU memory...")
                
                # Force model to CPU first for safer cleanup
                self.image_model = self.image_model.to("cpu")
                
                # Clear cache first to prevent fragmentation
                torch.cuda.empty_cache()
                
                # Explicitly delete model components
                del self.image_model.vae
                del self.image_model.text_encoder
                del self.image_model.unet
                del self.image_model.scheduler
                
                # Finally delete the model itself
                del self.image_model
                self.image_model = None
                
                # Force garbage collection pass
                gc.collect()
                
                # Second CUDA cache clear after GC
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
                print(f"Available VRAM after unloading: {self._get_available_vram():.2f} GB")
            except Exception as e:
                print(f"Error unloading image model: {e}")
            
        elif model_name == "img2img" and self.image_to_image_model is not None:
            try:
                print(f"Unloading img2img model to free GPU memory...")
                self.image_to_image_model = self.image_to_image_model.to("cpu")
                
                # Clear caches
                torch.cuda.empty_cache()
                
                # Explicit component deletion
                del self.image_to_image_model.vae
                del self.image_to_image_model.text_encoder
                del self.image_to_image_model.unet
                del self.image_to_image_model.scheduler
                
                del self.image_to_image_model
                self.image_to_image_model = None
                
                # Force garbage collection
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error unloading img2img model: {e}")
    
    def _generate_text_openai(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text using OpenAI's API with enhanced anti-hallucination measures.
        
        Args:
            prompt: The instruction/prompt for text generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text string with hallucination filtering
        """
        try:
            # Enhanced anti-hallucination system prompt with content quality controls
            anti_hallucination_prompt = """You are an educational content expert skilled at creating engaging, factual course materials.
            
            CRITICAL CONTENT QUALITY GUIDELINES:
            1. Maintain rigorous factual accuracy and educational integrity at all times
            2. Focus on substance over style - prioritize clear explanations over verbose descriptions
            3. NEVER produce lists of repeated adjectives, repetitive phrases, or filler content
            4. Use precise, technical language appropriate for the subject matter
            5. When providing examples, use realistic, practical scenarios that demonstrate concrete application
            6. If uncertain about a technical detail, acknowledge the limitation rather than inventing information
            7. Structure content with logical progression and clear section organization
            8. Restrict paragraphs to 3-5 focused sentences that develop a single point
            9. For technical topics, include specific methodologies, tools, or frameworks that are actually used in practice
            10. Eliminate redundancies, circular explanations, and content padding
            
            Remember: quality educational content is concise, precise, factual, and intensely focused on student learning outcomes."""
            
            # Rate limiting to avoid OpenAI rate limits
            time.sleep(0.5)  
            
            # Make the API call with improved system prompt and parameters
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": anti_hallucination_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.5,  # Lower temperature reduces randomness
                top_p=0.85,       # More conservative sampling
                frequency_penalty=0.8,  # Higher penalty to reduce repetition
                presence_penalty=0.6    # Higher penalty to improve diversity
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Apply enhanced post-processing to detect and fix hallucinations
            processed_text = self._filter_hallucinations(generated_text)
            
            return processed_text
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Return a basic response in case of API error
            return f"Error generating content. Please check your API key and try again. Details: {str(e)}"

    def _filter_hallucinations(self, text: str) -> str:
        """
        Apply advanced heuristic checks to detect and filter out hallucinated content.
        
        Args:
            text: The generated text to check
            
        Returns:
            Cleaned text with hallucinations removed
        """
        # Check 1: Detect long runs of adjectives or repeated content
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Skip if line has excessive commas (likely a run-on of adjectives)
            if line.count(',') > 10:
                continue
                
            # Skip if line has many repeated words
            words = re.findall(r'\b\w+\b', line.lower())
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Only check meaningful words
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
            # If any word appears too many times, skip this line
            if any(count > 3 for count in word_counts.values()):
                continue
                
            # Check for repeating patterns
            if len(line) > 50:
                word_pairs = [words[i] + ' ' + words[i+1] for i in range(len(words)-1) if i+1 < len(words)]
                pair_counts = {}
                for pair in word_pairs:
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    
                # If any word pair repeats too much, skip this line
                if any(count > 2 for count in pair_counts.values()):
                    continue
            
            filtered_lines.append(line)
        
        # Rejoin the filtered lines
        filtered_text = '\n'.join(filtered_lines)
        
        # Check 2: Detect and fix runaway paragraphs
        paragraphs = filtered_text.split('\n\n')
        fixed_paragraphs = []
        
        for paragraph in paragraphs:
            # If paragraph is excessively long, it may be hallucinated
            if len(paragraph.split()) > 150:  # More than ~150 words in a paragraph is suspicious
                # Try to salvage the first part of the paragraph
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                # Keep only the first few sentences
                salvaged = ' '.join(sentences[:3]) if len(sentences) > 3 else paragraph
                fixed_paragraphs.append(salvaged)
            else:
                fixed_paragraphs.append(paragraph)
        
        # Rejoin the fixed paragraphs
        fixed_text = '\n\n'.join(fixed_paragraphs)
        
        # Check 3: Detect lists of adjectives (common hallucination pattern)
        adjective_pattern = r'(?:[a-z]+ly\s+){3,}|(?:[a-z]+ing\s+){3,}|(?:[a-z]+ed\s+){3,}|(?:[a-z]+ive\s+){3,}'
        matches = re.finditer(adjective_pattern, fixed_text, re.IGNORECASE)
        
        # Replace adjective runs with a simple placeholder
        result_text = fixed_text
        for match in matches:
            result_text = result_text.replace(match.group(0), "effectively ")
        
        # Check 4: Detect repetitive phrases (another common pattern)
        repeated_phrases = re.findall(r'(.{10,30})\1{1,}', result_text)
        for phrase in repeated_phrases:
            result_text = result_text.replace(phrase * 2, phrase)  # Replace repetition with single occurrence
        
        # Check 5: Advanced pattern - extremely long sentences (likely hallucinated)
        sentences = re.split(r'(?<=[.!?])\s+', result_text)
        filtered_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            # Check for unreasonably long sentences (likely hallucinated)
            if len(words) > 60:
                # Take only first 30 words and add ending
                filtered_sentences.append(' '.join(words[:30]) + '.')
            else:
                filtered_sentences.append(sentence)
        
        # Rejoin the filtered sentences
        final_text = ' '.join(filtered_sentences)
        
        # Check 6: Remove "effectively effectively" and similar stutters
        stutter_patterns = [
            (r'\b(\w+)\s+\1\b', r'\1'),  # Repeated words
            (r'effectively\s+effectively', 'effectively'),
            (r'essentially\s+essentially', 'essentially')
        ]
        
        for pattern, replacement in stutter_patterns:
            final_text = re.sub(pattern, replacement, final_text)
            
        return final_text

    def generate_course(self, topic: str, depth: str = "intermediate") -> Dict:
        """
        Generate a single-lesson high-quality course structure based on a topic.
        
        Args:
            topic: The main subject of the course
            depth: The complexity level (beginner, intermediate, advanced)
            
        Returns:
            A dictionary containing the course title, description, module, and lesson
        """
        print(f"Generating high-quality single-lesson course on: {topic} (Level: {depth})")
        
        # Generate course title with improved prompt
        title_prompt = f"""
        Create an engaging, professional, and marketable course title for a {depth} level course on {topic}.
        
        The title should be:
        - Catchy and memorable
        - Professional sounding
        - Clear about the value proposition
        - Under 10 words
        - Specific to {topic}
        - Appealing to students interested in {topic}
        
        Just provide the title with no additional text or explanation.
        """
        
        course_title = self._generate_text_openai(title_prompt, max_tokens=50).strip()
        print(f"Course title: {course_title}")
        
        # Generate course description
        description_prompt = f"""
        Write a compelling course description for a {depth} level course titled "{course_title}" about {topic}.
        
        The description should:
        - Be 3-5 sentences long
        - Clearly explain what students will learn
        - Highlight the value and outcomes of the course
        - Specify who the course is designed for
        - Include a hook that encourages enrollment
        
        Just provide the description with no additional text or explanation.
        """
        
        course_description = self._generate_text_openai(description_prompt, max_tokens=250).strip()
        
        # Generate detailed course content for a single comprehensive lesson
        module_title = f"Complete {topic} Masterclass"
        lesson_title = f"Comprehensive Guide to {topic}"
        
        # Get learning objectives for better course quality
        objectives_prompt = f"""
        Create 5-7 specific learning objectives for the course "{course_title}" on {topic}.
        
        Each learning objective should:
        - Start with an action verb (e.g., Analyze, Create, Implement)
        - Be specific and measurable
        - Relate directly to {topic}
        - Be appropriate for a {depth} level course
        
        Format as a simple bullet point list with no additional text or explanation.
        """
        
        learning_objectives = self._generate_text_openai(objectives_prompt, max_tokens=400).strip()
        
        # Create a thumbnail for the course
        thumbnail_path = self.generate_thumbnail(course_title, topic, [{"title": module_title, "lessons": [lesson_title]}])
        
        # Structure the response in the same format as before
        modules = [{
            "title": module_title,
            "lessons": [lesson_title]
        }]
        
        # Generate the detailed content for the single lesson immediately
        print("Generating detailed content for the comprehensive lesson...")
        
        lesson_prompt = f"""
        Create extremely detailed and comprehensive educational content for a single-lesson masterclass titled "{lesson_title}" about {topic} at a {depth} level.
        
        This needs to be your absolute highest quality work as this will be the only lesson in the entire course. Include everything a student would need to know about {topic}.
        
        Structure the lesson content as follows:
        
        1. INTRODUCTION (2-3 paragraphs):
        A thorough introduction to {topic}, explaining its importance, applications, and relevance.
        
        2. LEARNING OBJECTIVES (5-7 points):
        Very specific, measurable objectives that students will achieve by the end of this comprehensive lesson.
        
        3. CORE CONCEPTS (3-5 sections):
        Detailed explanation of the fundamental concepts of {topic}, with clear definitions, examples, and context.
        
        4. DETAILED METHODOLOGY (5+ subsections):
        Step-by-step breakdown of methodologies, processes, or techniques related to {topic}.
        Include practical examples, case studies, and real-world applications.
        
        5. ADVANCED CONCEPTS (3-5 sections):
        Exploration of more complex aspects of {topic} appropriate for {depth} level students.
        Include detailed explanations, diagrams descriptions, formulas if applicable, and cutting-edge developments.
        
        6. PRACTICAL IMPLEMENTATION (2-3 detailed exercises):
        Comprehensive exercises that allow students to apply what they've learned.
        Include step-by-step instructions, expected outcomes, and troubleshooting guidance.
        
        7. COMMON CHALLENGES AND SOLUTIONS (3-5 challenges):
        Address typical obstacles students might face when working with {topic}.
        Provide detailed, actionable solutions for each challenge.
        
        8. INDUSTRY APPLICATIONS (2-3 examples):
        Real-world examples of how {topic} is applied in professional settings.
        Include specific use cases, outcomes, and implementation details.
        
        9. RESOURCES AND TOOLS (5-10 resources):
        Curated list of valuable tools, resources, books, websites, and references for further exploration.
        
        Format the content with professional Markdown formatting including headers (# for main sections, ## for subsections), bullet points, numbered lists, *emphasis* for important terms, and code blocks if needed.
        
        CRITICAL GUIDELINES TO ENSURE HIGH-QUALITY CONTENT:
        - Focus on technical accuracy and educational value in every section
        - For each section, provide concrete, specific information rather than vague generalities
        - Use precise terminology and concepts specific to {topic}
        - Avoid repetitive phrases, circular explanations, or content padding
        - Support claims with evidence, examples, or recognized methodologies
        - Structure content with logical progression from foundational to advanced concepts
        - For technical topics, include actual techniques, algorithms, or systems used in practice
        - Ensure explanations are clear without oversimplification
        """
        
        lesson_content = self._generate_text_openai(lesson_prompt, max_tokens=4000)
        
        # Generate assessment separately with specialized prompt
        print("Generating assessment questions...")
        assessment_content = self._generate_assessment(topic, course_title, depth)
        
        # Create the module content structure
        module_content = {
            "title": module_title,
            "introduction": f"This comprehensive masterclass covers everything you need to know about {topic}.",
            "lessons": [{
                "title": lesson_title,
                "content": lesson_content,
                "assessment": assessment_content
            }]
        }
        
        # Create module_contents dictionary
        module_contents = {"0": module_content}
        
        return {
            "title": course_title,
            "description": course_description,
            "level": depth,
            "topic": topic,
            "learning_objectives": learning_objectives,
            "modules": modules,
            "thumbnail_path": thumbnail_path,
            "module_contents": module_contents  # Include the content directly
        }
    
    def _generate_assessment(self, topic: str, title: str, depth: str) -> str:
        """
        Generate high-quality assessment questions for a course with improved rigor.
        
        Args:
            topic: The course topic
            title: The course title
            depth: The complexity level
            
        Returns:
            Assessment content with multiple choice and other question types
        """
        assessment_prompt = f"""
        Create a comprehensive assessment for the course "{title}" on {topic} at a {depth} level.
        
        Include exactly:
        1. 6 multiple-choice questions with 4 options each
        2. 2 scenario-based questions that test application of knowledge
        
        For multiple-choice questions:
        - Clearly mark correct answers with [x] and incorrect answers with [ ]
        - Provide a brief explanation for each correct answer
        - Cover different aspects of {topic}
        - Ensure questions test understanding, not just recall
        - Make questions technically precise and unambiguous
        
        For scenario-based questions:
        - Present a realistic scenario related to {topic}
        - Ask the student to solve a problem or make a recommendation
        - Provide evaluation criteria or key points that should be included in a good answer
        - Make scenarios technically accurate and plausible
        
        Format in Markdown with clear headers and proper formatting.
        
        IMPORTANT: 
        - Focus on testing practical application of knowledge
        - Questions should be challenging but fair for {depth} level students
        - Use precise technical language appropriate to the field
        - Ensure all scenarios represent authentic situations professionals would encounter
        """
        
        # Generate assessment with anti-hallucination measures
        assessment_content = self._generate_text_openai(assessment_prompt, max_tokens=2000)
        
        return assessment_content

    def generate_thumbnail(self, title: str, topic: str, modules: List[Dict]) -> str:
        """
        Generate a high-quality thumbnail image for the course with CUDA optimization.
        
        Args:
            title: The course title
            topic: The main subject of the course
            modules: List of module dictionaries to extract keywords
            
        Returns:
            Path to the saved thumbnail image
        """
        print("Generating course thumbnail with CUDA acceleration...")
        
        # Set the thumbnail directory - using relative path
        thumbnail_dir = "static/thumbnails"
        
        # Ensure the thumbnails directory exists
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # Create a unique filename based on topic and timestamp to avoid conflicts
        safe_filename = re.sub(r'[^\w\s-]', '', topic.lower())
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename).strip('-')
        timestamp = int(time.time())
        filename = f"{thumbnail_dir}/{safe_filename}-{timestamp}.png"
        
        # Create default thumbnail path
        default_thumbnail = f"{thumbnail_dir}/default-thumbnail.png"
        
        # If default thumbnail doesn't exist, create a simple one
        if not os.path.exists(default_thumbnail):
            try:
                # Create a simple default thumbnail
                placeholder = Image.new('RGB', (800, 600), color=(53, 92, 125))
                draw = ImageDraw.Draw(placeholder)
                
                # Add some text
                draw.text((400, 300), "Default Course Thumbnail", fill=(255, 255, 255), anchor="mm")
                
                # Save the default thumbnail
                placeholder.save(default_thumbnail)
                print(f"Created default thumbnail at {default_thumbnail}")
            except Exception as e:
                print(f"Could not create default thumbnail: {e}")
        
        try:
            # Load the image model if not already loaded
            self._load_image_model()
            
            # Check if model loading failed
            if self.image_model is None:
                print("Image model loading failed, using default thumbnail")
                return default_thumbnail
            
            # Extract keywords from modules for better image prompting
            keywords = set()
            for module in modules:
                words = re.findall(r'\b\w{4,}\b', module["title"].lower())
                keywords.update(words)
            
            # Filter out common stop words and limit to important keywords
            stop_words = {"introduction", "advanced", "basic", "understanding", "fundamentals"}
            keywords = [word for word in keywords if word not in stop_words]
            keywords = " ".join(keywords[:5])  # Use top 5 keywords
            
            # Use OpenAI to create a better image prompt
            image_prompt_request = f"""
            Create a detailed prompt for a text-to-image AI to generate a professional, visually appealing
            educational thumbnail for a course titled "{title}" about {topic}.
            
            The prompt should:
            - Include specific visual elements related to {topic}
            - Describe an illustration or scene that represents the course content
            - Specify a professional, modern design style with vibrant colors
            - Include artistic direction (lighting, color scheme, composition)
            - Be optimized for a Stable Diffusion model
            - Include adjectives like "8k", "detailed", "professional", "educational"
            
            Write just the prompt itself with no additional text or explanation.
            Limit to 75 words maximum.
            """
            
            # Get enhanced image prompt from OpenAI
            enhanced_prompt = self._generate_text_openai(image_prompt_request, max_tokens=150)
            print(f"Using image prompt: {enhanced_prompt}")
            
            # Adaptive parameters based on available VRAM
            if self.device == "cuda":
                available_vram = self._get_available_vram()
                
                # Adjust generation parameters based on available VRAM
                if available_vram > 3.5:  # More than 3.5GB free
                    width, height = 512, 384
                    steps = 25
                    guidance = 7.5
                elif available_vram > 2.0:  # 2-3GB free
                    width, height = 448, 336
                    steps = 20
                    guidance = 7.0
                else:  # Less than 2GB free
                    width, height = 384, 288
                    steps = 18
                    guidance = 7.0
                
                print(f"Thumbnail generation using {available_vram:.2f}GB VRAM")
                print(f"Parameters: {width}x{height}, {steps} steps")
            else:
                # CPU fallback parameters
                width, height = 384, 288
                steps = 15
                guidance = 7.0
            
            # Make sure to free any unused CUDA memory before generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Run generation with performance tracing
            generation_start = time.time()
            
            # Handle CUDA out-of-memory issues with progressive fallbacks
            for attempt in range(3):
                try:
                    # On retries, use more conservative parameters
                    if attempt > 0:
                        width = int(width * 0.8)
                        height = int(height * 0.8)
                        steps = max(12, steps - 5)
                        print(f"Retry {attempt} with reduced parameters: {width}x{height}, {steps} steps")
                    
                    # Use no_grad context to save memory
                    with torch.no_grad():
                        image = self.image_model(
                            prompt=f"{enhanced_prompt}, professional education thumbnail, course cover, clean design, high quality, detailed, vibrant",
                            negative_prompt="text, watermark, signature, blurry, distorted, low quality, grainy, pixelated, amateur, ugly",
                            height=height,
                            width=width,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                        ).images[0]
                    
                    # Add text overlay with course title and brief outline
                    image = self._add_text_overlay(image, title, topic, modules)
                    
                    # Save the image with high quality settings
                    image.save(filename, quality=95)
                    generation_time = time.time() - generation_start
                    print(f"Thumbnail generated in {generation_time:.2f} seconds")
                    
                    # Important: move the model back to CPU and free CUDA memory
                    if self.device == "cuda" and self.optimize_memory:
                        self._unload_model("image")
                        
                    return filename
                
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if "CUDA out of memory" in str(e) and attempt < 2:
                        print(f"CUDA OOM on attempt {attempt+1}, reducing parameters and retrying...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(1)  # Give system time to recover
                    else:
                        print(f"Error in CUDA-accelerated image generation: {e}")
                        break
                
                except Exception as e:
                    print(f"Error in image generation: {e}")
                    break
            
            # If all generation attempts failed, try to create a basic placeholder
            return self._create_fallback_thumbnail(title, topic)
            
        except Exception as e:
            print(f"Error in thumbnail generation: {e}")
            return self._create_fallback_thumbnail(title, topic)
    
    def _create_fallback_thumbnail(self, title: str, topic: str) -> str:
        """Create a basic fallback thumbnail when image generation fails"""
        print("Creating fallback placeholder image...")
        
        try:
            # Create a unique filename
            thumbnail_dir = "static/thumbnails"
            timestamp = int(time.time())
            safe_topic = re.sub(r'[^\w\s-]', '', topic.lower())
            safe_topic = re.sub(r'[-\s]+', '-', safe_topic).strip('-')
            filename = f"{thumbnail_dir}/fallback-{safe_topic}-{timestamp}.png"
            
            # Create a gradient background
            width, height = 800, 600
            placeholder = Image.new('RGB', (width, height), color=(53, 92, 125))
            
            # Create gradient overlay
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            # Add gradient from top to bottom
            for i in range(height):
                alpha = int(150 * i / height)
                draw_overlay.line([(0, i), (width, i)], fill=(0, 0, 0, alpha))
            
            # Composite the overlay onto the background
            placeholder = Image.alpha_composite(placeholder.convert('RGBA'), overlay).convert('RGB')
            
            # Add text
            draw = ImageDraw.Draw(placeholder)
            
            # Find available fonts
            title_font = None
            try:
                # Try standard system fonts
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                    "/Library/Fonts/Arial Bold.ttf",  # macOS
                    "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows
                ]
                
                for path in font_paths:
                    if os.path.exists(path):
                        title_font = ImageFont.truetype(path, 40)
                        subtitle_font = ImageFont.truetype(path, 30)
                        break
            except Exception as font_error:
                print(f"Font error: {font_error}, using default")
                
            # Fall back to default font if needed
            if title_font is None:
                title_font = ImageFont.load_default()
                subtitle_font = title_font
            
            # Draw text centered (using estimated centering if anchor not supported)
            title_x = width // 2
            title_y = height // 2 - 50
            topic_y = height // 2 + 50
            
            # Draw text with shadow effect
            draw.text((title_x+2, title_y+2), title, fill=(0, 0, 0, 200), font=title_font, anchor="mm")
            draw.text((title_x, title_y), title, fill=(255, 255, 255), font=title_font, anchor="mm")
            
            draw.text((title_x+2, topic_y+2), f"A course about {topic}", fill=(0, 0, 0, 200), font=subtitle_font, anchor="mm")
            draw.text((title_x, topic_y), f"A course about {topic}", fill=(200, 200, 255), font=subtitle_font, anchor="mm")
            
            # Save the image
            placeholder.save(filename, quality=90)
            print(f"Fallback thumbnail saved to {filename}")
            
            return filename
        except Exception as e:
            print(f"Error creating fallback thumbnail: {e}")
            # Return path to default thumbnail
            return "static/thumbnails/default-thumbnail.png"
    
    def _add_text_overlay(self, image: Image.Image, title: str, topic: str, modules: List[Dict]) -> Image.Image:
        """
        Add high-quality text overlay to the thumbnail with course title and brief outline.
        
        Args:
            image: PIL Image object
            title: Course title
            topic: Course topic
            modules: List of module dictionaries
            
        Returns:
            PIL Image with text overlay
        """
        try:
            # Create a copy of the image to work with
            result_image = image.copy()
            width, height = result_image.size
            
            # Create semi-transparent overlay for better text visibility
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            # Create gradient overlay from bottom to middle
            for i in range(height // 2, height):
                # Calculate alpha (more transparent at top, more opaque at bottom)
                alpha = int(220 * (i - height // 2) / (height - height // 2))
                draw_overlay.line([(0, i), (width, i)], fill=(0, 0, 0, alpha))
            
            # Add a subtle top bar for the course topic
            draw_overlay.rectangle(
                [(0, 0), (width, 60)],
                fill=(0, 0, 0, 180)
            )
            
            # Try to use nice fonts if available
            title_font_size = int(width / 18)  # Adaptive font size based on image width
            subtitle_font_size = int(width / 30)
            topic_font_size = int(width / 40)
            
            # Set defaults for fonts
            title_font = None
            subtitle_font = None
            topic_font = None
            
            try:
                # Try to find a good system font 
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                    "/Library/Fonts/Arial Bold.ttf",  # macOS
                    "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows
                ]
                
                for path in font_paths:
                    if os.path.exists(path):
                        title_font = ImageFont.truetype(path, title_font_size)
                        subtitle_font = ImageFont.truetype(path, subtitle_font_size)
                        topic_font = ImageFont.truetype(path, topic_font_size)
                        break
            except Exception as font_error:
                print(f"Font loading error: {font_error}")
            
            # Fall back to default font if needed
            if title_font is None:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                topic_font = ImageFont.load_default()
            
            # Simplified text drawing without anchor (for compatibility)
            # Draw topic text at the top
            topic_text = f"{topic.upper()}"
            try:
                # Modern Pillow version
                topic_w = title_font.getlength(topic_text)
            except AttributeError:
                try:
                    # Older Pillow version
                    topic_w, _ = title_font.getsize(topic_text)
                except Exception:
                    # Fallback estimate
                    topic_w = len(topic_text) * topic_font_size * 0.6
                
            topic_x = (width - topic_w) // 2
            topic_y = 30 - (topic_font_size // 2)  # Adjust for vertical centering
            draw_overlay.text(
                (topic_x, topic_y),
                topic_text,
                fill=(255, 255, 255, 230),
                font=topic_font
            )
            
            # Draw title
            title_text = title
            try:
                # Modern Pillow version
                title_w = title_font.getlength(title_text)
            except AttributeError:
                try:
                    # Older Pillow version
                    title_w, _ = title_font.getsize(title_text)
                except Exception:
                    # Fallback estimate
                    title_w = len(title_text) * title_font_size * 0.6
                
            title_x = (width - title_w) // 2
            title_y = height - 140
            
            # Draw text shadow
            draw_overlay.text(
                (title_x + 2, title_y + 2),
                title_text,
                fill=(0, 0, 0, 200),
                font=title_font
            )
            
            # Draw main title text
            draw_overlay.text(
                (title_x, title_y),
                title_text,
                fill=(255, 255, 255, 255),
                font=title_font
            )
            
            # Draw subtitle
            subtitle_text = "Comprehensive single-lesson masterclass"
            try:
                # Modern Pillow version
                subtitle_w = subtitle_font.getlength(subtitle_text)
            except AttributeError:
                try:
                    # Older Pillow version
                    subtitle_w, _ = subtitle_font.getsize(subtitle_text)
                except Exception:
                    # Fallback estimate
                    subtitle_w = len(subtitle_text) * subtitle_font_size * 0.6
                
            subtitle_x = (width - subtitle_w) // 2
            subtitle_y = title_y + title_font_size + 10
            
            draw_overlay.text(
                (subtitle_x, subtitle_y),
                subtitle_text,
                fill=(220, 220, 220, 230),
                font=subtitle_font
            )
            
            # Apply the overlay to the original image
            result_image = Image.alpha_composite(result_image.convert("RGBA"), overlay)
            
            # Enhance the image slightly for better visual appeal
            enhancer = ImageEnhance.Contrast(result_image.convert("RGB"))
            result_image = enhancer.enhance(1.1)
            
            # Add subtle vignette effect
            result_image = self._add_vignette(result_image)
            
            # Convert RGBA to RGB before returning
            return result_image.convert("RGB")
            
        except Exception as overlay_error:
            print(f"Error adding text overlay: {overlay_error}")
            # Return the original image if overlay fails
            return image.convert("RGB") if image.mode == "RGBA" else image
    
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
            result.putalpha(mask)
            
            # Convert back to RGB after applying the vignette
            return result.convert('RGB')
            
        except Exception as vignette_error:
            print(f"Error adding vignette: {vignette_error}")
            # Return the original image if vignette fails
            return image
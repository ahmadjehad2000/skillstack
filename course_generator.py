# course_generator.py
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
    robust anti-hallucination measures and quality control.
    
    This class generates comprehensive educational courses with high-quality
    content and custom thumbnails, preventing common hallucination issues.
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
    
    def _load_image_model(self):
        """Load the image generation model optimized for 1050 Ti's 4GB VRAM"""
        if self.image_model is not None:
            return
        
        print("Loading image generation model...")
        
        # Free any unused memory before loading model
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Choose an optimized model for 1050 Ti (4GB VRAM)
        # Use smaller base model that can fit in limited VRAM
        if self.high_quality:
            self.image_model_name = "stabilityai/stable-diffusion-2-1-base"
        else:
            self.image_model_name = "runwayml/stable-diffusion-v1-5"  # Even smaller memory footprint
        
        try:
            # Memory efficient loading
            self.image_model = StableDiffusionPipeline.from_pretrained(
                self.image_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                revision="fp16" if self.device == "cuda" else "main",
                safety_checker=None,  # Disable safety checker to save memory
            )
            
            # Enable memory efficient settings for low VRAM GPUs
            if hasattr(self.image_model, "enable_attention_slicing"):
                self.image_model.enable_attention_slicing("max")
            
            if hasattr(self.image_model, "enable_xformers_memory_efficient_attention"):
                try:
                    self.image_model.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available, using standard attention")
            
            # Use efficient scheduler
            if self.high_quality:
                self.image_model.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.image_model.scheduler.config,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=True
                )
            else:
                self.image_model.scheduler = EulerDiscreteScheduler.from_config(
                    self.image_model.scheduler.config
                )
            
            if self.device == "cuda":
                self.image_model = self.image_model.to(self.device)
                
        except Exception as e:
            print(f"Error loading image model: {e}")
            self.image_model = None
    
    def _load_image_to_image_model(self):
        """Load the img2img model for refinement"""
        if self.image_to_image_model is not None:
            return
            
        print("Loading image refinement model...")
        
        # Free memory first
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Load img2img pipeline for refinement
            self.image_to_image_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.image_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                revision="fp16" if self.device == "cuda" else "main",
                safety_checker=None,
            )
            
            if hasattr(self.image_to_image_model, "enable_attention_slicing"):
                self.image_to_image_model.enable_attention_slicing("max")
            
            if hasattr(self.image_to_image_model, "enable_xformers_memory_efficient_attention"):
                try:
                    self.image_to_image_model.enable_xformers_memory_efficient_attention()
                except:
                    print("xformers not available for img2img")
                
            if self.device == "cuda":
                self.image_to_image_model = self.image_to_image_model.to(self.device)
        
        except Exception as e:
            print(f"Error loading img2img model: {e}")
            self.image_to_image_model = None
    
    def _unload_model(self, model_name):
        """Unload a model to free GPU memory"""
        if not self.optimize_memory:
            return
            
        if model_name == "image" and self.image_model is not None:
            try:
                self.image_model = self.image_model.to("cpu")
                del self.image_model
                self.image_model = None
            except Exception as e:
                print(f"Error unloading image model: {e}")
            
        elif model_name == "img2img" and self.image_to_image_model is not None:
            try:
                self.image_to_image_model = self.image_to_image_model.to("cpu")
                del self.image_to_image_model
                self.image_to_image_model = None
            except Exception as e:
                print(f"Error unloading img2img model: {e}")
        
        # Force garbage collection
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def _generate_text_openai(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text using OpenAI's API with anti-hallucination measures.
        
        Args:
            prompt: The instruction/prompt for text generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text string with hallucination filtering
        """
        try:
            # Add anti-hallucination system prompt
            anti_hallucination_prompt = """You are an educational content expert skilled at creating engaging, factual course materials.
            
            IMPORTANT GUIDELINES:
            1. Be concise and focused - quality over quantity
            2. Stick strictly to verifiable facts and standard practices
            3. NEVER generate lists of adjectives or repetitive content
            4. Use simple, clear language - avoid flowery or excessive descriptions
            5. If you're unsure about something, acknowledge the limitation rather than making it up
            6. Keep your responses structured and organized
            7. Use bullet points and numbered lists for clarity where appropriate
            8. STOP writing when you've addressed the topic completely
            9. Each paragraph should contain no more than 3-5 sentences
            10. For industry examples and applications, focus on 2-3 clear, specific points rather than general statements
            
            Remember your goal is to create educational content that is factual, practical, and focused."""
            
            # Rate limiting to avoid OpenAI rate limits
            time.sleep(0.5)  
            
            # Make the API call with improved system prompt
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
            
            # Apply post-processing to detect and fix hallucinations
            processed_text = self._filter_hallucinations(generated_text)
            
            return processed_text
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Return a basic response in case of API error
            return f"Error generating content. Please check your API key and try again. Details: {str(e)}"

    def _filter_hallucinations(self, text: str) -> str:
        """
        Apply heuristic checks to detect and filter out hallucinated content.
        
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
        
        return result_text

    def _enhance_industry_applications_prompt(self, topic: str) -> str:
        """
        Create a specialized prompt for industry applications to prevent hallucinations.
        
        Args:
            topic: The main subject of the course
            
        Returns:
            Enhanced prompt for industry applications section
        """
        prompt = f"""
        Provide 2-3 specific, real-world industry applications of {topic}. 

        For each example:
        1. Name a specific industry or sector
        2. Describe 3-4 concrete ways {topic} is used in this industry
        3. Include specific tools, processes, or measurable outcomes where relevant
        
        Format each example with bullet points for clarity.
        
        IMPORTANT GUIDELINES:
        - Be extremely specific and practical
        - Avoid general statements about benefits
        - Use plain, direct language - avoid adjective strings and flowery descriptions
        - Maximum 4-5 sentences per example
        - Focus on current, practical applications rather than future possibilities
        - Stop after providing the 2-3 examples
        """
        
        return prompt

    def _generate_assessment(self, topic: str, title: str, depth: str) -> str:
        """
        Generate high-quality assessment questions for a course.
        
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
        
        For scenario-based questions:
        - Present a realistic scenario related to {topic}
        - Ask the student to solve a problem or make a recommendation
        - Provide evaluation criteria or key points that should be included in a good answer
        
        Format in Markdown with clear headers and proper formatting.
        
        IMPORTANT: 
        - Be specific and practical - no vague or overly theoretical questions
        - Questions should be challenging but fair for {depth} level students
        - Keep language simple and direct
        """
        
        # Generate assessment with anti-hallucination measures
        assessment_content = self._generate_text_openai(assessment_prompt, max_tokens=2000)
        
        return assessment_content

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
        
        IMPORTANT GUIDELINES TO PREVENT CONTENT ISSUES:

        - Avoid repetitive language or listing excessive adjectives
        - For each section, focus on 3-5 clear, distinct points rather than padding with filler
        - Use professional, educational language - avoid marketing-style or flowery language
        - Keep sections concise and focused on practical information
        - Use simple, direct language rather than complex or verbose descriptions
        - Limit each section to a reasonable length - quality over quantity
        - Structure content with clear headings and bullet points
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
        module_content = {
            "title": module_title,
            "lessons": []
        }
        
        # First, generate an introduction for the entire module
        module_intro_prompt = f"""
        Write a comprehensive introduction for the module "{module_title}" which is part of the course "{course_title}".
        
        This introduction should:
        - Explain the purpose and importance of this module within the larger course context
        - Outline what students will learn in this module specifically
        - Connect concepts from previous modules if applicable
        - Set expectations for the difficulty and time commitment
        - Motivate students by explaining the value of mastering these concepts
        
        Write 2-3 paragraphs (250-350 words total) with no additional text or explanation.

        IMPORTANT: Keep language concise and factual, avoid excessive adjectives or repetitive phrases.
        """
        
        module_intro = self._generate_text_openai(module_intro_prompt, max_tokens=500).strip()
        module_content["introduction"] = module_intro
        
        # Generate content for each lesson
        for lesson in lessons:
            print(f"Generating content for lesson: {lesson}")
            
            lesson_prompt = f"""
            Create detailed educational content for the lesson titled "{lesson}" which is part of the module "{module_title}" in the course "{course_title}".
            
            Structure the lesson content as follows:
            
            1. INTRODUCTION:
            A brief paragraph introducing the topic and explaining its importance.
            
            2. LEARNING OBJECTIVES:
            3-4 specific, measurable objectives that students will achieve by the end of this lesson.
            
            3. MAIN CONTENT:
            The core educational material, divided into logical sections with clear headings.
            Include relevant examples, code snippets if applicable, diagrams descriptions, and explanations.
            
            4. PRACTICAL EXERCISE:
            A hands-on activity or assignment that reinforces the lesson concepts.
            
            5. KEY TAKEAWAYS:
            A bullet point summary of the most important concepts covered.
            
            Format the content with appropriate Markdown formatting including headers, bullet points, code blocks if needed, and emphasis where appropriate.
            
            IMPORTANT GUIDELINES:
            - Use clear, factual, and concise language
            - Avoid long chains of adjectives or flowery language
            - Focus on practical, applicable knowledge
            - Provide specific examples, not generalities
            - Keep explanations structured and organized
            """
            
            content = self._generate_text_openai(lesson_prompt, max_tokens=1200)
            
            # Generate quiz questions for assessment
            quiz_prompt = f"""
            Create 3-5 assessment questions for the lesson "{lesson}" in the course "{course_title}".
            
            Include these types of questions:
            - 2-3 multiple-choice questions with 4 options each, clearly marking correct answers with [x]
            - 1-2 practical application questions that test deeper understanding
            
            For each question:
            1. Write a clear, concise question that tests understanding of key concepts
            2. For multiple choice: provide options marked with [x] for correct and [ ] for incorrect
            3. Include a brief explanation of why the correct answer is right
            
            Format each question with proper Markdown.
            
            IMPORTANT:
            - Focus on testing meaningful understanding, not trivial details
            - Questions should be clearly written and unambiguous
            - Avoid overly complex language or unnecessary jargon
            """
            
            quiz_questions = self._generate_text_openai(quiz_prompt, max_tokens=600)
            
            module_content["lessons"].append({
                "title": lesson,
                "content": content,
                "assessment": quiz_questions
            })
        
        return module_content
    
    def generate_thumbnail(self, title: str, topic: str, modules: List[Dict]) -> str:
        """
        Generate a high-quality thumbnail image for the course with better error handling.
        
        Args:
            title: The course title
            topic: The main subject of the course
            modules: List of module dictionaries to extract keywords
            
        Returns:
            Path to the saved thumbnail image
        """
        print("Generating course thumbnail...")
        
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
            
            # Generate base image using Stable Diffusion with optimized settings
            # Using smaller resolution for initial generation to fit in VRAM
            initial_width = 512 if self.high_quality else 384
            initial_height = 384 if self.high_quality else 256
            
            # Set safe defaults in case generation fails
            generation_steps = 20  # Reduced steps to avoid memory issues
            guidance_scale = 7.0   # Standard guidance scale
            
            print(f"Generating initial image with dimensions {initial_width}x{initial_height}...")
            
            # Make sure to free any unused CUDA memory before generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Wrap the image generation in a try-except block to handle potential issues
            try:
                image = self.image_model(
                    prompt=f"{enhanced_prompt}, professional education thumbnail, course cover, clean design, high quality, 8k, detailed, vibrant",
                    negative_prompt="text, watermark, signature, blurry, distorted, low quality, grainy, pixelated, amateur, ugly",
                    height=initial_height,
                    width=initial_width,
                    num_inference_steps=generation_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
                
                # Skip the img2img refinement step to avoid memory issues
                # This will make generation more reliable, especially on GPUs with limited VRAM
                
                # Add text overlay with course title and brief outline
                image = self._add_text_overlay(image, title, topic, modules)
                
                # Save the image with high quality settings
                image.save(filename, quality=95)
                print(f"Thumbnail saved to {filename}")
                
                return filename
                
            except Exception as e:
                print(f"Error in image generation: {e}")
                
                # Try with further reduced parameters if the first attempt failed
                try:
                    print("Retrying with reduced parameters...")
                    
                    # Further reduce dimensions and steps
                    smaller_width = 320
                    smaller_height = 240
                    fewer_steps = 15
                    
                    # Clear memory again
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Try again with smaller parameters
                    image = self.image_model(
                        prompt=f"{topic} course, educational, professional",
                        negative_prompt="text, blurry, low quality",
                        height=smaller_height,
                        width=smaller_width,
                        num_inference_steps=fewer_steps,
                        guidance_scale=7.0,
                    ).images[0]
                    
                    # Add text overlay
                    image = self._add_text_overlay(image, title, topic, modules)
                    image.save(filename, quality=90)
                    print(f"Thumbnail saved with reduced parameters to {filename}")
                    
                    return filename
                    
                except Exception as backup_error:
                    print(f"Backup generation also failed: {backup_error}")
                    
                    # Create a basic placeholder image with text as final fallback
                    try:
                        print("Creating fallback placeholder image...")
                        placeholder = Image.new('RGB', (800, 600), color=(53, 92, 125))
                        draw = ImageDraw.Draw(placeholder)
                        
                        # Try to use a system font or fallback to default
                        try:
                            # Look for fonts in Windows fonts directory
                            for font_name in ["Arial.ttf", "Calibri.ttf", "Segoe UI.ttf"]:
                                font_path = os.path.join("C:\\Windows\\Fonts", font_name)
                                if os.path.exists(font_path):
                                    title_font = ImageFont.truetype(font_path, 40)
                                    topic_font = ImageFont.truetype(font_path, 30)
                                    break
                            else:
                                # If no system fonts found
                                title_font = ImageFont.load_default()
                                topic_font = ImageFont.load_default()
                        except:
                            title_font = ImageFont.load_default()
                            topic_font = ImageFont.load_default()
                        
                        # Add text to the placeholder - using simpler approach for compatibility
                        w, h = placeholder.size
                        draw.text((w//2, h//2 - 50), title, fill=(255, 255, 255), font=title_font)
                        draw.text((w//2, h//2 + 50), f"A course about {topic}", fill=(200, 200, 200), font=topic_font)
                        
                        # Save the placeholder
                        placeholder.save(filename)
                        print(f"Placeholder image saved to {filename}")
                        
                        return filename
                        
                    except Exception as placeholder_error:
                        print(f"Even placeholder creation failed: {placeholder_error}")
                        # Use the default thumbnail we created at the beginning
                        return default_thumbnail
                    
        except Exception as outer_error:
            print(f"Outer thumbnail generation error: {outer_error}")
            return default_thumbnail
    
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
                # Try to find a good system font (Windows specific)
                font_path = None
                for font_name in ["Arial.ttf", "Calibri.ttf", "Segoe UI.ttf"]:
                    potential_path = os.path.join("C:\\Windows\\Fonts", font_name)
                    if os.path.exists(potential_path):
                        font_path = potential_path
                        break
                
                if font_path:
                    title_font = ImageFont.truetype(font_path, title_font_size)
                    subtitle_font = ImageFont.truetype(font_path, subtitle_font_size)
                    topic_font = ImageFont.truetype(font_path, topic_font_size)
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
                topic_w = title_font.getlength(topic_text)
            except:
                topic_w = len(topic_text) * topic_font_size * 0.6  # Estimate width if getlength not available
                
            topic_x = (width - topic_w) // 2
            topic_y = 30
            draw_overlay.text(
                (topic_x, topic_y),
                topic_text,
                fill=(255, 255, 255, 230),
                font=topic_font
            )
            
            # Draw title
            title_text = title
            try:
                title_w = title_font.getlength(title_text)
            except:
                title_w = len(title_text) * title_font_size * 0.6  # Estimate width
                
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
                subtitle_w = subtitle_font.getlength(subtitle_text)
            except:
                subtitle_w = len(subtitle_text) * subtitle_font_size * 0.6  # Estimate width
                
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
    
    def _parse_outline(self, outline: str) -> List[Dict]:
        """
        Parse the generated course outline into structured data.
        
        Args:
            outline: Text outline of the course
            
        Returns:
            List of dictionaries for each module and its lessons
        """
        modules = []
        current_module = None
        
        for line in outline.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Match module lines
            module_match = re.match(r'^Module\s+\d+:?\s+(.*)', line, re.IGNORECASE)
            if module_match:
                if current_module:
                    modules.append(current_module)
                
                module_title = module_match.group(1).strip()
                current_module = {
                    "title": module_title,
                    "lessons": []
                }
                continue
                
            # Match lesson lines
            lesson_match = re.match(r'^[-•*]\s+(?:Lesson\s+\d+\.\d+:?\s+)?(.*)', line)
            if lesson_match and current_module:
                lesson_title = lesson_match.group(1).strip()
                current_module["lessons"].append(lesson_title)
                
        # Add the last module
        if current_module:
            modules.append(current_module)
            
        return modules
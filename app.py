# app.py
import os
import re
import time
import json
import uuid
import secrets
import shutil
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
from flask_session import Session
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import openai
import torch
from threading import Thread
from queue import Queue
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the CourseGenerator class
from course_generator import CourseGenerator

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Set paths (use relative paths to avoid cross-drive issues)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['THUMBNAIL_FOLDER'] = os.getenv('THUMBNAIL_FOLDER', 'static/thumbnails')
app.config['COURSE_FOLDER'] = os.getenv('COURSE_FOLDER', 'static/courses')
app.config['MODULE_FOLDER'] = os.getenv('MODULE_FOLDER', 'static/modules')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['THUMBNAIL_FOLDER'], 
               app.config['COURSE_FOLDER'], app.config['MODULE_FOLDER'], 'flask_session']:
    os.makedirs(folder, exist_ok=True)

# Create default thumbnail if it doesn't exist
default_thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], 'default-thumbnail.png')
if not os.path.exists(default_thumbnail_path):
    try:
        from PIL import Image, ImageDraw, ImageFont
        # Create a simple default thumbnail
        placeholder = Image.new('RGB', (800, 600), color=(53, 92, 125))
        draw = ImageDraw.Draw(placeholder)
        draw.text((400, 300), "Default Course Thumbnail", fill=(255, 255, 255))
        placeholder.save(default_thumbnail_path)
    except Exception as e:
        print(f"Could not create default thumbnail: {e}")

Session(app)

# Job queue and status tracking
job_queue = Queue()
job_statuses = {}

# Process to handle background jobs
def job_worker():
    while True:
        try:
            job_id, api_key, topic, level, quality = job_queue.get()
            try:
                # Update job status
                job_statuses[job_id]['status'] = 'processing'
                
                # Create course generator
                course_gen = CourseGenerator(
                    openai_api_key=api_key,
                    use_gpu=torch.cuda.is_available(),
                    optimize_memory=True,
                    high_quality=(quality == 'high')
                )
                
                # Generate course
                course = course_gen.generate_course(topic, level)
                
                # Save course to JSON
                filename = f"course_{job_id}.json"
                filepath = os.path.join(app.config['COURSE_FOLDER'], filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(course, f, indent=2, ensure_ascii=False)
                
                # Handle thumbnail path - copy instead of moving to avoid cross-drive issues
                if os.path.exists(course['thumbnail_path']):
                    try:
                        new_thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], f"{job_id}.png")
                        shutil.copy2(course['thumbnail_path'], new_thumbnail_path)
                        # Try to remove original file after successful copy
                        try:
                            os.remove(course['thumbnail_path'])
                        except:
                            pass  # Ignore errors on cleanup
                        course['thumbnail_path'] = f"/static/thumbnails/{job_id}.png"
                    except Exception as copy_error:
                        print(f"Error copying thumbnail: {copy_error}")
                        # Keep original path if copying fails
                        course['thumbnail_path'] = course['thumbnail_path'].replace("\\", "/")
                        if not course['thumbnail_path'].startswith("/"):
                            course['thumbnail_path'] = "/" + course['thumbnail_path']
                
                # Update job status with completion and course data
                job_statuses[job_id]['status'] = 'completed'
                job_statuses[job_id]['course'] = course
                job_statuses[job_id]['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            except Exception as e:
                print(f"Error processing job {job_id}: {str(e)}")
                # Update job status with error
                job_statuses[job_id]['status'] = 'failed'
                job_statuses[job_id]['error'] = str(e)
            
            finally:
                job_queue.task_done()
                
        except Exception as worker_error:
            print(f"Worker thread error: {worker_error}")
            # Continue processing next job even if this one fails completely
            continue

# Start worker thread
worker_thread = Thread(target=job_worker, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    """Home page with course generation form"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_course():
    """API endpoint to generate a new course"""
    # Get form data
    topic = request.form.get('topic', '')
    level = request.form.get('level', 'intermediate')
    api_key = request.form.get('api_key', '')
    quality = request.form.get('quality', 'high')
    
    # Validate input
    if not topic or not api_key:
        flash('Please provide both a topic and an OpenAI API key', 'danger')
        return redirect(url_for('index'))
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Store in session to track user's jobs
    if 'jobs' not in session:
        session['jobs'] = []
    
    session['jobs'].append(job_id)
    session.modified = True
    
    # Create job status entry
    job_statuses[job_id] = {
        'id': job_id,
        'topic': topic,
        'level': level,
        'quality': quality,
        'status': 'queued',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add job to queue
    job_queue.put((job_id, api_key, topic, level, quality))
    
    # Redirect to status page
    return redirect(url_for('job_status', job_id=job_id))

@app.route('/status/<job_id>')
def job_status(job_id):
    """Page to view job status and results"""
    # Check if job exists
    if job_id not in job_statuses:
        flash('Job not found', 'danger')
        return redirect(url_for('index'))
    
    # Get job status
    status = job_statuses[job_id]
    
    # Render status template
    return render_template('status.html', job=status)

@app.route('/api/status/<job_id>')
def api_job_status(job_id):
    """API endpoint to get job status"""
    if job_id not in job_statuses:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_statuses[job_id])

@app.route('/dashboard')
def dashboard():
    """Dashboard to view all user's jobs"""
    user_jobs = []
    if 'jobs' in session:
        for job_id in session['jobs']:
            if job_id in job_statuses:
                user_jobs.append(job_statuses[job_id])
    
    return render_template('dashboard.html', jobs=user_jobs)

@app.route('/view/<job_id>')
def view_course(job_id):
    """Page to view a completed course"""
    # Check if job exists and is completed
    if job_id not in job_statuses or job_statuses[job_id]['status'] != 'completed':
        flash('Course not found or still processing', 'warning')
        return redirect(url_for('dashboard'))
    
    course = job_statuses[job_id]['course']
    
    # Fix any path issues with the thumbnail
    if 'thumbnail_path' in course:
        if not course['thumbnail_path'].startswith('/'):
            course['thumbnail_path'] = '/' + course['thumbnail_path'].replace('\\', '/')
    
    return render_template('course.html', course=course, job_id=job_id)

@app.route('/download/<job_id>')
def download_course(job_id):
    """Download the course JSON file"""
    filename = f"course_{job_id}.json"
    filepath = os.path.join(app.config['COURSE_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        flash('Course file not found', 'danger')
        return redirect(url_for('dashboard'))
        
    return send_from_directory(app.config['COURSE_FOLDER'], filename, 
                              as_attachment=True, 
                              download_name=f"course_{job_id}.json")

@app.route('/generate_module/<job_id>/<int:module_index>', methods=['POST'])
def generate_module(job_id, module_index):
    """Generate detailed content for a specific module"""
    # Check if job exists and is completed
    if job_id not in job_statuses or job_statuses[job_id]['status'] != 'completed':
        return jsonify({'error': 'Course not found or still processing'}), 404
    
    # Get course data
    course = job_statuses[job_id]['course']
    
    # Check if module index is valid
    if module_index < 0 or module_index >= len(course['modules']):
        return jsonify({'error': 'Invalid module index'}), 400
    
    # Get API key from request
    api_key = request.form.get('api_key', '')
    if not api_key:
        return jsonify({'error': 'OpenAI API key is required'}), 400
    
    try:
        # Create course generator
        course_gen = CourseGenerator(
            openai_api_key=api_key,
            use_gpu=torch.cuda.is_available(),
            optimize_memory=True,
            high_quality=True
        )
        
        # Check if module content already exists in the course
        if 'module_contents' in course and str(module_index) in course['module_contents']:
            module_content = course['module_contents'][str(module_index)]
        else:
            # Generate module content
            module = course['modules'][module_index]
            module_content = course_gen.generate_module_content(
                course['title'],
                module['title'],
                module['lessons']
            )
            
            # Save module content
            filename = f"module_{job_id}_{module_index}.json"
            filepath = os.path.join(app.config['MODULE_FOLDER'], filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(module_content, f, indent=2, ensure_ascii=False)
            
            # Update job status with module data
            if 'module_contents' not in job_statuses[job_id]:
                job_statuses[job_id]['module_contents'] = {}
            
            job_statuses[job_id]['module_contents'][str(module_index)] = module_content
            
            # Update the course object as well
            if 'module_contents' not in course:
                course['module_contents'] = {}
            course['module_contents'][str(module_index)] = module_content
        
        return jsonify({
            'success': True,
            'module': module_content
        })
        
    except Exception as e:
        print(f"Error generating module: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_module/<job_id>/<int:module_index>')
def view_module(job_id, module_index):
    """Page to view a generated module"""
    # Check if job exists and is completed
    if job_id not in job_statuses or job_statuses[job_id]['status'] != 'completed':
        flash('Course not found or still processing', 'warning')
        return redirect(url_for('dashboard'))
    
    # Get course data
    course = job_statuses[job_id]['course']
    
    # Check if module index is valid
    if module_index < 0 or module_index >= len(course['modules']):
        flash('Invalid module index', 'danger')
        return redirect(url_for('view_course', job_id=job_id))
    
    # Check if module content has been generated
    if 'module_contents' in course and str(module_index) in course['module_contents']:
        # Module content already exists in the course object
        module_content = course['module_contents'][str(module_index)]
    elif ('module_contents' in job_statuses[job_id] and 
          str(module_index) in job_statuses[job_id]['module_contents']):
        # Module content exists in job status but not in course object
        module_content = job_statuses[job_id]['module_contents'][str(module_index)]
        # Update course object
        if 'module_contents' not in course:
            course['module_contents'] = {}
        course['module_contents'][str(module_index)] = module_content
    else:
        # Module content not generated yet
        return render_template('generate_module.html', 
                              course=course, 
                              module=course['modules'][module_index],
                              module_index=module_index,
                              job_id=job_id)
    
    return render_template('module.html', 
                          course=course,
                          module=module_content,
                          module_index=module_index,
                          job_id=job_id)

@app.route('/about')
def about():
    """About page with information about the application"""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
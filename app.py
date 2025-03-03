import os
import uuid
import base64
import subprocess
import requests
import json
import time
import re
from gtts import gTTS
from flask import Flask, request, jsonify, render_template, session as flask_session
from dotenv import load_dotenv
import speech_recognition as sr
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import smartsheet  # New: Import smartsheet SDK

# Load .env
load_dotenv()

# OpenAI API configurations
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"  # If nesessary, we can use the more capable GPT-4o model for better conversational abilities

# Smartsheet API configurations (New)
SMARTSHEET_API_KEY = os.getenv("SMARTSHEET_API_KEY")
SMARTSHEET_SHEET_ID = os.getenv("SMARTSHEET_SHEET_ID")

# FFmpeg path configuration
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
if FFMPEG_PATH:
    os.environ["PATH"] = FFMPEG_PATH + ";" + os.environ["PATH"]

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URI", 'sqlite:///coaching_sessions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define coaching steps
COACHING_STEPS = [
    "center_together", 
    "clarify_focus", 
    "identify_goal", 
    "develop_action_plan", 
    "gain_commitment", 
    "assess_progress"
]

STEP_DESCRIPTIONS = {
    "center_together": "Centering together involves helping the coachee clear their mind and focus on the present moment.",
    "clarify_focus": "Clarifying the focus involves understanding the specific challenge or issue the coachee wants to address.",
    "identify_goal": "Identifying the goal involves establishing what the coachee wants to achieve from this coaching session.",
    "develop_action_plan": "Developing an action plan involves determining specific steps to achieve the identified goal.",
    "gain_commitment": "Gaining commitment involves ensuring the coachee is motivated and committed to the action plan.",
    "assess_progress": "Assessing progress involves discussing how the coachee will track their progress and how you'll follow up."
}

# Load the coaching content reference material
with open("content_summary.txt", "r", encoding="utf-8") as f:
    CONTENT_SUMMARY = f.read()

# Load step-specific content
STEP_CONTENT = {}
for step in COACHING_STEPS:
    try:
        with open(f"content/{step}.txt", "r", encoding="utf-8") as f:
            STEP_CONTENT[step] = f.read()
    except FileNotFoundError:
        STEP_CONTENT[step] = ""  # Fallback if file doesn't exist

# Database models
class CoachingSession(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    scenario_type = db.Column(db.String(50))
    difficulty = db.Column(db.String(20))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    final_score = db.Column(db.Float, nullable=True)
    final_grade = db.Column(db.String(2), nullable=True)
    strengths = db.Column(db.Text, nullable=True)
    areas_of_improvement = db.Column(db.Text, nullable=True)
    conversation = db.relationship('ConversationMessage', backref='session', lazy=True)

class ConversationMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('coaching_session.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'coach' or 'employee'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    step = db.Column(db.String(50), nullable=True)  # Current coaching step
    feedback = db.Column(db.Text, nullable=True)  # Feedback for this message

# Create the database and tables
with app.app_context():
    db.create_all()

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
reference_embed = model.encode(CONTENT_SUMMARY, convert_to_tensor=True)
step_embeds = {step: model.encode(content, convert_to_tensor=True) for step, content in STEP_CONTENT.items()}

#######################################################
# Audio Processing Functions
#######################################################
def convert_to_wav(input_file, output_file):
    """Convert audio file to WAV format using ffmpeg."""
    try:
        ffmpeg_cmd = [FFMPEG_PATH, "-y", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_file]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        print(f"STDERR: {e.stderr.decode() if e.stderr else 'None'}")
        return False

def generate_speech(text):
    """Convert text to speech using gTTS  and return as a data URI without saving to disk."""
    tts = gTTS(text=text, lang='en', slow=False)  # Create gTTS object
    fp = BytesIO()  # Create a BytesIO object to work in memory
    tts.write_to_fp(fp)  # Write audio data to memory
    fp.seek(0)  # Reset pointer to beginning
    base64_audio = base64.b64encode(fp.read()).decode("utf-8")  # Base64 encode the binary data
    data_uri = f"data:audio/mp3;base64,{base64_audio}"  # Format as a data URI
    return data_uri  # Return the data URI

def transcribe_audio(base64_audio):
    """Transcribe audio using speech recognition."""
    try:
        # Create temp files
        audio_bytes = base64.b64decode(base64_audio.split(',')[1])
        webm_filename = f"temp_{uuid.uuid4()}.webm"
        wav_filename = f"temp_{uuid.uuid4()}.wav"
        
        # Write audio bytes to file
        with open(webm_filename, 'wb') as f:
            f.write(audio_bytes)
        
        # Convert to WAV
        if not convert_to_wav(webm_filename, wav_filename):
            return "Could not process audio. Please try again."
        
        # Use speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_filename) as source:
            audio = recognizer.record(source)
            try:
                # Try Google's service first
                text = recognizer.recognize_google(audio)
            except:
                try:
                    # Fallback to Sphinx (offline)
                    text = recognizer.recognize_sphinx(audio)
                except:
                    text = "Could not understand audio. Please try again."
        
        # Clean up files
        try:
            os.remove(webm_filename)
            os.remove(wav_filename)
        except:
            pass
            
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "Error processing audio. Please try again."

#######################################################
# OpenAI API Functions
#######################################################
def call_openai_api(messages, model=DEFAULT_MODEL, temperature=0.7, max_tokens=800):
    """Call OpenAI API with a chat completion request"""
    try:
        if not OPENAI_API_KEY:
            return "Error: OpenAI API key not found in environment variables."
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Retry mechanism for API calls
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(OPENAI_API_ENDPOINT, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "No content returned from API."
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e
                    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error: Could not generate text. {str(e)}"

#######################################################
# Coaching Scenario Generation
#######################################################
def generate_scenario_intro(difficulty, scenario_type):
    """
    Generate a short introduction (3-4 sentences) for a child welfare or child protective services scenario.
    The introduction should be from the perspective of a worker who needs coaching in a child welfare context.
    """
    system_message = {
        "role": "system", 
        "content": "You are a coaching scenario generator, creating realistic workplace situations in child welfare or child protective services that require coaching."
    }
    
    user_message = {
        "role": "user", 
        "content": f"""
Create a short introduction (3-4 sentences) from the perspective of a child welfare worker (e.g., a Child Protective Specialist)
who needs coaching. The scenario should focus on child welfare or child protective contexts.

- Difficulty level: {difficulty.capitalize()}
- Scenario type: {scenario_type.capitalize()}
Possible topics include safety assessments, mandated reporting, family engagement, dealing with resistant parents,
or provider agency collaboration. Make it conversational and realistic. The worker should briefly explain their situation and challenge.
"""
    }
    
    return call_openai_api([system_message, user_message], temperature=0.8)

#######################################################
# Relevance and Step Detection
#######################################################
def detect_coaching_step(text):
    """Detect which coaching step the text is most aligned with."""
    text_embed = model.encode(text, convert_to_tensor=True)
    
    best_score = -1
    best_step = COACHING_STEPS[0]  # Default to first step
    
    for step, step_embed in step_embeds.items():
        score = float(util.cos_sim(text_embed, step_embed)[0][0])
        if score > best_score:
            best_score = score
            best_step = step
    
    return best_step, best_score

def check_step_alignment(text, current_step):
    """Check if text aligns with the current coaching step."""
    text_embed = model.encode(text, convert_to_tensor=True)
    step_embed = step_embeds.get(current_step, reference_embed)
    
    score = float(util.cos_sim(text_embed, step_embed)[0][0])
    # Return true if the score meets threshold
    return score >= 0.25, score

def check_overall_alignment(text):
    """Check if text aligns with the overall coaching content."""
    text_embed = model.encode(text, convert_to_tensor=True)
    score = float(util.cos_sim(text_embed, reference_embed)[0][0])
    return score >= 0.15, score

#######################################################
# Employee Response Generation
#######################################################
def generate_employee_response(session_id, coach_text, current_step):
    """Generate the employee's response based on the conversation context and current step."""
    # Get the conversation history
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    
    # Format conversation for OpenAI
    conversation_history = []
    for msg in messages:
        if msg.role == "coach":
            conversation_history.append({"role": "user", "content": msg.content})
        else:
            conversation_history.append({"role": "assistant", "content": msg.content})
    
    # Add system message with context about the coaching step and FORCE JSON output format
    system_message = {
        "role": "system", 
        "content": f"""
You are playing the role of an employee in a coaching scenario. 
The coach is currently in the '{current_step.replace('_', ' ').title()}' step of the coaching process.

Your responses should:
1. Be realistic and conversational (1-3 sentences)
2. Respond appropriately to the coach's questions/statements
3. Provide a response that helps the coach practice the current step: {STEP_DESCRIPTIONS[current_step]}

Also evaluate how well the coach is doing with these skills:
- Presence and listening
- Reflecting and questioning
- Providing feedback
- Building accountability

YOU MUST return ONLY a valid JSON object with EXACTLY this format:
{{
  "reply": "your in-character employee response",
  "feedback": ["feedback point 1", "feedback point 2"]
}}

The feedback field must be an array of strings. Do not include any explanation, preamble, or other text.
"""
    }
    
    # Add the coach's latest message
    latest_message = {"role": "user", "content": coach_text}
    
    # Combine all messages
    all_messages = [system_message] + conversation_history + [latest_message]
    
    # Get response from OpenAI
    response = call_openai_api(all_messages, temperature=0.7)
    
    # Improved JSON parsing with better error handling
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.replace("```json", "", 1)
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response.rsplit("```", 1)[0]
        cleaned_response = cleaned_response.strip()
        
        parsed = json.loads(cleaned_response)
        
        if "reply" not in parsed:
            parsed["reply"] = "I'm not sure how to respond to that."
        if "feedback" not in parsed:
            parsed["feedback"] = ["The coach could improve their question clarity."]
        
        if not isinstance(parsed["feedback"], list):
            parsed["feedback"] = [str(parsed["feedback"])]
            
        return parsed["reply"], parsed["feedback"]
        
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response}")
        
        # Fallback
        try:
            reply_match = re.search(r'"reply"\s*:\s*"([^"]+)"', response)
            if reply_match:
                reply = reply_match.group(1)
            else:
                reply = response.split('\n')[0][:100] + "..."
            return reply, ["Could not parse feedback: Please check LLM response format"]
        except:
            return "I'm sorry, could you repeat that?", ["Could not parse feedback from response"]

#######################################################
# Generate Final (or Partial) Evaluation
#######################################################
def generate_final_evaluation(session_id):
    """Generate a comprehensive final evaluation of the coaching session (can be used for partial too)."""
    # 1. 모든 메시지 가져오기
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    
    #2. 코치 메시지 수 체크 (예: 3개 미만이면 충분한 참여가 없다고 판단)
    coach_messages = [msg for msg in messages if msg.role == "coach"]
    if len(coach_messages) < 3:
        return {
            "percentage": 0,
            "grade": "F",
            "strengths": ["No significant coaching interaction occurred."],
            "areas_of_improvement": ["Coach needs to engage more actively in the session."]
        }    
    
    # 3. 대화 내용을 하나의 문자열로 구성 (각 메시지 앞에 역할 태그 포함)
    conversation_text = ""
    for msg in messages:
        if msg.role == "coach":
            conversation_text += f"Coach: {msg.content}\n"
        else:
            conversation_text += f"Employee: {msg.content}\n"
    
    # 4. 평가를 위한 시스템 프롬프트 구성 (추가 조건 포함)
    system_message = {
        "role": "system", 
        "content": """
You are an expert coach evaluator. Analyze the coaching conversation, focusing exclusively on the coach's performance. Evaluate how effectively the coach demonstrated the following skills and adhered to the coaching process:

1. Coaching Steps:
   - Centering Together
   - Clarifying the Focus
   - Identifying the Goal
   - Developing an Action Plan
   - Gaining Commitment
   - Assessing Progress

2. Coaching Skills:
   - Questioning (open-ended and probing)
   - Presence & Listening (attentiveness and allowing silence)
   - Reflecting & Clarifying (paraphrasing and summarizing)
   - Providing Feedback (strengths-based and specific)
   - Accountability (establishing clear expectations)

Provide a detailed evaluation that includes:
- An overall percentage score (0-100)
- A letter grade (A-F)
- 3-5 key strengths
- 2-3 focused areas for improvement

Format your response as JSON:
{
  "percentage": 85,
  "grade": "B",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "areas_of_improvement": ["Area 1", "Area 2"]
}
"""
    }
    # 5. 사용자 메시지에 대화 내용 첨부
    user_message = {
        "role": "user", 
        "content": f"""
Here is the complete coaching conversation to evaluate:

{conversation_text}

Please provide a final evaluation as specified.
"""
    }
     # 6. OpenAI API 호출하여 평가 응답 받기
    response = call_openai_api([system_message, user_message], temperature=0.3, max_tokens=1000)
    
    # 7. 응답을 JSON으로 파싱 시도 및 fallback 처리
    try:
        parsed = json.loads(response)
    except:
        try:
            json_match = re.search(r'\{.*"percentage".*"grade".*"strengths".*"areas_of_improvement".*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                return {
                    "percentage": 70,
                    "grade": "C",
                    "strengths": ["Communication skills", "Building rapport"],
                    "areas_of_improvement": ["Structure coaching process better", "Ask more open-ended questions"]
                }
        except:
            return {
                "percentage": 70,
                "grade": "C",
                "strengths": ["Communication skills", "Building rapport"],
                "areas_of_improvement": ["Structure coaching process better", "Ask more open-ended questions"]
            }
    
    return parsed

#######################################################
# Smartsheet Integration Functions (New)
#######################################################
def save_session_to_smartsheet(session_id):
    """
    Save the coaching session data to Smartsheet.
    Columns: Difficulty Level, Scenario Type, Conversation, Coaching Feedback, Feedback Summary, Score, Grade.
    """
    # Get session information from the database
    session_obj = CoachingSession.query.get(session_id)
    if not session_obj:
        print("Session not found.")
        return

    # Retrieve all conversation messages for this session (both coach and employee)
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    conversation_text = ""
    for msg in messages:
        conversation_text += f"{msg.role.capitalize()}: {msg.content}\n"
    
    # Create a feedback summary from the final evaluation (strengths and areas_of_improvement)
    try:
        strengths = json.loads(session_obj.strengths) if session_obj.strengths else []
    except:
        strengths = []
    try:
        improvements = json.loads(session_obj.areas_of_improvement) if session_obj.areas_of_improvement else []
    except:
        improvements = []
    feedback_summary = "Strengths: " + ", ".join(strengths) + " | Areas for Improvement: " + ", ".join(improvements)
    
    # For Coaching Feedback, aggregate all coach messages as an example
    coaching_feedback = ""
    for msg in messages:
         # Initialize feedback_list to ensure it is always defined
        feedback_list = []
         # If the message is from an employee and has a feedback field, parse it
        if msg.role == "employee" and msg.feedback:
            try:
                feedback_list = json.loads(msg.feedback)
            except json.JSONDecodeError:
                feedback_list = []
        # Append each feedback line with a newline separator
        for f in feedback_list:
            coaching_feedback += f + "\n"
    
    # Initialize Smartsheet client
    smartsheet_client = smartsheet.Smartsheet(SMARTSHEET_API_KEY)
    smartsheet_client.errors_as_exceptions(True)
    
    # Prepare the new row with the required columns.
    new_row = smartsheet.models.Row()
    new_row.to_bottom = True  # Append the row at the bottom of the sheet

    # Create cell objects and assign properties
    cell0 = smartsheet.models.Cell()
    cell0.column_id = get_column_id("ID")
    cell0.value = session_obj.id

    cell1 = smartsheet.models.Cell()
    cell1.column_id = get_column_id("Difficulty Level")
    cell1.value = session_obj.difficulty

    cell2 = smartsheet.models.Cell()
    cell2.column_id = get_column_id("Scenario Type")
    cell2.value = session_obj.scenario_type

    cell3 = smartsheet.models.Cell()
    cell3.column_id = get_column_id("Conversation")
    cell3.value = conversation_text

    cell4 = smartsheet.models.Cell()
    cell4.column_id = get_column_id("Coaching Feedback")
    cell4.value = coaching_feedback

    cell5 = smartsheet.models.Cell()
    cell5.column_id = get_column_id("Feedback Summary")
    cell5.value = feedback_summary

    cell6 = smartsheet.models.Cell()
    cell6.column_id = get_column_id("Score")
    cell6.value = session_obj.final_score

    cell7 = smartsheet.models.Cell()
    cell7.column_id = get_column_id("Grade")
    cell7.value = session_obj.final_grade

    new_row.cells = [cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7]
    
    # Add the new row to the Smartsheet
    try:
        response = smartsheet_client.Sheets.add_rows(SMARTSHEET_SHEET_ID, [new_row])
        print("Session saved to Smartsheet.")
    except Exception as e:
        print("Error saving to Smartsheet:", e)

def get_column_id(column_name):
    """
    Helper function to get the Smartsheet column ID by its title.
    This function should return the column ID corresponding to the given column_name.
    You need to fill in the mapping from column names to their actual Smartsheet column IDs.
    """
    # IMPORTANT: Replace the dictionary below with your actual column name to column ID mapping.
    column_mapping = {
        "ID": 6891979856367492,  # Replace with your actual column ID
        "Difficulty Level": 4480157681405828,  # Replace with your actual column ID
        "Scenario Type": 8983757308776324,      # Replace with your actual column ID
        "Conversation": 4532934239539076,         # Replace with your actual column ID
        "Coaching Feedback": 29334612168580,    # Replace with your actual column ID
        "Feedback Summary": 7250363805814660,      # Replace with your actual column ID
        "Score": 2746764178444164,                # Replace with your actual column ID
        "Grade": 4998563992129412                 # Replace with your actual column ID
    }
    return column_mapping.get(column_name)

#######################################################
# Flask Routes
#######################################################
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start-simulation", methods=["POST"])
def start_simulation():
    data = request.json
    difficulty = data.get("difficulty", "beginner")
    scenario_type = data.get("scenario_type", "performance")
    
    intro_text = generate_scenario_intro(difficulty, scenario_type)
    
    session_id = str(uuid.uuid4())
    
    new_session = CoachingSession(
        id=session_id,
        scenario_type=scenario_type,
        difficulty=difficulty
    )
    db.session.add(new_session)
    
    intro_message = ConversationMessage(
        session_id=session_id,
        role="employee",
        content=intro_text,
        step="center_together"
    )
    db.session.add(intro_message)
    db.session.commit()
    
    audio_file = generate_speech(intro_text)
    
    return jsonify({
        "session_id": session_id,
        "text": intro_text,
        "audio": audio_file,
        "scenario_name": f"{scenario_type.capitalize()} ({difficulty.capitalize()})",
        "current_step": "center_together",
        "step_description": STEP_DESCRIPTIONS["center_together"]
    })

@app.route("/api/respond", methods=["POST"])
def respond():
    data = request.json
    session_id = data.get("session_id")
    audio_data = data.get("audio")
    text = data.get("text", "")
    
    session = CoachingSession.query.get(session_id)
    if not session:
        return jsonify({"error": "Invalid session"}), 400
    
    if session.end_time:  # already complete
        return jsonify({"error": "Session is already complete"}), 400
    
    # Get user (coach) text
    if audio_data:
        user_text = transcribe_audio(audio_data)
    else:
        user_text = text.strip()
    
    if not user_text or user_text == "Could not understand audio. Please try again.":
        return jsonify({
            "error": "Could not understand audio", 
            "text": "I couldn't hear you clearly. Could you please repeat that?",
            "audio": generate_speech("I couldn't hear you clearly. Could you please repeat that?")
        }), 200
    
    # Determine current step
    last_message = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp.desc()).first()
    current_step = last_message.step if last_message else "center_together"
    
    # Save coach message
    coach_message = ConversationMessage(
        session_id=session_id,
        role="coach",
        content=user_text,
        step=current_step
    )
    db.session.add(coach_message)
    db.session.commit()
    
    # Step alignment logic
    is_aligned, alignment_score = check_step_alignment(user_text, current_step)
    detected_step, step_score = detect_coaching_step(user_text)
    current_step_index = COACHING_STEPS.index(current_step)
    
    should_advance = False
    step_count = ConversationMessage.query.filter_by(session_id=session_id, step=current_step).count()
    
    if COACHING_STEPS.index(detected_step) > current_step_index and step_score > 0.35:
        should_advance = True
        next_step = detected_step
    elif step_count >= 6 and COACHING_STEPS.index(detected_step) >= current_step_index:
        should_advance = True
        next_step = COACHING_STEPS[min(current_step_index + 1, len(COACHING_STEPS) - 1)]
    else:
        next_step = current_step
    
    # Generate employee response
    reply_text, feedback_list = generate_employee_response(session_id, user_text, next_step)
    
    # Save employee message
    employee_message = ConversationMessage(
        session_id=session_id,
        role="employee",
        content=reply_text,
        step=next_step,
        feedback=json.dumps(feedback_list)
    )
    db.session.add(employee_message)
    db.session.commit()
    
    audio_file = generate_speech(reply_text)
    
    # Check completion
    completed_steps = {msg.step for msg in ConversationMessage.query.filter_by(session_id=session_id).all()}
    is_complete = False
    final_score = None
    
    if set(COACHING_STEPS).issubset(completed_steps) and next_step == COACHING_STEPS[-1]:
        # All steps done, do final
        is_complete = True
        evaluation = generate_final_evaluation(session_id)
        session.end_time = datetime.utcnow()
        session.final_score = evaluation["percentage"]
        session.final_grade = evaluation["grade"]
        session.strengths = json.dumps(evaluation["strengths"])
        session.areas_of_improvement = json.dumps(evaluation["areas_of_improvement"])
        db.session.commit()
        final_score = evaluation
    
    # --- [추가] 세션이 아직 완료되지 않았어도, 지금까지 대화로 partial 평가 생성 ---
    partial_score = None
    if not is_complete:
        partial_score = generate_final_evaluation(session_id)
    # --------------------------------------------------------------------------
    
    return jsonify({
        "text": reply_text,
        "audio": audio_file,
        "evaluation": {"feedback": feedback_list},
        "is_complete": is_complete,
        "final_score": final_score,
        "partial_score": partial_score,  # 아직 완료되지 않은 경우에 대한 부분 평가
        "current_step": next_step,
        "step_description": STEP_DESCRIPTIONS[next_step],
        "step_advanced": should_advance,
        "coach_input": user_text
    })

@app.route("/api/skip-to-end", methods=["POST"])
def skip_to_end():
    data = request.json
    session_id = data.get("session_id")
    
    session = CoachingSession.query.get(session_id)
    if not session:
        return jsonify({"error": "Invalid session"}), 400
    
    if session.end_time:  # already complete
        return jsonify({"error": "Session is already complete"}), 400
    
    evaluation = generate_final_evaluation(session_id)
    session.end_time = datetime.utcnow()
    session.final_score = evaluation["percentage"]
    session.final_grade = evaluation["grade"]
    session.strengths = json.dumps(evaluation["strengths"])
    session.areas_of_improvement = json.dumps(evaluation["areas_of_improvement"])
    db.session.commit()
    
    # New: Save the session data to Smartsheet
    save_session_to_smartsheet(session_id)
    
    return jsonify({
        "message": "Session completed successfully.",
        "final_score": evaluation
    })

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    sessions = CoachingSession.query.filter(CoachingSession.end_time.isnot(None)).order_by(CoachingSession.end_time.desc()).all()
    
    result = []
    for session in sessions:
        result.append({
            "id": session.id,
            "scenario_type": session.scenario_type,
            "difficulty": session.difficulty,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "final_score": session.final_score,
            "final_grade": session.final_grade,
            "strengths": json.loads(session.strengths) if session.strengths else [],
            "areas_of_improvement": json.loads(session.areas_of_improvement) if session.areas_of_improvement else []
        })
    
    return jsonify(result)

@app.route("/api/session/<session_id>", methods=["GET"])
def get_session_details(session_id):
    session = CoachingSession.query.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    
    message_list = []
    for msg in messages:
        message_list.append({
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "step": msg.step,
            "feedback": json.loads(msg.feedback) if msg.feedback else None
        })
    
    result = {
        "id": session.id,
        "scenario_type": session.scenario_type,
        "difficulty": session.difficulty,
        "start_time": session.start_time.isoformat(),
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "final_score": session.final_score,
        "final_grade": session.final_grade,
        "strengths": json.loads(session.strengths) if session.strengths else [],
        "areas_of_improvement": json.loads(session.areas_of_improvement) if session.areas_of_improvement else [],
        "messages": message_list
    }
    
    return jsonify(result)

if __name__ == "__main__":
    os.makedirs("static/audio", exist_ok=True)
    app.run(debug=True, port=5000)

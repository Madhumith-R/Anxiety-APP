import streamlit as st
import sqlite3
import os
import google.generativeai as genai
import json
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from io import BytesIO
from collections import deque
import seaborn as sns

# Configure Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("Please set your GEMINI_API_KEY environment variable")
    st.stop()
genai.configure(api_key=API_KEY)

def ask_gemini(prompt, model_name="gemini-2.0-flash"):
    """Interacts with the Gemini API with error handling and retries."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        # Provide a fallback response
        return '{"score": 5, "classification": "Moderate anxiety"}'

# Enhanced database functions
class Database:
    def __init__(self, db_name="anxiety_app.db"):
        self.db_name = db_name
        self.init_db()
    
    def get_connection(self):
        return sqlite3.connect(self.db_name)
    
    def init_db(self):
        conn = self.get_connection()
        c = conn.cursor()
        
        # Users table with more profile information
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                email TEXT,
                anxiety_state TEXT DEFAULT 'initial',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
        """)
        
        # Enhanced progress tracking
        c.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                score REAL,
                classification TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        """)
        
        # Suggestions and feedback tracking
        c.execute("""
            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                suggestion TEXT,
                state TEXT,
                feedback INTEGER,
                effectiveness REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        """)
        
        # Store model parameters
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                parameters BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def authenticate(self, username, password):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        
        if user:
            # Update last login time
            c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?", (username,))
            conn.commit()
            
        conn.close()
        return user
    
    def register_user(self, username, password, email=None):
        conn = self.get_connection()
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", 
                     (username, password, email))
            conn.commit()
            success = True
        except sqlite3.IntegrityError:
            success = False
        finally:
            conn.close()
        return success
    
    def save_progress(self, username, score, classification, notes=None):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("INSERT INTO progress (username, score, classification, notes) VALUES (?, ?, ?, ?)", 
                 (username, score, classification, notes))
        conn.commit()
        conn.close()
    
    def get_user_progress(self, username, limit=10):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT score, classification, timestamp 
            FROM progress 
            WHERE username = ? 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (username, limit))
        progress = c.fetchall()
        conn.close()
        return progress
    
    def save_suggestion_feedback(self, username, suggestion, state, feedback, effectiveness):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO suggestions (username, suggestion, state, feedback, effectiveness) 
            VALUES (?, ?, ?, ?, ?)
        """, (username, suggestion, state, feedback, effectiveness))
        conn.commit()
        conn.close()
    
    def get_best_suggestions(self, username, state, limit=3):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT suggestion, AVG(effectiveness) as avg_effectiveness
            FROM suggestions
            WHERE username = ? AND state = ? AND effectiveness > 0
            GROUP BY suggestion
            ORDER BY avg_effectiveness DESC
            LIMIT ?
        """, (username, state, limit))
        suggestions = c.fetchall()
        conn.close()
        return suggestions if suggestions else []
    
    def save_model_state(self, model_type, parameters):
        conn = self.get_connection()
        c = conn.cursor()
        try:
            # Serialize the parameters properly
            import pickle
            serialized = pickle.dumps(parameters)
        
            c.execute("INSERT INTO model_states (model_type, parameters) VALUES (?, ?)", 
                 (model_type, serialized))
            conn.commit()
            print(f"Model state for {model_type} saved successfully")
        except Exception as e:
            print(f"Error saving model state: {str(e)}")
        finally:
            conn.close()

    def load_latest_model_state(self, model_type):
        conn = self.get_connection()
        c = conn.cursor()
        try:
            c.execute("""
                SELECT parameters FROM model_states 
                WHERE model_type = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (model_type,))
            result = c.fetchone()
        
            if result:
                # Deserialize parameters
                import pickle
                return pickle.loads(result[0])
        except Exception as e:
            print(f"Error loading model state: {str(e)}")
        finally:
            conn.close()
    
        return None
    
# Advanced Deep Q-Network for Anxiety Management
class DeepQLearningAgent:
    def __init__(self, state_size=10, memory_size=2000):
        # Actions catalog - expanded with cognitive behavioral techniques
        self.actions = [
            "Practice deep breathing: Inhale for 4 counts, hold for 7, exhale for 8. Repeat 5 times.",
            "Take a 15-minute nature walk, focusing on your surroundings rather than your worries.",
            "Write three things you're grateful for today to help shift your perspective.",
            "Listen to your anxiety playlist with calming or uplifting music.",
            "Practice progressive muscle relaxation starting from your toes up to your head.",
            "Do 10 minutes of gentle yoga stretches to release physical tension.",
            "Make a cup of herbal tea and practice mindful drinking, focusing on temperature and taste.",
            "Call or message a supportive friend with a simple 'How are you?' to connect.",
            "Spend 15 minutes journaling about your feelings without judgment.",
            "Try '5-4-3-2-1' grounding: Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
            "Challenge a negative thought by writing evidence for and against it.",
            "Set a 10-minute timer to tidy your immediate environment.",
            "Practice a brief body scan meditation to reconnect with physical sensations.",
            "Listen to a 5-minute guided meditation from a reputable app.",
            "Draw or doodle freely for 10 minutes without judging the results."
        ]
        
        # Set action_size based on the length of the actions list
        self.action_size = len(self.actions)
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        assert self.action_size == len(self.actions), "action_size must match the number of actions"

    def _build_model(self):
        """Neural Net for Deep-Q learning Model"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(48, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, exploration=True):
        """Determine action based on state with proper bounds checking"""
        if exploration and np.random.rand() <= self.epsilon:
            return random.randrange(len(self.actions))
    
        act_values = self.model.predict(state, verbose=0)
    
        # Ensure index is within bounds
        action_index = np.argmax(act_values[0])
        if action_index >= len(self.actions):
            print(f"Warning: Model predicted invalid action index {action_index}, adjusting to valid range")
            action_index = action_index % len(self.actions)  # Wrap around to valid range
    
        return action_index
    
    def replay(self, batch_size=32):
        """Train model with experiences from memory"""
        if len(self.memory) < batch_size:
            return
    
        minibatch = random.sample(self.memory, batch_size)
    
        for state, action, reward, next_state, done in minibatch:
            # Ensure action index is valid
            if not 0 <= action < self.action_size:
                print(f"Warning: Skipping invalid action {action} in replay (valid range: 0-{self.action_size-1})")
                continue
            
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
        
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, weights):
        """Load model weights from memory with error handling"""
        if weights:
            try:
            # Convert bytes to numpy array if stored as BLOB
                if isinstance(weights, bytes):
                    import pickle
                    weights = pickle.loads(weights)
                
                # Check if weights match the expected shapes
                if len(weights) == len(self.model.get_weights()):
                    self.model.set_weights(weights)
                    self.target_model.set_weights(weights)
                    print("Model weights loaded successfully")
                else:
                    print(f"Warning: Weight shapes don't match. Expected {len(self.model.get_weights())}, got {len(weights)}")
            except Exception as e:
                print(f"Error loading model weights: {str(e)}")
    
    def save(self):
        """Get model weights for storage"""
        return self.model.get_weights()
    
    def encode_state(self, classification, user_history=None):
        """Convert anxiety state and user history to a numerical state vector"""
        # Basic encoding of anxiety levels
        anxiety_levels = {
            "minimal": 0.0,
            "mild": 0.25,
            "moderate": 0.5, 
            "severe": 0.75,
            "extreme": 1.0
        }
        
        # Default state vector
        state = np.zeros(self.state_size)
        
        # Set anxiety level
        classification = classification.lower()
        for key in anxiety_levels:
            if key in classification:
                state[0] = anxiety_levels[key]
                break
        
        # If user history available, encode it
        if user_history:
            # Recent trend (improving/worsening)
            if len(user_history) >= 2:
                recent_trend = user_history[0][0] - user_history[1][0]  # Most recent score change
                state[1] = np.clip(recent_trend / 10.0, -1.0, 1.0)  # Normalize
            
            # Volatility
            if len(user_history) >= 3:
                scores = [entry[0] for entry in user_history[:5]]
                state[2] = np.std(scores) / 10.0  # Normalized standard deviation
            
            # Time of day preference (encoded from timestamp)
            if len(user_history) >= 1:
                time_of_day = datetime.datetime.strptime(user_history[0][2], "%Y-%m-%d %H:%M:%S").hour / 24.0
                state[3] = time_of_day
        
        return state.reshape(1, self.state_size)
    
    def get_suggestion(self, classification, user_history=None, exploration=True):
        """Get a suitable action for the current state with bounds checking"""
        state = self.encode_state(classification, user_history)
        action_index = self.act(state, exploration)
        
        # Ensure action_index is within the valid range
        if action_index >= len(self.actions):
            action_index = random.randrange(len(self.actions))
            
        return self.actions[action_index], action_index
    
    def process_feedback(self, classification, action_index, feedback_score, user_history=None):
        """Process user feedback to improve model"""
        state = self.encode_state(classification, user_history)
        next_state = state  # In this case, the state doesn't change immediately after an action
        
        # Convert feedback to reward
        reward = feedback_score  # Assuming feedback_score is already normalized between -1 and 1
        
        # Add to memory and train
        self.memorize(state, action_index, reward, next_state, False)
        self.replay(batch_size=min(32, len(self.memory)))


# User Interface Manager
class UIManager:
    def __init__(self, db, agent):
        self.db = db
        self.agent = agent
        
    def show_login_page(self):
        st.subheader("Login to Your Account")
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", key="login_button"):
                if self.db.authenticate(username, password):
                    st.session_state["user"] = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        
        with col2:
            st.markdown("""
            ### Welcome Back
            
            Sign in to access your anxiety tracking dashboard and get personalized recommendations.
            
            New here? Register from the sidebar menu.
            """)
            
    def show_register_page(self):
        st.subheader("Create an Account")
        col1, col2 = st.columns(2)
        
        with col1:
            new_user = st.text_input("Choose a Username")
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email (optional)")
            
            if st.button("Register", key="register_button"):
                if not new_user or not new_password:
                    st.error("Username and password are required.")
                elif new_password != confirm_password:
                    st.error("Passwords don't match.")
                elif self.db.register_user(new_user, new_password, email):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists.")
        
        with col2:
            st.markdown("""
            ### Join Our Community
            
            Create an account to:
            - Track your anxiety levels over time
            - Get personalized coping strategies
            - See your progress visually
            - Learn effective anxiety management techniques
            """)
    
    def show_assessment_page(self):
        st.subheader("Anxiety Assessment")
        
        if "current_question" not in st.session_state:
            st.session_state.current_question = 0
        if "responses" not in st.session_state:
            st.session_state.responses = []
        if "questions" not in st.session_state:
            st.session_state.questions = []
        if "assessment_done" not in st.session_state:
            st.session_state.assessment_done = False
        
        # Assessment instructions and progress bar
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            Please answer each question honestly to get the most accurate assessment.
            Your responses are confidential and will help us provide personalized recommendations.
            """)
        with col2:
            if st.session_state.questions:
                progress = min(100, int(st.session_state.current_question / len(st.session_state.questions) * 100))
                st.progress(progress / 100)
                st.text(f"Question {st.session_state.current_question + 1} of {len(st.session_state.questions)}")
        
        # Load questions if needed
        if not st.session_state.questions:
            with st.spinner("Loading assessment questions..."):
                question_prompt = (
                    "Generate 10 multiple-choice questions to assess anxiety levels. "
                    "Questions should cover physical symptoms, cognitive patterns, behavioral responses, "
                    "and emotional experiences related to anxiety. "
                    "Each question should have 5 answer choices, allowing only one answer. "
                    "Format as valid JSON: {\"questions\": [{\"question\": \"text\", \"choices\": [\"Option 1\", \"Option 2\", ...]}]}"
                )
                questions_json = ask_gemini(question_prompt)
                
                # Extract JSON from the response if needed
                try:
                    # First attempt direct loading
                    st.session_state.questions = json.loads(questions_json)["questions"]
                except json.JSONDecodeError:
                    # Try to extract JSON with regex
                    import re
                    json_match = re.search(r"\{.*\}", questions_json, re.DOTALL)
                    if json_match:
                        try:
                            st.session_state.questions = json.loads(json_match.group(0))["questions"]
                        except:
                            # Use backup questions if all else fails
                            st.session_state.questions = self._get_backup_questions()
                    else:
                        st.session_state.questions = self._get_backup_questions()
        
        # Display current question
        q_index = st.session_state.current_question
        if q_index < len(st.session_state.questions) and not st.session_state.assessment_done:
            question = st.session_state.questions[q_index]
            st.write(f"**{question['question']}**")
            
            # Card-style choices
            selected = st.radio(
                "Select one answer:",
                question["choices"],
                index=None,
                key=f"q{q_index}"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Next Question" if q_index < len(st.session_state.questions) - 1 else "Complete Assessment"):
                    if selected:
                        st.session_state.responses.append({"question": question["question"], "selected": selected})
                        st.session_state.current_question += 1
                        
                        # Check if assessment is complete
                        if st.session_state.current_question >= len(st.session_state.questions):
                            st.session_state.assessment_done = True
                        
                        st.rerun()
                    else:
                        st.warning("Please select an answer before proceeding.")
        
        # Process completed assessment
        elif st.session_state.assessment_done:
            self._process_assessment_results()
    
    def _process_assessment_results(self):
        st.subheader("Your Assessment Results")
        
        with st.spinner("Analyzing your responses..."):
            # Prepare the prompt for evaluation
            assessment_prompt = (
                "Evaluate anxiety based on the following responses and return a JSON format:\n"
                f"{json.dumps(st.session_state.responses)}\n"
                "Response must be valid JSON with this exact format (no additional text): "
                "{\"score\": <integer_between_0_and_10>, \"classification\": \"<text_classification>\", "
                "\"summary\": \"<brief_explanation>\"}"
            )
            
            result = ask_gemini(assessment_prompt)
            
            try:
                # Try to extract JSON with regex first
                import re
                json_match = re.search(r"\{.*\}", result, re.DOTALL)
                
                if json_match:
                    assessment = json.loads(json_match.group(0))
                else:
                    # Fallback to direct parsing
                    assessment = json.loads(result)
                
                score = assessment.get("score", 5)
                classification = assessment.get("classification", "Moderate anxiety")
                summary = assessment.get("summary", "Based on your responses, you show some signs of anxiety.")
                
                # Save progress to database
                self.db.save_progress(
                    st.session_state["user"], 
                    score, 
                    classification,
                    summary
                )
                
                # Get user history for better recommendation
                user_history = self.db.get_user_progress(st.session_state["user"])
                
                # Get personalized suggestion
                suggestion, action_index = self.agent.get_suggestion(
                    classification, 
                    user_history=user_history
                )
                
                # Display results with visual elements
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### Anxiety Level: {classification}")
                    st.markdown(f"**Score:** {score}/10")
                    st.markdown(f"**Summary:** {summary}")
                
                with col2:
                    # Create gauge chart for score visualization
                    fig, ax = plt.subplots(figsize=(4, 4))
                    
                    # Create a gauge chart
                    ang_range = 180
                    pos = score/10 * ang_range - ang_range/2
                    
                    # Background arc
                    ax.add_patch(plt.matplotlib.patches.Arc((0.5, 0), 0.8, 0.8, 
                                                         theta1=180, theta2=0, 
                                                         color='lightgray', linewidth=20))
                    
                    # Colored arc for score
                    if score <= 3:
                        color = 'green'
                    elif score <= 6:
                        color = 'orange'
                    else:
                        color = 'red'
                        
                    ax.add_patch(plt.matplotlib.patches.Arc((0.5, 0), 0.8, 0.8, 
                                                         theta1=180, theta2=180-pos, 
                                                         color=color, linewidth=20))
                    
                    # Add needle
                    ax.arrow(0.5, 0, 0.35*np.cos(np.radians(180-pos)), 0.35*np.sin(np.radians(180-pos)),
                            head_width=0.05, head_length=0.05, fc=color, ec=color)
                    
                    # Add score text
                    ax.text(0.5, -0.2, f"{score}/10", ha='center', va='center', fontsize=20, fontweight='bold')
                    
                    # Hide axis
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    
                    st.pyplot(fig)
                
                # Display personalized recommendation
                st.markdown("## Your Personalized Recommendation")
                st.markdown(f"### {suggestion}")
                
                # User feedback section
                st.markdown("### Was this recommendation helpful?")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                feedback_value = None
                
                if col1.button("😞 Not at all", key="fb1"):
                    feedback_value = -1.0
                elif col2.button("🙁 Somewhat unhelpful", key="fb2"):
                    feedback_value = -0.5
                elif col3.button("😐 Neutral", key="fb3"):
                    feedback_value = 0.0
                elif col4.button("🙂 Somewhat helpful", key="fb4"):
                    feedback_value = 0.5
                elif col5.button("😊 Very helpful", key="fb5"):
                    feedback_value = 1.0
                
                # When processing feedback, use proper error handling
                if feedback_value is not None:
                    try:
                        # Save feedback
                        self.db.save_suggestion_feedback(
                            st.session_state["user"],
                            suggestion,
                            classification,
                            int(feedback_value * 2),  # Convert to -2 to 2 scale for storage
                            feedback_value
                        )
            
                        # Update the agent - ensure action_index is valid
                        if 0 <= action_index < len(self.agent.actions):
                            self.agent.process_feedback(
                                classification,
                                action_index,
                                feedback_value,
                                user_history
                            )
                
                            # Save updated model
                            model_weights = self.agent.save()
                            self.db.save_model_state("dqn", model_weights)
                
                            st.success("Thank you for your feedback! It helps us provide better recommendations.")
                        else:
                            st.warning("Your feedback was recorded, but there was an issue with the recommendation index.")
                            print(f"Invalid action_index: {action_index}, valid range: 0-{len(self.agent.actions)-1}")
            
                        # Option to start a new assessment
                        if st.button("Start a New Assessment"):
                            # Reset assessment state
                            st.session_state.current_question = 0
                            st.session_state.responses = []
                            st.session_state.assessment_done = False
                            st.rerun()
                
                    except Exception as e:
                        st.error(f"Error processing feedback: {str(e)}")
                        # Log the error details for debugging
                        print(f"Error details: action_index={action_index}, len(actions)={len(self.agent.actions)}")
                        st.warning("Your feedback was recorded, but there was an issue updating the recommendation system.")
            except Exception as e:
                st.error(f"Error processing feedback: {str(e)}")
                # Log the error details for debugging
                print(f"Error details: action_index={action_index}, len(actions)={len(self.agent.actions)}")
                st.warning("Your feedback was recorded, but there was an issue updating the recommendation system.")
    def _get_backup_questions(self):
        """Return backup questions if API fails"""
        return [
            {
                "question": "How often do you feel restless or on edge?",
                "choices": ["Never", "Rarely", "Sometimes", "Often", "Almost always"]
            },
            {
                "question": "How difficult is it for you to relax?",
                "choices": ["Not difficult at all", "Slightly difficult", "Moderately difficult", "Very difficult", "Extremely difficult"]
            },
            {
                "question": "How often do you worry too much about different things?",
                "choices": ["Never", "Rarely", "Sometimes", "Often", "Almost always"]
            },
            {
                "question": "How often do you experience physical symptoms like racing heart, sweating, or shortness of breath?",
                "choices": ["Never", "Rarely", "Sometimes", "Often", "Almost always"]
            },
            {
                "question": "How often do you avoid situations due to feelings of anxiety?",
                "choices": ["Never", "Rarely", "Sometimes", "Often", "Almost always"]
            }
        ]
    
    def show_dashboard(self):
        st.subheader("Your Anxiety Management Dashboard")
    
        try:
            # Get user data
            username = st.session_state["user"]
            user_progress = self.db.get_user_progress(username, limit=20)
        
            if not user_progress:
                st.info("You haven't completed any assessments yet. Take an assessment to see your dashboard.")
                if st.button("Take Assessment Now"):
                    st.session_state["choice"] = "Assessment"
                    st.rerun()
                return
        
            # Convert to dataframe for plotting with error handling
            try:
                df = pd.DataFrame(user_progress, columns=["score", "classification", "timestamp"])
            
                # Convert score to numeric if it's not already
                df["score"] = pd.to_numeric(df["score"], errors="coerce")
                # Drop any rows with NaN scores
                df = df.dropna(subset=["score"])
            
                if len(df) == 0:
                    st.warning("No valid score data available to display")
                    return
                
                # Convert timestamp to datetime with error handling
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"])
                df = df.sort_values("timestamp")
            
                # Line chart
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(df["timestamp"], df["score"], marker='o', linestyle='-', color='#5B9BD5')
            
                # Add trend line
                if len(df) >= 2:
                    z = np.polyfit(range(len(df)), df["score"], 1)
                    p = np.poly1d(z)
                    plt.plot(df["timestamp"], p(range(len(df))), "r--", alpha=0.7)
            
                plt.title("Your Anxiety Score Over Time", fontsize=16)
                plt.ylabel("Anxiety Score (0-10)", fontsize=12)
                plt.xlabel("Date", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 10)
            
                # Add appropriate annotations
                if len(df) >= 2:
                    if df["score"].iloc[-1] < df["score"].iloc[-2]:
                        plt.annotate("Improving!", 
                                xy=(df["timestamp"].iloc[-1], df["score"].iloc[-1]),
                                xytext=(10, -30), textcoords="offset points",
                                arrowprops=dict(arrowstyle="->", color="green"))
                    elif df["score"].iloc[-1] > df["score"].iloc[-2]:
                        plt.annotate("Let's work on this", 
                                xy=(df["timestamp"].iloc[-1], df["score"].iloc[-1]),
                                xytext=(10, 30), textcoords="offset points",
                                arrowprops=dict(arrowstyle="->", color="orange"))
            
                plt.tight_layout()
                st.pyplot(fig)
            
                # Recent assessments table
                st.markdown("### Recent Assessments")
            
                # Display in reverse order (most recent first)
                for i, (score, classification, timestamp) in enumerate(user_progress[:5]):
                    with st.expander(f"{classification} - {timestamp}", expanded=(i==0)):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            # Mini gauge chart
                            fig, ax = plt.subplots(figsize=(3, 3))
                            ax.add_patch(plt.matplotlib.patches.Wedge((0.5, 0.5), 0.3, 0, 360, color='lightgray'))
                            ax.add_patch(plt.matplotlib.patches.Wedge((0.5, 0.5), 0.3, 0, 360*(score/10), color='#5B9BD5'))
                            ax.text(0.5, 0.5, f"{score}/10", ha='center', va='center', fontsize=14, fontweight='bold')
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            ax.axis('off')
                            st.pyplot(fig)
                    
                        with col2:
                            st.markdown(f"**Score:** {score}/10")
                            st.markdown(f"**Classification:** {classification}")
                            st.markdown(f"**Date:** {timestamp}")
        
                with st.tabs:
                    st.markdown("### Anxiety Insights")
            
                # Time of day analysis if enough data
                if len(df) >= 3:
                    st.markdown("#### Time of Day Patterns")
                    df["hour"] = df["timestamp"].dt.hour
                
                    # Group by hour and get average score
                    hour_data = df.groupby(df["hour"]).mean().reset_index()
                
                    # Create time of day chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x="hour", y="score", data=hour_data, color="#5B9BD5", alpha=0.8)
                    plt.title("Average Anxiety Score by Time of Day", fontsize=14)
                    plt.xlabel("Hour of Day (24h format)", fontsize=12)
                    plt.ylabel("Average Anxiety Score", fontsize=12)
                    plt.xticks(range(0, 24, 3))
                    plt.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig)
                
                    # Peak anxiety times
                    peak_hour = hour_data.loc[hour_data["score"].idxmax()]
                    st.markdown(f"⚠️ **Peak anxiety time:** Around {int(peak_hour['hour']):02d}:00")
                
                    # Best times
                    if len(hour_data) > 1:
                        best_hour = hour_data.loc[hour_data["score"].idxmin()]
                        st.markdown(f"✅ **Lowest anxiety time:** Around {int(best_hour['hour']):02d}:00")
            
                # Classification distribution
                st.markdown("#### Anxiety Level Distribution")
                classification_counts = df["classification"].value_counts().reset_index()
                classification_counts.columns = ["classification", "count"]
            
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
                plt.pie(classification_counts["count"], 
                   labels=classification_counts["classification"],
                   autopct='%1.1f%%',
                   colors=colors,
                   startangle=90,
                   shadow=True)
                plt.axis('equal')
                st.pyplot(fig)
            
                # Trend analysis
                if len(df) >= 3:
                    st.markdown("#### Trend Analysis")
                
                    # Calculate linear regression for trend
                    x = range(len(df))
                    y = df["score"]
                    slope, intercept = np.polyfit(x, y, 1)
                
                    if slope < -0.1:
                        st.success("🎉 **Great progress!** Your anxiety scores show a significant improving trend.")
                    elif slope < 0:
                        st.info("👍 **Positive trend.** Your anxiety scores are gradually improving.")
                    elif slope < 0.1:
                        st.info("➡️ **Stable.** Your anxiety levels have been relatively consistent.")
                    else:
                        st.warning("⚠️ **Increasing trend.** Your anxiety scores have been rising. Consider trying new coping strategies.")
                
                    # Volatility
                    volatility = np.std(df["score"])
                    if volatility > 2:
                        st.warning(f"⚠️ **High variability detected** (score volatility: {volatility:.2f}). Your anxiety levels fluctuate considerably.")
                    else:
                        st.info(f"📊 **Score stability: {volatility:.2f}**. Your anxiety levels are relatively stable.")
        
                with st.tabs:
                    st.markdown("### Personalized Recommendations")
            
                    latest_score = df["score"].iloc[-1]
                    latest_classification = df["classification"].iloc[-1]
            
                    # Get best suggestions based on user history and state
                    best_suggestions = self.db.get_best_suggestions(username, latest_classification)
            
                    if best_suggestions:
                        st.markdown("#### Your Top Effective Strategies")
                        for suggestion, effectiveness in best_suggestions:
                            st.markdown(f"**{suggestion}** *(Effectiveness: {effectiveness:.2f})*")
            
                    # General recommendations based on current state
                    st.markdown("#### Recommended Strategies")
            
                    if latest_score <= 3:
                        strategies = [
                        "**Maintenance:** Continue your current practices that are working well",
                        "**Mindfulness:** Regular brief meditation to maintain awareness",
                        "**Exercise:** Maintain regular physical activity"
                        ]
                    elif latest_score <= 6:
                        strategies = [
                        "**Breathing techniques:** Practice 4-7-8 breathing when feeling stressed",
                        "**Cognitive reframing:** Notice and challenge negative thought patterns",
                        "**Regular breaks:** Schedule short breaks throughout your day",
                        "**Physical activity:** Aim for 30 minutes of moderate exercise daily"
                        ]
                    else:
                        strategies = [
                        "**Immediate relief:** Try the 5-4-3-2-1 grounding technique when anxiety spikes",
                        "**Professional support:** Consider speaking with a mental health professional",
                        "**Sleep hygiene:** Prioritize consistent, quality sleep",
                        "**Social connection:** Reach out to supportive friends or family",
                        "**Limit stimulants:** Reduce caffeine and other stimulants"
                        ]
            
                    for strategy in strategies:
                        st.markdown(f"• {strategy}")
            
                    # Get a new AI suggestion
                    st.markdown("#### Get a Fresh Recommendation")
                    if st.button("Generate New Strategy"):
                        with st.spinner("Creating personalized recommendation..."):
                            suggestion, _ = self.agent.get_suggestion(
                                latest_classification, 
                                user_history=user_progress
                            )
                            st.success(f"**Try this:** {suggestion}")
            
            except Exception as e:
                st.error(f"Error displaying dashboard data: {str(e)}")
                st.info("Try completing a new assessment to refresh your data.")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.") 
    
    def show_settings(self):
        st.subheader("Account Settings")
        
        if "user" not in st.session_state:
            st.warning("Please log in to access settings.")
            return
        
        username = st.session_state["user"]
        
        # Create tabs for different settings
        tab1, tab2, tab3 = st.tabs(["Profile", "Notifications", "Data Management"])
        
        with tab1:
            st.markdown("### Profile Settings")
            
            # Dummy fields for demonstration
            email = st.text_input("Email Address", "user@example.com")
            password = st.text_input("New Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            
            if st.button("Update Profile"):
                st.success("Profile updated successfully!")
        
        with tab2:
            st.markdown("### Notification Preferences")
            
            st.checkbox("Daily reminders", value=True)
            st.checkbox("Weekly progress report", value=True)
            st.checkbox("Tips and suggestions", value=True)
            
            reminder_time = st.time_input("Reminder Time", datetime.time(20, 0))
            
            if st.button("Save Notification Settings"):
                st.success("Notification preferences saved!")
        
        with tab3:
            st.markdown("### Data Management")
            
            st.markdown("""
            Your data is stored securely and used only to provide personalized anxiety management recommendations.
            """)
            
            if st.button("Export My Data"):
                # Create a downloadable CSV
                user_progress = self.db.get_user_progress(username, limit=1000)
                if user_progress:
                    df = pd.DataFrame(user_progress, columns=["score", "classification", "timestamp"])
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        "Download CSV",
                        csv,
                        "anxiety_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.info("No data available to export.")
            
            st.warning("Danger Zone")
            if st.button("Delete All My Data", type="primary"):
                st.error("This action cannot be undone!")
                # This would need confirmation in a real app

# Main Application
def main():
    st.set_page_config(page_title="Advanced Anxiety Assessment System", page_icon="🧘", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton button {
        border-radius: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stProgress > div > div {
        background-color: #5B9BD5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize database and agent
    db = Database()
    
    # Create and initialize the agent
    agent = DeepQLearningAgent()
    
    # Try to load existing model weights
    model_weights = db.load_latest_model_state("dqn")
    if model_weights:
        agent.load(model_weights)
    
    # Create UI manager
    ui = UIManager(db, agent)
    
    # App header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🧘 Advanced Anxiety Assessment System")
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Logo", width=150)
        st.markdown("## Navigation")
        
        if "user" in st.session_state:
            st.success(f"Logged in as: {st.session_state['user']}")
            options = ["Dashboard", "Assessment", "Journal", "Resources", "Settings", "Logout"]
        else:
            options = ["Login", "Register"]
        
        choice = st.radio("Go to:", options)
        st.session_state["choice"] = choice
        
        if "user" in st.session_state and choice != "Logout":
            # Quick access to assessment
            if st.button("Quick Assessment"):
                st.session_state["choice"] = "Assessment"
                st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This advanced anxiety assessment system uses deep reinforcement learning to provide personalized coping strategies based on your specific needs and feedback.
        """)
    
    # Main content area
    if "choice" not in st.session_state:
        st.session_state["choice"] = "Login" if "user" not in st.session_state else "Dashboard"
    
    choice = st.session_state["choice"]
    
    if choice == "Login":
        ui.show_login_page()
    elif choice == "Register":
        ui.show_register_page()
    elif choice == "Logout":
        st.session_state.pop("user", None)
        st.session_state["choice"] = "Login"
        st.success("You have been logged out.")
        st.rerun()
    elif choice == "Dashboard":
        ui.show_dashboard()
    elif choice == "Assessment":
        ui.show_assessment_page()
    elif choice == "Settings":
        ui.show_settings()
    elif choice == "Journal":
        st.subheader("Anxiety Journal")
        st.markdown("Coming soon! Track your daily moods and triggers.")
    elif choice == "Resources":
        st.subheader("Anxiety Resources")
        st.markdown("Coming soon! Access helpful articles and guides.")

if __name__ == "__main__":
    main()
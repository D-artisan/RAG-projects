import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Learning Quest 🤖",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .level-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .quest-card {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .achievement {
        background: #ffd700;
        color: #333;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class AILearningGame:
    def __init__(self):
        # Debug environment loading
        print("Loading environment variables...")
        load_dotenv()  # Load again to be sure


        # Always use session_state override if present
        if 'ollama_url' in st.session_state and st.session_state.ollama_url:
            self.ollama_url = st.session_state.ollama_url
        else:
            env_url = os.getenv("OLLAMA_API_URL")
            if not env_url:
                env_url = os.getenv("OLLAMA_BASE_URL")
                if env_url and not env_url.endswith("/api/generate"):
                    env_url += "/api/generate"
            self.ollama_url = env_url
            st.session_state.ollama_url = self.ollama_url

        if 'ollama_model' in st.session_state and st.session_state.ollama_model:
            self.ollama_model = st.session_state.ollama_model
        else:
            env_model = os.getenv("OLLAMA_MODEL")
            self.ollama_model = env_model
            st.session_state.ollama_model = self.ollama_model

        print(f"Final OLLAMA_API_URL: {self.ollama_url}")
        print(f"Final OLLAMA_MODEL: {self.ollama_model}")

        # Initialize session state
        if 'level' not in st.session_state:
            st.session_state.level = 1
        if 'score' not in st.session_state:
            st.session_state.score = 0
        if 'achievements' not in st.session_state:
            st.session_state.achievements = []
        if 'completed_quests' not in st.session_state:
            st.session_state.completed_quests = set()
        if 'current_quest' not in st.session_state:
            st.session_state.current_quest = None
    
    def query_ollama(self, prompt):
        """Query Ollama model for explanations"""
        if not self.ollama_url or not self.ollama_model:
            return "⚠️ Ollama not configured. Please check your .env file with OLLAMA_API_URL and OLLAMA_MODEL."
        
        try:
            # Add headers for ngrok tunnels
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Add ngrok-skip-browser-warning header if using ngrok
            if 'ngrok' in self.ollama_url:
                headers['ngrok-skip-browser-warning'] = 'true'
            
            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
            
            # Increase timeout for remote connections
            response = requests.post(self.ollama_url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response available")
            else:
                return f"⚠️ Ollama API Error (Status {response.status_code}): {response.text[:200]}..."
                
        except requests.exceptions.ConnectionError:
            return f"""🔌 **Connection Error**: Cannot reach Ollama at `{self.ollama_url}`
            
**Troubleshooting:**
- If using ngrok: Make sure your ngrok tunnel is active
- If using local Ollama: Start Ollama with `ollama serve`
- Check your .env file has the correct OLLAMA_API_URL
            
**Current config:**
- URL: {self.ollama_url}
- Model: {self.ollama_model}
            """
        except requests.exceptions.Timeout:
            return "⏱️ **Timeout Error**: Ollama took too long to respond. Try again or check your connection."
        except Exception as e:
            return f"❌ **Unexpected Error**: {str(e)}"
    
    def add_achievement(self, achievement):
        """Add achievement to user's collection"""
        if achievement not in st.session_state.achievements:
            st.session_state.achievements.append(achievement)
            st.success(f"🏆 Achievement Unlocked: {achievement}")
    
    def add_score(self, points):
        """Add points to user's score"""
        st.session_state.score += points
        
        # Level up system - more reasonable progression
        required_points = st.session_state.level * 75  # Reduced from 100 to 75
        if st.session_state.score >= required_points:
            old_level = st.session_state.level
            st.session_state.level += 1
            st.balloons()
            st.success(f"🎉 Level Up! You're now Level {st.session_state.level}")
            
            # Show what new quests are available
            if st.session_state.level == 2:
                st.info("🎯 New quest unlocked: ML vs Deep Learning Quiz!")
            elif st.session_state.level == 3:
                st.info("🎯 New quests unlocked: Supervised & Unsupervised Learning!")
            elif st.session_state.level == 4:
                st.info("🎯 New quest unlocked: Neural Network Fundamentals!")
            elif st.session_state.level == 5:
                st.info("🎯 New quests unlocked: Advanced Deep Learning & AI Ethics!")
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 AI Learning Quest</h1>
            <p>Journey from Zero to AI Hero!</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with user progress"""
        st.sidebar.markdown("### 👤 Player Stats")
        st.sidebar.metric("Level", st.session_state.level)
        st.sidebar.metric("Score", st.session_state.score)
        st.sidebar.metric("Quests Completed", len(st.session_state.completed_quests))
        
        # Show progress to next level
        current_level = st.session_state.level
        required_for_next = current_level * 75
        if st.session_state.score < required_for_next:
            points_needed = required_for_next - st.session_state.score
            st.sidebar.progress(st.session_state.score / required_for_next)
            st.sidebar.caption(f"Next level: {points_needed} points needed")
        else:
            st.sidebar.success("✨ Ready to level up!")
        
        st.sidebar.markdown("### 🏆 Achievements")
        for achievement in st.session_state.achievements:
            st.sidebar.markdown(f'<div class="achievement">{achievement}</div>', unsafe_allow_html=True)
        
        if not st.session_state.achievements:
            st.sidebar.write("No achievements yet. Complete quests to earn them!")
        
        # Debug panel (can be removed later)
        with st.sidebar.expander("🔧 Debug Info"):
            st.write(f"Current Level: {st.session_state.level}")
            st.write(f"Current Score: {st.session_state.score}")
            st.write(f"Points for next level: {st.session_state.level * 75}")
            st.write(f"Completed quests: {list(st.session_state.completed_quests)}")
            
            # Environment variables debug
            st.markdown("**🔍 Environment Variables:**")
            env_vars = {
                "OLLAMA_API_URL": os.getenv("OLLAMA_API_URL"),
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
                "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL")
            }
            for key, value in env_vars.items():
                if value:
                    st.caption(f"{key}: {value}")
                else:
                    st.caption(f"{key}: ❌ Not set")
            
            # Ollama connection status
            st.markdown("**🤖 Ollama Config:**")
            if self.game.ollama_url and self.game.ollama_model:
                st.success("✅ Configured")
                st.caption(f"URL: {self.game.ollama_url}")
                st.caption(f"Model: {self.game.ollama_model}")
                
                # Test connection button
                if st.button("🔍 Test Connection"):
                    with st.spinner("Testing..."):
                        test_result = self.game.query_ollama("Hello! Just testing the connection.")
                        if "Connection Error" in test_result or "Error" in test_result:
                            st.error("❌ Connection failed")
                            st.caption(test_result[:100])
                        else:
                            st.success("✅ Connection successful!")
            else:
                st.error("❌ Not configured")
                st.caption("Check your .env file")
            
            # Manual override section
            st.markdown("**⚙️ Manual Override:**")
            manual_url = st.text_input("Ollama URL", value=self.game.ollama_url or "", 
                                     placeholder="https://your-ngrok-url.ngrok-free.app/api/generate")
            manual_model = st.text_input("Model Name", value=self.game.ollama_model or "", 
                                       placeholder="gemma3:27b")
            
            if st.button("💾 Update Config"):
                if manual_url and manual_model:
                    st.session_state.ollama_url = manual_url
                    st.session_state.ollama_model = manual_model
                    st.success("✅ Configuration updated!")
                    st.rerun()
                else:
                    st.error("Please fill in both URL and model name")
            
            if st.button("🚀 Force Level Up", help="Use if stuck"):
                st.session_state.level += 1
                st.success("Level boosted!")
                st.rerun()

class QuestManager:
    def __init__(self, game):
        self.game = game
        self.quests = self.define_quests()
    
    def define_quests(self):
        """Define all learning quests"""
        return {
            # Beginner Level (1-2)
            "intro_ai": {
                "title": "What is AI? 🤖",
                "level": 1,
                "description": "Learn the basics of Artificial Intelligence",
                "type": "theory",
                "points": 50
            },
            "data_basics": {
                "title": "Understanding Data 📊",
                "level": 1,
                "description": "Learn about different types of data and their importance",
                "type": "interactive",
                "points": 75
            },
            "ml_vs_dl": {
                "title": "ML vs Deep Learning 🧠",
                "level": 2,
                "description": "Understand the difference between Machine Learning and Deep Learning",
                "type": "quiz",
                "points": 100
            },
            
            # Intermediate Level (3-4)
            "supervised_learning": {
                "title": "Supervised Learning Adventure 📚",
                "level": 3,
                "description": "Explore supervised learning with hands-on examples",
                "type": "hands_on",
                "points": 150
            },
            "unsupervised_learning": {
                "title": "Unsupervised Learning Mystery 🔍",
                "level": 3,
                "description": "Discover patterns in data without labels",
                "type": "hands_on",
                "points": 150
            },
            "neural_network_basics": {
                "title": "Neural Network Fundamentals ⚡",
                "level": 4,
                "description": "Build your first neural network",
                "type": "hands_on",
                "points": 200
            },
            
            # Advanced Level (5+)
            "deep_learning_architectures": {
                "title": "Deep Learning Architectures 🏗️",
                "level": 5,
                "description": "Explore different neural network architectures",
                "type": "advanced",
                "points": 250
            },
            "ai_ethics": {
                "title": "AI Ethics & Responsibility 🤝",
                "level": 5,
                "description": "Understand the ethical implications of AI",
                "type": "theory",
                "points": 200
            }
        }
    
    def get_available_quests(self):
        """Get quests available for current level"""
        available = []
        for quest_id, quest in self.quests.items():
            if (quest["level"] <= st.session_state.level and 
                quest_id not in st.session_state.completed_quests):
                available.append((quest_id, quest))
        return available
    
    def render_quest_selection(self):
        """Render quest selection interface"""
        st.markdown("## 🗺️ Available Quests")
        
        available_quests = self.get_available_quests()
        
        if not available_quests:
            st.success("🎉 Congratulations! You've completed all available quests!")
            return
        
        # Show available quests
        for quest_id, quest in available_quests:
            with st.expander(f"{'⭐' * quest['level']} {quest['title']} (Level {quest['level']})"):
                st.write(quest['description'])
                st.write(f"**Points:** {quest['points']}")
                st.write(f"**Type:** {quest['type'].title()}")
                
                if st.button(f"Start Quest: {quest['title']}", key=f"start_{quest_id}"):
                    st.session_state.current_quest = quest_id
                    st.rerun()
        
        # Show locked quests for motivation
        st.markdown("### 🔒 Locked Quests (Level up to unlock!)")
        locked_quests = []
        for quest_id, quest in self.quests.items():
            if (quest["level"] > st.session_state.level and 
                quest_id not in st.session_state.completed_quests):
                locked_quests.append((quest_id, quest))
        
        if locked_quests:
            for quest_id, quest in locked_quests[:3]:  # Show first 3 locked quests
                st.markdown(f"🔒 **{quest['title']}** (Level {quest['level']}) - {quest['description']}")
        else:
            st.info("🎉 All quests are unlocked! Complete the available ones above.")

class QuestRenderer:
    def __init__(self, game):
        self.game = game
    
    def render_current_quest(self, quest_id, quest_data):
        """Render the current quest based on its type"""
        st.markdown(f"## 🎯 Current Quest: {quest_data['title']}")
        
        if quest_data['type'] == 'theory':
            self.render_theory_quest(quest_id, quest_data)
        elif quest_data['type'] == 'interactive':
            self.render_interactive_quest(quest_id, quest_data)
        elif quest_data['type'] == 'quiz':
            self.render_quiz_quest(quest_id, quest_data)
        elif quest_data['type'] == 'hands_on':
            self.render_hands_on_quest(quest_id, quest_data)
        elif quest_data['type'] == 'advanced':
            self.render_advanced_quest(quest_id, quest_data)
    
    def render_theory_quest(self, quest_id, quest_data):
        """Render theory-based quests"""
        if quest_id == "intro_ai":
            st.markdown("""
            ### Welcome to the World of AI! 🌟
            
            Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.
            
            **Key Concepts:**
            - **Intelligence**: The ability to learn, reason, and solve problems
            - **Machine Learning**: A subset of AI that learns from data
            - **Deep Learning**: A subset of ML inspired by the human brain
            - **Applications**: From recommendation systems to autonomous vehicles
            """)
            
            st.info("💡 **Fun Fact**: The term 'Artificial Intelligence' was first coined in 1956!")
            
            # Interactive AI explanation
            st.markdown("### 🤖 Ask AI to Explain More!")
            
            if self.game.ollama_url and self.game.ollama_model:
                if st.button("Get AI Explanation"):
                    with st.spinner("AI is thinking..."):
                        explanation = self.game.query_ollama(
                            "Explain artificial intelligence in simple terms for a beginner. "
                            "Include examples and why it's important in today's world."
                        )
                        st.markdown(explanation)
            else:
                st.info("💡 **AI explanations require Ollama to be configured.** You can still complete this quest without it!")
                st.markdown("""
                **Alternative Learning:**
                - AI is like teaching computers to think and make decisions
                - Examples: Netflix recommendations, voice assistants, autonomous cars
                - Important because it can solve complex problems and automate tasks
                - The future of technology and innovation
                """)
        
        elif quest_id == "ai_ethics":
            st.markdown("""
            ### AI Ethics & Responsibility 🤝
            
            As AI becomes more powerful, we must consider its ethical implications.
            
            **Key Ethical Considerations:**
            - **Bias**: AI systems can perpetuate human biases
            - **Privacy**: How data is collected and used
            - **Transparency**: Understanding how AI makes decisions
            - **Accountability**: Who is responsible for AI decisions?
            - **Job Impact**: How AI affects employment
            """)
            
            st.warning("⚠️ **Remember**: With great AI power comes great responsibility!")
        
        # Complete quest button
        if st.button("Complete Quest", key=f"complete_{quest_id}"):
            self.complete_quest(quest_id, quest_data)
    
    def render_interactive_quest(self, quest_id, quest_data):
        """Render interactive quests with visualizations"""
        if quest_id == "data_basics":
            st.markdown("### 📊 Understanding Data Types")
            
            # Interactive data type explorer
            data_type = st.selectbox("Choose a data type to explore:", 
                                   ["Numerical", "Categorical", "Text", "Image", "Time Series"])
            
            if data_type == "Numerical":
                st.markdown("**Numerical Data**: Numbers that can be measured or counted")
                # Generate sample numerical data
                data = np.random.normal(50, 15, 100)
                fig = px.histogram(x=data, title="Sample Numerical Data Distribution")
                st.plotly_chart(fig)
                
            elif data_type == "Categorical":
                st.markdown("**Categorical Data**: Data that can be divided into categories")
                categories = ['Red', 'Blue', 'Green', 'Yellow']
                values = [23, 45, 56, 78]
                fig = px.pie(values=values, names=categories, title="Sample Categorical Data")
                st.plotly_chart(fig)
                
            elif data_type == "Text":
                st.markdown("**Text Data**: Unstructured text that can be analyzed")
                sample_text = "Artificial Intelligence is revolutionizing the world!"
                st.code(sample_text)
                st.write("Text can be processed for sentiment, topics, and more!")
                
            elif data_type == "Image":
                st.markdown("**Image Data**: Visual data represented as pixels")
                # Create a simple pattern
                img_data = np.random.rand(50, 50, 3)
                st.image(img_data, caption="Sample Image Data (Random Pixels)")
                
            elif data_type == "Time Series":
                st.markdown("**Time Series Data**: Data points collected over time")
                dates = pd.date_range('2024-01-01', periods=100)
                values = np.cumsum(np.random.randn(100))
                fig = px.line(x=dates, y=values, title="Sample Time Series Data")
                st.plotly_chart(fig)
            
            if st.button("I understand data types!", key=f"complete_{quest_id}"):
                self.complete_quest(quest_id, quest_data)
    
    def render_quiz_quest(self, quest_id, quest_data):
        """Render quiz-based quests"""
        if quest_id == "ml_vs_dl":
            st.markdown("### 🧠 ML vs Deep Learning Quiz")
            
            questions = [
                {
                    "question": "What is the main difference between ML and Deep Learning?",
                    "options": [
                        "ML uses algorithms, DL doesn't",
                        "DL is a subset of ML that uses neural networks",
                        "ML is newer than DL",
                        "They are the same thing"
                    ],
                    "correct": 1
                },
                {
                    "question": "Which requires more data to work effectively?",
                    "options": ["Machine Learning", "Deep Learning", "Both equally", "Neither"],
                    "correct": 1
                },
                {
                    "question": "What is a neural network inspired by?",
                    "options": ["Computer circuits", "The human brain", "Mathematical formulas", "Database structures"],
                    "correct": 1
                }
            ]
            
            # Initialize quiz state
            quiz_key = f"{quest_id}_quiz"
            if quiz_key not in st.session_state:
                st.session_state[quiz_key] = {
                    'answers_checked': [False] * len(questions),
                    'correct_answers': [False] * len(questions),
                    'all_answered': False
                }
            
            quiz_state = st.session_state[quiz_key]
            
            for i, q in enumerate(questions):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                answer = st.radio(f"Select your answer for question {i+1}:", 
                                q['options'], key=f"q{i}_{quest_id}")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button(f"Check Answer {i+1}", key=f"check_{i}_{quest_id}"):
                        quiz_state['answers_checked'][i] = True
                        if q['options'].index(answer) == q['correct']:
                            quiz_state['correct_answers'][i] = True
                            st.success("✅ Correct!")
                        else:
                            quiz_state['correct_answers'][i] = False
                            st.error(f"❌ Wrong! Correct answer: {q['options'][q['correct']]}")
                
                with col2:
                    if quiz_state['answers_checked'][i]:
                        if quiz_state['correct_answers'][i]:
                            st.success("✅ Answered correctly!")
                        else:
                            st.error("❌ Incorrect answer")
            
            # Check if all questions are answered correctly
            all_correct = all(quiz_state['correct_answers']) and all(quiz_state['answers_checked'])
            
            if all_correct:
                st.success("🎉 Perfect! You answered all questions correctly!")
                if st.button("Complete Quiz Quest", key=f"complete_{quest_id}"):
                    self.complete_quest(quest_id, quest_data)
                    self.game.add_achievement("Quiz Master")
                    # Clear quiz state
                    del st.session_state[quiz_key]
            elif all(quiz_state['answers_checked']):
                st.warning("⚠️ You need to answer all questions correctly to complete the quest. Try again!")
                if st.button("Reset Quiz", key=f"reset_{quest_id}"):
                    st.session_state[quiz_key] = {
                        'answers_checked': [False] * len(questions),
                        'correct_answers': [False] * len(questions),
                        'all_answered': False
                    }
                    st.rerun()
            else:
                st.info("📝 Answer all questions by clicking 'Check Answer' for each one.")
    
    def render_hands_on_quest(self, quest_id, quest_data):
        """Render hands-on coding quests"""
        if quest_id == "supervised_learning":
            st.markdown("### 📚 Supervised Learning Adventure")
            
            st.markdown("""
            In supervised learning, we learn from labeled examples to make predictions on new data.
            Let's explore this with a classification problem!
            """)
            
            # Generate sample data
            if st.button("Generate Classification Data"):
                X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                         n_informative=2, random_state=42, n_clusters_per_class=1)
                
                # Store in session state
                st.session_state.X = X
                st.session_state.y = y
                
                # Visualize data
                fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y, 
                               title="Classification Data", 
                               labels={'color': 'Class'})
                st.plotly_chart(fig)
            
            if 'X' in st.session_state:
                if st.button("Train Model"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state.X, st.session_state.y, test_size=0.3, random_state=42)
                    
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    
                    accuracy = model.score(X_test, y_test)
                    st.success(f"Model trained! Accuracy: {accuracy:.2%}")
                    
                    if accuracy > 0.8:
                        if st.button("Complete Supervised Learning Quest", key=f"complete_{quest_id}"):
                            self.complete_quest(quest_id, quest_data)
                            self.game.add_achievement("Data Scientist")
        
        elif quest_id == "unsupervised_learning":
            st.markdown("### 🔍 Unsupervised Learning Mystery")
            
            st.markdown("""
            In unsupervised learning, we find hidden patterns in data without labels.
            Let's explore clustering!
            """)
            
            if st.button("Generate Mystery Data"):
                X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
                st.session_state.cluster_X = X
                
                fig = px.scatter(x=X[:, 0], y=X[:, 1], title="Mystery Data - Can you find the patterns?")
                st.plotly_chart(fig)
            
            if 'cluster_X' in st.session_state:
                n_clusters = st.slider("How many clusters do you think there are?", 2, 8, 4)
                
                if st.button("Apply Clustering"):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(st.session_state.cluster_X)
                    
                    fig = px.scatter(x=st.session_state.cluster_X[:, 0], 
                                   y=st.session_state.cluster_X[:, 1], 
                                   color=clusters,
                                   title=f"Clustering Results with {n_clusters} clusters")
                    st.plotly_chart(fig)
                    
                    if n_clusters == 4:  # Correct number
                        st.success("🎉 Perfect! You found the right number of clusters!")
                        if st.button("Complete Unsupervised Learning Quest", key=f"complete_{quest_id}"):
                            self.complete_quest(quest_id, quest_data)
                            self.game.add_achievement("Pattern Detective")
        
        elif quest_id == "neural_network_basics":
            st.markdown("### ⚡ Neural Network Fundamentals")
            
            st.markdown("""
            A neural network is inspired by the human brain and consists of interconnected nodes (neurons).
            Let's build a simple neural network visualization!
            """)
            
            # Neural network architecture selector
            col1, col2, col3 = st.columns(3)
            with col1:
                input_neurons = st.number_input("Input Neurons", 2, 10, 3)
            with col2:
                hidden_neurons = st.number_input("Hidden Neurons", 2, 10, 4)
            with col3:
                output_neurons = st.number_input("Output Neurons", 1, 5, 2)
            
            if st.button("Visualize Neural Network"):
                # Create a simple network visualization
                fig = go.Figure()
                
                # Input layer
                for i in range(input_neurons):
                    fig.add_trace(go.Scatter(x=[0], y=[i], mode='markers', 
                                           marker=dict(size=20, color='lightblue'),
                                           name=f'Input {i+1}', showlegend=False))
                
                # Hidden layer
                for i in range(hidden_neurons):
                    fig.add_trace(go.Scatter(x=[1], y=[i], mode='markers', 
                                           marker=dict(size=20, color='lightgreen'),
                                           name=f'Hidden {i+1}', showlegend=False))
                
                # Output layer
                for i in range(output_neurons):
                    fig.add_trace(go.Scatter(x=[2], y=[i], mode='markers', 
                                           marker=dict(size=20, color='lightcoral'),
                                           name=f'Output {i+1}', showlegend=False))
                
                # Add connections (simplified)
                for i in range(input_neurons):
                    for j in range(hidden_neurons):
                        fig.add_trace(go.Scatter(x=[0, 1], y=[i, j], mode='lines',
                                               line=dict(color='gray', width=1),
                                               showlegend=False))
                
                for i in range(hidden_neurons):
                    for j in range(output_neurons):
                        fig.add_trace(go.Scatter(x=[1, 2], y=[i, j], mode='lines',
                                               line=dict(color='gray', width=1),
                                               showlegend=False))
                
                fig.update_layout(title="Your Neural Network Architecture",
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                
                st.plotly_chart(fig)
                
                st.success("🧠 Great! You've designed your first neural network!")
                if st.button("Complete Neural Network Quest", key=f"complete_{quest_id}"):
                    self.complete_quest(quest_id, quest_data)
                    self.game.add_achievement("Neural Architect")
    
    def render_advanced_quest(self, quest_id, quest_data):
        """Render advanced quests"""
        if quest_id == "deep_learning_architectures":
            st.markdown("### 🏗️ Deep Learning Architectures")
            
            architectures = {
                "Convolutional Neural Networks (CNN)": {
                    "description": "Excellent for image processing and computer vision",
                    "use_cases": ["Image classification", "Object detection", "Medical imaging"],
                    "key_features": ["Convolutional layers", "Pooling layers", "Feature maps"]
                },
                "Recurrent Neural Networks (RNN)": {
                    "description": "Great for sequential data and time series",
                    "use_cases": ["Natural language processing", "Speech recognition", "Time series prediction"],
                    "key_features": ["Memory cells", "Sequential processing", "Hidden states"]
                },
                "Transformer Networks": {
                    "description": "State-of-the-art for language understanding",
                    "use_cases": ["Machine translation", "Text generation", "Chatbots"],
                    "key_features": ["Attention mechanism", "Parallel processing", "Self-attention"]
                }
            }
            
            selected_arch = st.selectbox("Choose an architecture to explore:", 
                                       list(architectures.keys()))
            
            arch_info = architectures[selected_arch]
            st.markdown(f"**{selected_arch}**")
            st.write(arch_info['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Use Cases:**")
                for use_case in arch_info['use_cases']:
                    st.write(f"• {use_case}")
            
            with col2:
                st.markdown("**Key Features:**")
                for feature in arch_info['key_features']:
                    st.write(f"• {feature}")
            
            # AI explanation
            if st.button(f"Get AI explanation of {selected_arch}"):
                if self.game.ollama_url and self.game.ollama_model:
                    with st.spinner("AI is explaining..."):
                        explanation = self.game.query_ollama(
                            f"Explain {selected_arch} in detail for someone learning deep learning. "
                            f"Include how it works, why it's useful, and provide simple examples."
                        )
                        st.markdown(explanation)
                else:
                    st.info("💡 **AI explanations require Ollama to be configured.**")
                    
                    # Provide fallback explanations
                    fallback_explanations = {
                        "Convolutional Neural Networks (CNN)": """
                        **CNNs work by:**
                        - Using filters to detect features in images (like edges, shapes)
                        - Pooling layers reduce image size while keeping important info
                        - Multiple layers learn increasingly complex features
                        - Great for image recognition, medical imaging, self-driving cars
                        """,
                        "Recurrent Neural Networks (RNN)": """
                        **RNNs work by:**
                        - Having memory to remember previous inputs
                        - Processing sequences one step at a time
                        - Good for text, speech, and time-series data
                        - Examples: language translation, speech recognition, stock prediction
                        """,
                        "Transformer Networks": """
                        **Transformers work by:**
                        - Using attention mechanisms to focus on important parts
                        - Processing all data simultaneously (not sequentially)
                        - The foundation of modern AI like GPT and BERT
                        - Examples: ChatGPT, Google Translate, text summarization
                        """
                    }
                    st.markdown(fallback_explanations.get(selected_arch, "No explanation available."))
            
            if st.button("Complete Architecture Quest", key=f"complete_{quest_id}"):
                self.complete_quest(quest_id, quest_data)
                self.game.add_achievement("Architecture Master")
    
    def complete_quest(self, quest_id, quest_data):
        """Complete a quest and update user progress"""
        st.session_state.completed_quests.add(quest_id)
        st.session_state.current_quest = None
        self.game.add_score(quest_data['points'])
        st.success(f"🎉 Quest Complete! +{quest_data['points']} points")
        
        # Check for special achievements
        if len(st.session_state.completed_quests) == 1:
            self.game.add_achievement("First Steps")
        elif len(st.session_state.completed_quests) == 5:
            self.game.add_achievement("Learning Enthusiast")
        elif len(st.session_state.completed_quests) == 8:
            self.game.add_achievement("AI Master")
        
        st.rerun()

def main():
    # Initialize the game
    game = AILearningGame()
    quest_manager = QuestManager(game)
    quest_renderer = QuestRenderer(game)
    
    # Render header and sidebar
    game.render_header()
    game.render_sidebar()
    
    # Main content area
    if st.session_state.current_quest:
        # Render current quest
        quest_data = quest_manager.quests[st.session_state.current_quest]
        quest_renderer.render_current_quest(st.session_state.current_quest, quest_data)
        
        # Back to quests button
        if st.button("← Back to Quest Selection"):
            st.session_state.current_quest = None
            st.rerun()
    else:
        # Render quest selection
        quest_manager.render_quest_selection()
        
        # Show progress summary
        st.markdown("## 📈 Your Progress")
        
        total_quests = len(quest_manager.quests)
        completed_quests = len(st.session_state.completed_quests)
        progress = completed_quests / total_quests if total_quests > 0 else 0
        
        st.progress(progress)
        st.write(f"Completed: {completed_quests}/{total_quests} quests ({progress:.1%})")
        
        # Learning resources
        with st.expander("📚 Additional Learning Resources"):
            st.markdown("""
            ### Recommended Resources:
            - **Books**: "Hands-On Machine Learning" by Aurélien Géron
            - **Online Courses**: Coursera Machine Learning Course by Andrew Ng
            - **Websites**: Kaggle Learn, fast.ai
            - **Practice**: Kaggle competitions, Google Colab notebooks
            - **Communities**: Reddit r/MachineLearning, Stack Overflow
            """)

if __name__ == "__main__":
    main()

# 🤖 AI Learning Quest

An interactive Streamlit game designed to take you from zero understanding of Machine Learning, Deep Learning, Neural Networks, and AI to pro-level understanding!

## 🎯 Features

- **Progressive Learning**: Start from basics and advance through levels
- **Interactive Quests**: Hands-on coding, visualizations, and quizzes
- **AI-Powered Explanations**: Get personalized explanations using Ollama
- **Achievement System**: Unlock achievements as you progress
- **Multiple Quest Types**:
  - Theory lessons with AI explanations
  - Interactive data visualizations
  - Coding challenges
  - Quizzes to test knowledge
  - Advanced architecture exploration

## 🎮 Game Levels & Quests

### Beginner (Level 1-2)
- **What is AI?** - Introduction to Artificial Intelligence
- **Understanding Data** - Learn about different data types
- **ML vs Deep Learning** - Quiz on fundamental differences

### Intermediate (Level 3-4)
- **Supervised Learning Adventure** - Hands-on classification
- **Unsupervised Learning Mystery** - Discover patterns with clustering
- **Neural Network Fundamentals** - Build your first neural network

### Advanced (Level 5+)
- **Deep Learning Architectures** - Explore CNNs, RNNs, and Transformers
- **AI Ethics & Responsibility** - Understanding ethical implications

## 🚀 Quick Start

### Method 1: Using the Launcher (Recommended)
```bash
python run_game.py
```

### Method 2: Direct Streamlit
```bash
streamlit run ai_learning_game.py
```

## ⚙️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Update `.env` file with your Ollama settings:
   ```
   OLLAMA_API_URL=http://localhost:11434/api/generate
   OLLAMA_MODEL=llama3.2
   ```

3. **Start Ollama** (if using local Ollama):
   ```bash
   ollama serve
   ```

## 🎲 How to Play

1. **Start the Game**: Run the launcher or streamlit command
2. **Check Your Stats**: View your level, score, and achievements in the sidebar
3. **Select Quests**: Choose from available quests based on your current level
4. **Complete Challenges**: Engage with interactive content, coding exercises, and quizzes
5. **Earn Points**: Complete quests to gain points and level up
6. **Unlock Achievements**: Special accomplishments for reaching milestones
7. **Get AI Help**: Use the AI explanations feature for deeper understanding

## 🏆 Achievement System

- **First Steps**: Complete your first quest
- **Quiz Master**: Ace a quiz without mistakes
- **Data Scientist**: Successfully train a model
- **Pattern Detective**: Find the right number of clusters
- **Neural Architect**: Design a neural network
- **Architecture Master**: Complete advanced architecture quest
- **Learning Enthusiast**: Complete 5 quests
- **AI Master**: Complete all available quests

## 🛠️ Technical Features

- **Streamlit Interface**: Modern, responsive web interface
- **Ollama Integration**: AI-powered explanations and help
- **Interactive Visualizations**: Plotly charts and graphs
- **Real Machine Learning**: Actual scikit-learn implementations
- **Progressive Complexity**: Gradually increasing difficulty
- **Session State**: Progress is maintained during your session

## 📚 Learning Path

The game follows a structured learning path:

1. **Foundation**: Understanding what AI/ML is
2. **Data Understanding**: Types and importance of data
3. **Core Concepts**: Supervised vs Unsupervised learning
4. **Practical Skills**: Hands-on coding with real datasets
5. **Advanced Topics**: Neural networks and deep learning
6. **Professional Level**: Architecture understanding and ethics

## 🔧 Customization

You can extend the game by:
- Adding new quests in the `QuestManager.define_quests()` method
- Creating new quest types in `QuestRenderer`
- Modifying the scoring and level system
- Adding new achievements
- Integrating additional AI models

## 📋 Requirements

- Python 3.7+
- Streamlit
- Ollama (local or remote)
- Required Python packages (see requirements.txt)

## 🤝 Contributing

Feel free to contribute by:
- Adding new learning quests
- Improving visualizations
- Adding new achievement types
- Enhancing AI interactions
- Fixing bugs or improving performance

## 📄 License

This project is open source and available under the MIT License.

---

**Ready to become an AI expert? Start your quest now!** 🚀🧠

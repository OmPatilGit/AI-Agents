## Autonomous AI Research Agent
### 📖 Introduction
This project is a terminal-based autonomous AI agent designed to perform comprehensive research on any given topic. Leveraging the power of LangGraph and large language models, the agent can plan a research strategy, browse the web for up-to-date information, synthesize findings, and critically evaluate its own work to produce a high-quality summary.

This agent demonstrates a self-correcting loop, where it can critique its own summary and automatically perform another round of research to improve the output, mimicking a human research workflow.

### 🛠️ Core Technologies
This agent is built with a modern stack of AI and Python libraries:

**LangGraph**: For building the stateful, multi-step agent as a cyclical graph.

**LangChain**: For integrating the language model and other components.

**Tavily AI**: As the search engine tool, providing real-time, AI-optimized web search results.

**Rich**: For creating a beautiful and interactive command-line interface.

### ✨ Features
**Dynamic Planning**: The agent creates a custom research plan for each query.

**Live Web Research**: Uses the Tavily API to gather current information from the internet.

**Summarization**: Synthesizes the collected data into a coherent summary.

**Self-Critique & Refinement**: The agent evaluates its own summary and can trigger a revision loop if the quality is not high enough.

**Persistent State**: Utilizes SQLite to save the state of the conversation, allowing for robust and stateful interactions.

### 🚀 Setup and Installation
Follow these steps to get the agent running on your local machine.

1. Clone the Repository
`git clone https://github.com/OmPatilGit/AI-Agents`
`cd ReAct-Agent`

2. Create a Virtual Environment
This project uses **uv** for fast package and environment management. First, make sure you have uv installed:

### Install uv if you don't have it
`pip install uv`

Then, create and activate the virtual environment:

### Create the virtual environment
`uv venv`

### Activate the environment
#### For Windows
`.\.venv\Scripts\activate`
#### For macOS/Linux
`source .venv/bin/activate`

3. Install Dependencies
Use uv to install all the required Python packages from requirements.txt.

`uv pip install -r requirements.txt`

(Note: You may need to create a requirements.txt file by running uv pip freeze > requirements.txt)

4. Set Up Environment Variables
The agent requires API keys for the language model (e.g., OpenAI) and the Tavily search tool.

Create a file named .env in the root of your project directory and add your keys:

`OPENAI_API_KEY="your-openai-api-key-here"`
`TAVILY_API_KEY="your-tavily-api-key-here"`

### ▶️ How to Run
Once the setup is complete, you can run the agent from your terminal:

`uv run main.py`

You will be greeted by an interactive prompt where you can enter your research topic.

### 🔮 Future Plans
This project is currently a powerful terminal-based application. The next major step in its development is to build a **graphical user interface (UI)** to make it more accessible and user-friendly. Potential plans include:

**Web Interface**: Using a framework like Chainlit or Streamlit to create a chat-like web application.

**Desktop Application**: Exploring options for a native desktop UI.

Stay tuned for updates!

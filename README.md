# 🧠 Self-Learning AI Agent

A locally running AI agent with short-term and long-term memory.

This project is being built step by step to understand how AI memory systems such as Mem0 work. The current version uses Mem0 for long-term memory, and the goal is eventually to build a custom memory system from scratch.

## 🚀 Current Features

- 🤖 Local AI model using Ollama
- 🧠 Long-term memory using Mem0
- 🔍 Vector search using Qdrant
- 📊 Embeddings using `nomic-embed-text`
- 💬 Short-term conversation memory
- 🎯 Memory relevance filtering
- 💻 Runs locally on the user's machine

## 🛠️ Tech Stack

- Python
- Ollama
- Llama 3.2
- Mem0
- Qdrant
- nomic-embed-text

## 📁 Project Structure

```text
self-learning-agent/
│
├── main.py          # Main AI agent
├── memory.py        # Mem0 memory configuration
├── mem0_test.py     # Testing the memory system
├── README.md
└── .gitignore
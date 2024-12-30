# L'Oracolo 🔮

A powerful multi-agent system for advanced forum data analysis and insights generation.

## Overview

L'Oracolo is a sophisticated AI-powered platform that leverages multiple agents to analyze forum discussions, extract insights, and provide comprehensive answers to complex queries. The system uses advanced language models and a swarm of specialized agents to process and synthesize information from large datasets.

## Features

- 🤖 Multi-agent system for parallel data processing
- 📊 Advanced forum data analysis
- 💡 Intelligent query understanding
- 🔄 Real-time processing and responses
- 📈 Sentiment analysis and trend detection
- 🔍 Deep context understanding
- 🌐 Multi-language support (Italian/English)

## Technical Stack

- Python 3.9+
- Streamlit for UI
- LangChain for AI orchestration
- OpenAI GPT models
- Pinecone for vector database
- Sentence Transformers for embeddings

## Requirements

```bash
streamlit>=1.31.0
langchain>=0.1.0
openai>=1.12.0
pinecone-client>=3.0.0
sentence-transformers
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`

## Usage

⚠️ **IMPORTANT**: This software is proprietary and requires explicit authorization for any use. Please refer to the LICENSE file for detailed terms.

To request authorization, please contact the owner.

## Architecture

The system consists of several components:

1. **Frontend Layer**
   - Streamlit-based user interface
   - Real-time interaction
   - Progress tracking

2. **Agent Layer**
   - Multiple analyzer agents
   - Synthesis agent
   - Load balancing

3. **Processing Layer**
   - Document chunking
   - Embedding generation
   - Vector storage

4. **Database Layer**
   - Vector database
   - Metadata storage
   - Query optimization

## License

Proprietary - All Rights Reserved

Copyright (c) 2024

This software requires explicit written authorization for any use. See the LICENSE file for details.
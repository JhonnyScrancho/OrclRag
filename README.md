# 🔮 L'Oracolo

Un sistema RAG (Retrieval-Augmented Generation) per l'analisi di discussioni di forum utilizzando LangChain, Pinecone e GPT.

## Requisiti
- Python 3.9+
- Account Pinecone
- API Key OpenAI

## Setup
1. Clona il repository
2. Installa le dipendenze: `pip install -r requirements.txt`
3. Configura `.streamlit/secrets.toml` con le tue API keys
4. Avvia l'app: `streamlit run src/app.py`

## Struttura Dati
Il sistema accetta file JSON con la seguente struttura:
- Thread con titolo, URL e timestamp
- Posts con autore, contenuto, timestamp e keywords
- Metadati per l'analisi del sentiment

## Funzionalità
- Caricamento e processing di dati da JSON
- Indicizzazione vettoriale con Pinecone
- Chat interattiva con RAG
- Visualizzazione dei dati processati
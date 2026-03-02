**Project Overview:**
The Government Scheme Explainer Chatbot is an AI-powered conversational system designed to help citizens easily understand Andhra Pradesh government schemes. Many government welfare schemes exist for students, farmers, women, and other communities, but people often struggle to access correct and clear information.
This chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate, reliable, and easy-to-understand answers directly from official government documents.

**Team Members:**
G. Lakshmi Prasanna
Ch. Durga Meenakshi
S. Rishika
E. Jaya Sri

**Problem Statement:**
Although many government schemes are designed for public welfare, citizens face several challenges:
Scheme details are written in complex legal language
Information is scattered across multiple websites and PDFs
Official documents are lengthy and time-consuming to read
Users need direct answers for questions like:
Eligibility criteria
Required documents
Benefits and application process

**To solve these issues, we developed a RAG-based chatbot that:**
Retrieves information from official government documents
Answers user queries in simple and clear language
Avoids misinformation and hallucinations
Provides context-aware and accurate responses

What is Retrieval-Augmented Generation (RAG)?
RAG combines document retrieval with language generation to ensure factual correctness.
Why RAG?
**Without RAG:**
Chatbot may hallucinate answers
Central and state schemes may get mixed
Information may become outdated
**With RAG:**
Answers are generated only from retrieved official content
Ensures reliability and accuracy

**RAG Workflow**
User asks a question
Question is converted into embeddings
Embeddings are searched in a Vector Database (FAISS)
FAISS retrieves the most relevant document chunks from PDFs
Retrieved content + user question are sent to the LLM prompt
LLM generates a clear and simple response

**Existing Chatbots vs Our Chatbot**
**Existing Chatbots**
Only list schemes
Do not answer user-specific queries
Often provide outdated information
Give unnecessary or irrelevant content
**Our Chatbot**
Provides detailed scheme information
Answers user-specific questions
Uses semantic understanding, not just keyword matching
Generates responses in simple language
Ensures up-to-date and factual data

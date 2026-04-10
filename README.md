# Apollo Hospitals Chennai - RAG Project

A Retrieval-Augmented Generation (RAG) system for Apollo Hospitals Chennai, enabling intelligent querying over hospital documents, policies, and medical information.

## What This Project Does

This project builds a RAG pipeline that allows users to ask natural language questions and receive accurate, context-grounded answers sourced from Apollo Hospitals Chennai's internal documents. It combines a vector search retrieval system with a large language model (LLM) to generate responses backed by real document content.

## Use Cases

- Patient FAQs (appointments, departments, doctors, procedures)
- Hospital policy and procedure lookups
- Medical service and facility information retrieval
- Staff knowledge base queries

## Folder Structure

```
apollo-rag/
├── documents/    # Raw source documents (PDFs, Word files, text files, etc.)
├── src/          # Source code for the RAG pipeline (ingestion, retrieval, generation)
├── data/         # Processed data, vector embeddings, and indexes
└── README.md     # This file
```

## Getting Started

1. Place source documents in the `documents/` folder.
2. Run the ingestion pipeline in `src/` to chunk, embed, and index the documents into `data/`.
3. Query the RAG system via the provided interface or API.

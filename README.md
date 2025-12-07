# Advanced RAG PDF Question Answering Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for complex question answering over PDF documents. It is containerized with Docker and ready for git integration.

## Project Structure
- `ingestion/`: PDF ingestion and preprocessing
- `retrieval/`: Document retrieval logic
- `generation/`: Answer generation modules
- `evaluation/`: Evaluation scripts and metrics
- `data/pdfs/`: Sample PDF data

## Setup
1. Clone the repository
2. Build the Docker container: `docker build -t rag-pdf-qa .`
3. Run the container: `docker run -it --rm -v $(pwd)/data/pdfs:/app/data/pdfs rag-pdf-qa`

## Usage
- Place your PDFs in `data/pdfs/`
- Run the entrypoint script inside the container

## Requirements
See `requirements.txt` for dependencies.

## License
MIT

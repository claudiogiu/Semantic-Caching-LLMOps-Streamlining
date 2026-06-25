# Semantic Caching for LLMOps Streamlining via Ollama and Redis
## Introduction  

This repository is designed for deploying an entirely local semantic‑caching system that streamlines LLMOps workflows by adopting Ollama as the small‑language‑model runtime and Redis as the vector‑indexing and similarity‑search engine, integrating embedding generation, vector storage, semantic matching and request deduplication within a unified environment that enhances end‑to‑end model interaction through reduced latency and minimized LLM invocation overhead.

Ollama provides the local execution engine for the small language models used for embedding generation and text generation,  while Redis offers the vector‑indexing and high‑performance similarity‑search capabilities required to store embedding representations and execute efficient semantic‑matching operations over the cached query space.

## Getting Started

To set up the repository properly, follow these steps:

**1.** **Configure the Environment File**  

- Initialize the environment configuration by copying the `.env.example` file template into the project root as `.env`:

  ```bash
  mv .env.example .env  
  ```

- Assign valid values to all required variables.

**2.** **Execute the Service Provisioning with Makefile**  

- The repository includes a **Makefile** that automates the initialization of all components required to run the local semantic-caching system.

- Run the following command to start the complete service suite:

  ```bash
  make all
  ```

- This command sequentially performs the following operations:

  - Starts all required services using Docker Compose, ensuring that the API, the vector‑indexing engine and the model runtime are correctly instantiated.
  - Interprets the model identifiers provided through the environment variables `OLLAMA_EMBED_MODEL` and `OLLAMA_CHAT_MODEL`, which respectively specify the embedding model used for vector derivation and the generation model used for producing textual outputs.
  - Downloads the specified models from Ollama, ensuring that the embedding model is available for vector derivation and that the generation model is available for text production within the semantic-caching workflow.

**3.** **Access the API** 
  
  - Once the container is running, the API is accessible at:

    - **Swagger UI for interactive docs:** `localhost:8000/docs`  
    - **Cache warming endpoint:** `api/v1/cache/warmup`  
    - **Semantic‑cache query requests:** `api/v1/cache/query`
    - **Cache‑flush maintenance resource:** `api/v1/cache`
    - **Record‑level deletion endpoint:** `api/v1/cache/record`  

## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 

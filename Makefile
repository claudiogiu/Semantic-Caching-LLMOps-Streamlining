.PHONY: all start-services pull-models

# Step 1: Environment variables provisioning
ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

# Step 2: Service instantiation
start-services:
	@echo "[SERVICE INSTANTIATION] Initialization of Docker services initiated."
	@docker compose up -d >/dev/null 2>&1
	@echo "[SERVICE INSTANTIATION] Docker services successfully initialized."

# Step 3: Model synchronization
pull-models: start-services
	@echo "[MODEL SYNCHRONIZATION] Retrieval of required Ollama models initiated."
	@docker exec ollama ollama pull $(OLLAMA_EMBED_MODEL) >/dev/null 2>&1
	@docker exec ollama ollama pull $(OLLAMA_CHAT_MODEL) >/dev/null 2>&1
	@echo "[MODEL SYNCHRONIZATION] Required Ollama models successfully retrieved."

# Master target: full pipeline execution
all: start-services pull-models
	@echo "[ALL] Complete environment initialization and model provisioning finalized."
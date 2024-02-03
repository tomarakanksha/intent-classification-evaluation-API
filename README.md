# Intent Classification with Transformers

This repository contains implementation of an intent classification model using the Hugging Face Transformers library. The model is trained to classify user queries into various intents related to to-do list operations, including "add_todos," "update_todos," "search_todos," "view_todos," "complete_todos," "set_reminder," and "delete_todos."

## Description

The `IntentClassifier` class in the provided script (`intent_classifier.py`) leverages a pre-trained transformer model (default: BERT-base-uncased) to train on custom datasets for different intent collections. The model is trained using the Trainer module from Hugging Face, and it can be used for real-time intent classification in applications.

The repository also includes a FastAPI script (`main.py`) that serves as an API endpoint to classify user queries using the trained model. You can send a POST request with a user query, and the API will respond with the predicted intent.

## Contents

- `intent_classifier.py`: Python script containing the `IntentClassifier` class for training and classifying intents.
- `main.py`: FastAPI script providing an API endpoint for real-time intent classification.
- `IntentDataset.py`: Module defining the custom dataset class for intent classification.
- `intents`: Directory for the intents of to-do list operations.

## Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Load your environment variables: `cp .env.example .env` (Linux/Mac) or `copy .env.example .env` (Windows)

## Usage

1. Add your intents in '/intents' folder.
2. Run the FastAPI application: `uvicorn main:app --reload`
3. Visit `http://127.0.0.1:8000/docs` in your browser to interact with the API using Swagger documentation.

## Model Evaluation

For model evaluation, a validation dataset can be provided in CSV format (example_queries.csv). The file should have header-['user_queries','actual_intent']. Use the evaluate method in IntentClassifier to assess the model's accuracy, precision, recall, and F1 score on the provided dataset.s
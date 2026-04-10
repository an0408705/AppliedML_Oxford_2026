# Legal RAG Context Evaluation Pipeline

This repository contains code to evaluate large language models (LLMs) under different context conditions (gold, retrieved, poisoned, none) and prompting strategies (non-RAG, neutral, skeptical, faithful).

## Setup

Install dependencies:


pip install -r requirements.txt


Create a `.env` file in the root directory with your configuration:


OPENROUTER_API_KEY=your_api_key

GEN_MODEL=openai/gpt-5-mini
JUDGE_MODEL=openai/gpt-5

SAMPLE_SIZE=100
SAMPLE_SEED=42

CONTEXT_CONDITIONS=gold,retrieved,poisoned,none
PROMPT_TYPES=non_rag,neutral,skeptical,faithful

TEMPERATURE=0.0
MAX_TOKENS=1024

RUN_NAME=experiment_name
OUTPUT_ROOT=outputs/runs


## Running

Run the full pipeline:


bash run.sh


This will generate model outputs, evaluate them using an LLM-as-a-judge, and aggregate the results.

## Outputs

Results are saved in:


outputs/runs/<RUN_NAME>/


## Notes

- The dataset is located in `data/clean_input.csv`

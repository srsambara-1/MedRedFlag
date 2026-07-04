# RAG (Retrieval-Augmented Generation)

This module requires the external MedRAG repository for retrieval functionality.

## Setup

1. Clone MedRAG:
   ```bash
   git clone https://github.com/Teddy-XiongGZ/MedRAG
   ```

2. Follow MedRAG setup instructions for downloading corpora and models.

3. **Important**: Copy the scripts in this folder (`run_retrieval.py`, `rag.py`, `run_rag.sh`) to the root of your MedRAG repository and run them from there.

## Usage

From the MedRAG repository root, edit the placeholder paths and model settings at the top of `run_rag.sh`, then:
```bash
sh run_rag.sh
```
This runs retrieval (`run_retrieval.py`) followed by RAG-enhanced generation (`rag.py`). Set `API_TYPE` to `vllm`, `openai`, or `anthropic`; `vllm` uses the `VLLM_HOST`/`VLLM_PORT` server, while `openai`/`anthropic` require the matching API key in the environment.

## Additional Options

Both scripts take extra flags beyond what `run_rag.sh` sets by default (documented inline in the script):

```
run_retrieval.py
--n_rows N             Limit retrieval to first N rows
--k N                  Documents retrieved per query (default: 32)
--db_dir PATH          Corpus/index directory (default: ./corpus)

rag.py
--top-k N              Number of retrieved documents to include (default: 5)
--output-column Name   Custom output column name (default: model_answer_rag)
--max-tokens N         Maximum tokens in response (default: 2048)
--temperature T        Sampling temperature (default: 0.0)
--max-rows N           Limit processing to first N rows
--resume               Resume from existing output file
```

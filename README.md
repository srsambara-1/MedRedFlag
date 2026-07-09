# MedRedFlag

Official repository for the paper **[MedRedFlag: Investigating How LLMs Redirect Misconceptions in Real-World Health Communication](https://arxiv.org/abs/2601.09853)**

## Dataset
The MedRedFlag dataset is available at [Duke Research Data Repository](https://research.repository.duke.edu/record/542?ln=en&v=zip).

> **Note:** By downloading this dataset, you agree to have obtained ethics approval from your institution. This is a requirement of the creator of the base dataset [MedRedQA](https://data.csiro.au/collection/csiro:62454) presented by [Nguyen et al. (2023)](https://aclanthology.org/2023.ijcnlp-main.42/).

## Repository
This repository present scripts used for the automatic part of MedRedFlag dataset formation, LLM responses generation, and LLM answers evaluation. The examples provided in LLM prompts for dataset formation pipeline are redacted for data privacy (check for `[!!!!!! REDACTED !!!!!!]`).

### Setup
- Python 3.10+ and bash or Conda.
- API keys: `OPENAI_API_KEY` (GPT‑5) and `ANTHROPIC_API_KEY` (Claude Opus 4.5), plus a vLLM host/port for local models (Llama‑3.3, MedGemma).
- Environment: `conda create -n medred python=3.10 && conda activate medred` (or `python -m venv .venv && source .venv/bin/activate`), then `pip install -r requirements.txt`.
- Base dataset: [MedRedQA](https://data.csiro.au/collection/csiro:62454) presented by [Nguyen et al. (2023)](https://aclanthology.org/2023.ijcnlp-main.42/).

### Pipelines

#### 1. Dataset Formation
```
cd dataset_formation
sh run_dataset_formation.sh
```
Require downloading MedRedQA first. Contains scripts for preprocessing MedRedQA (`prefilter.py`), tagging redirected cases (`redirection_tagging.py`), post-processing for removing cases with missing context information (`postprocessing.py`), and false assumptions extraction and cleaning (`presupposition_extraction.py`). In the paper we ran every annotation with GPT‑5. We performed manual inspections and clean-ups outside this automatic pipeline. 

#### 2. LLM Response Generation
```
cd llm_response_generation
sh run_llm_response.sh
```
Runs baseline LLM responses (`baseline.py`), as well as with mitigation strategies of Identify and Respond (`identify_and_respond.py`) and Oracle Assumptions Provided (`oracle_assumptions_provided.py`). The paper evaluated four models across these stages: GPT‑5 (OpenAI), Claude Opus 4.5 (Anthropic), Llama‑3.3‑70B‑Instruct (vLLM), and MedGemma‑27b‑it (vLLM). For mitigation strategy with RAG, follow `rag/README.md` (requires [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) by [Xiong et al. (2024)](https://aclanthology.org/2024.findings-acl.372/)).

#### 3. Evaluation
```
cd llm_response_evaluation
sh run_eval.sh
```
Assign "False Assumptions Addressed" (`eval_addressed.py`) and "False Assumptions Accommodated" (`eval_accommodated.py`) labels and compute statistics. We used GPT‑5 as the judge in the paper. Note that "False Assumptions Accommodated" requires supporting conditions for each question, which was clinician-written in our study and outside this automatic pipeline.


## Citation
If you use or reference this work and/or MedRedFlag dataset, please cite the paper:
```bibtex
@misc{sambara2026medredflaginvestigatingllmsredirect,
      title={MedRedFlag: Investigating how LLMs Redirect Misconceptions in Real-World Health Communication}, 
      author={Sraavya Sambara and Yuan Pu and Ayman Ali and Vishala Mishra and Lionel Wong and Monica Agrawal},
      year={2026},
      eprint={2601.09853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.09853}, 
}
```

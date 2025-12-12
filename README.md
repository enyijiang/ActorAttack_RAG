<div align="center">
    <h2>
      Multi Turn Jailbreak Attacks using High-Quality Actors: A Retrieval-Augmented Generation Approach<br><br>
     <a href="https://arxiv.org/abs/2410.10700"> <img alt="paper link" src="https://img.shields.io/badge/Paper-arXiv-red"> </a>
     <a href="https://huggingface.co/datasets/SafeMTData/SafeMTData"> <img alt="model link" src="https://img.shields.io/badge/Data-SafeMTData-blue"> </a> 
    </h2>
</div>

<h4 align="center">RESEARCH USE ONLY✅ NO MISUSE❌</h4>
<h4 align="center">LOVE💗 and Peace🌊</h4>
<!-- <h4 align="center"></h3> -->

## 📄 Brief Information for each file and directory
- `data` ---> includes the original jailbreak benchmark data.
- `prompts` ---> are the prompts for attack data generation, evaluation, and safety alignment data generation.
- `main.py` ---> is the file to run ActorAttack, which consists of two-stages: pre-attack (`preattack.py`) and in-attack (`inattack.py`).
- `judge.py` ---> is the file to define our GPT-Judge.
- `ft` ---> contains the script and python file to train LLMs.
- `construct_dataset.py` ---> is the file to construct the multi-turn safety alignment data.
 

## 🛠️ Attack data generation
- Installation
```
conda create -n actorattack python=3.10
conda activate actorattack
pip install -r requirements.txt
```
- Before running, you need to set the API credentials in your environment variables. An example of using your `.env` file is:
```
BASE_URL_GPT="https://api.openai.com/v1"
GPT_API_KEY="YOUR_API_KEY"

BASE_URL_CLAUDE="https://api.anthropic.com/v1"
CLAUDE_API_KEY="YOUR_API_KEY"

BASE_URL_DEEPSEEK="https://api.deepseek.com/v1"
DEEPSEEK_API_KEY="YOUR_API_KEY"

BASE_URL_DEEPINFRA="https://api.deepinfra.com/v1/openai"
DEEPINFRA_API_KEY="YOUR_API_KEY"
```

## ⚡️ Model Recommendation for Attack Generation
We have noticed that GPT-4o, when used as an attack model, tends to refuse to generate multi-turn attack prompts. Therefore, we recommend using the open-source LLM WizardLM-2-8x22B. (You can also access the model through the DeepInfra API via microsoft/WizardLM-2-8x22B.)

✨An example run for RAG and Original modes:

```
python3 main.py --questions 50 --actors 3 --behavior ./data/harmbench.csv --attack_model_name gpt-4.1-mini --target_model_name meta-llama/Meta-Llama-3-8B-Instruct --early_stop true --dynamic_modify true --mode rag

python3 main.py --questions 50 --actors 3 --behavior ./data/harmbench.csv --attack_model_name gpt-4.1-mini --target_model_name meta-llama/Meta-Llama-3-8B-Instruct --early_stop true --dynamic_modify true --mode ori
```


You can find the actors and initial jailbreak queries for each instruction in `pre_attack_result`, and the final attack result in `attack_result`.

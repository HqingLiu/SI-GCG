# Code Description

## File Structure

### data

- input.json：Original behavior files（such as 50 malicious questions from track 1a）
- suffix_lists.json：successful attack suffix list from input.json
- output.json：save the results

### models：Storing model weights

- Llama-2-7b-chat-hf
- roberta：from paper "GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts" https://huggingface.co/hubert233/GPTFuzz/tree/main

### processing_pipeline
- llm_attacks：IGCG Attack Config Profile
- attack_init_suffix.py：Get the list of suffixes that can be migrated after the attack is successful
- attack.py：Attack llama model

### requirements.txt

### README.md







## Quick Start 

cd processing_pipeline

### 1. Get the suffix init list
```python
cd code
python processing_pipeline/attack_init_suffix.py
```

### 2. Attack llama
```python
cd code
python processing_pipeline/attack.py
```












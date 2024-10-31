# SI-GCG

## Quick Start 

- install dependents

```
pip install -r requirements.txt
```

### 1. Get the suffix init list

```python
python processing_pipeline/attack_init_suffix.py
```

### 2. Attack Large language models

```python
python processing_pipeline/attack.py
```



## File Structure

### data

- input.json：Original behavior files（from https://github.com/AISG-Technology-Team/GCSS-Track-1A-Submission-Guide/blob/master/20240612-behaviors.json）
- suffix_lists.json：successful attack suffix list from input.json
- output.json：save the results

### models：Storing model weights

- Llama-2-7b-chat-hf
- roberta：from paper "GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts" https://huggingface.co/hubert233/GPTFuzz/tree/main

### processing_pipeline
- llm_attacks：IGCG Attack Config Profile
- attack_init_suffix.py：Get the list of suffixes that can be migrated after the attack is successful
- attack.py：Attack LLMs

### 
















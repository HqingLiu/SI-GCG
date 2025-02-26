# SI-GCG

The official repository for [Boosting Jailbreak Transferability for Large Language Models](https://arxiv.org/abs/2410.15645).

Please feel free to contact [hqliu@buaa.edu.cn](mailto:hqliu@buaa.edu.cn) if you have any question.

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

- input.json：Original behavior files (from https://github.com/AISG-Technology-Team/GCSS-Track-1A-Submission-Guide/blob/master/20240612-behaviors.json)
- suffix_lists.json：successful attack suffix list from input.json
- output.json：save the results

### models：Storing model weights

- Llama-2-7b-chat-hf
- roberta：from paper "GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts" https://huggingface.co/hubert233/GPTFuzz/tree/main

### processing_pipeline
- llm_attacks：IGCG Attack Config Profile
- attack_init_suffix.py：Get the list of suffixes that can be migrated after the attack is successful
- attack.py：Attack LLMs



## Citation

Kindly include a reference to this paper in your publications if it helps your research:

```
@article{liu2024boosting,
  title={Boosting Jailbreak Transferability for Large Language Models},
  author={Liu, Hanqing and Zhou, Lifeng and Yan, Huanqian},
  journal={arXiv preprint arXiv:2410.15645},
  year={2024}
}
```
















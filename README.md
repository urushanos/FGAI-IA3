# English–French Machine Translation Using Transformer and Large Language Models

## 1. Problem Statement and Objective

This project compares four neural machine-translation approaches for English-to-French translation:

1. Basic Transformer (encoder–decoder), trained from scratch.
2. mT5 (encoder–decoder), fine-tuned for translation.
3. LLaMA (decoder-only), used with instruction prompting.
4. Gemma (decoder-only), used with instruction prompting.

The objective is to compare translation quality and computational requirements using a common test subset and metrics including BLEU, chrF, training/fine-tuning time where applicable, inference time, and parameter count.

> **Important:** The comparison is not a controlled architecture-only experiment. The models differ in parameter count, pretraining, tokenizer, training data, and adaptation method.

## 2. Dataset

- **Dataset:** Helsinki-NLP OPUS Books English–French
- **Dataset source:** https://huggingface.co/datasets/Helsinki-NLP/opus_books
- **Language direction:** English → French
- **Preprocessing:** Empty pairs and unusually long sentence pairs were filtered. The remaining pairs were shuffled with seed `42`, and up to 6,000 examples were selected.
- **Split:** 4,800 training, 600 validation, and 600 test examples.
- **Evaluation subset:** The first 100 examples from the test split were used for the reported model comparison.

The notebook loads the public dataset using the Hugging Face `datasets` library. The exact dataset version/configuration and the final number of valid pairs should be rechecked from the executed notebook output before submission.

## 3. Models and Implementation

| Model | Architecture | Model / configuration | Adaptation |
|---|---|---|---|
| Basic Transformer | Encoder–decoder | Custom 2-layer encoder and decoder; 128-dimensional embeddings | Trained from scratch for 5 epochs |
| mT5 | Encoder–decoder | `google/mt5-small` | Fine-tuned for 1 epoch using 1,000 training examples |
| LLaMA | Decoder-only | `meta-llama/Llama-3.2-1B-Instruct` | Instruction prompting; no fine-tuning |
| Gemma | Decoder-only | `google/gemma-3-1b-it` | Instruction prompting; no fine-tuning |

LLaMA and Gemma are loaded using 4-bit quantization to reduce GPU memory requirements. The notebook uses deterministic generation for the reported baseline runs (`do_sample=False`).

## 4. Repository Contents

```text
FGAI-IA3/
├── English_French_Translation.ipynb
├── README.md
├── requirements.txt
│ 
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── all_model_translations.csv
```

The notebook contains data loading, preprocessing, model implementation/loading, training or prompting, inference, metric calculation, and responsible-AI discussion.

## 5. Results

The following values are transcribed from the current `model_comparison.csv`. They are preliminary results from the current notebook run and should be regenerated after the final notebook changes.

| Model | BLEU | chrF | Training / fine-tuning time (s) | Inference time for 100 examples (s) | Parameter count reported by notebook |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 0.220975 | 6.083893 | 19.702126 | 11.750217 | 3,743,040 |
| mT5 | 0.076989 | 2.436307 | 53.941712 | 66.776010 | 300,176,768 |
| LLaMA | 8.141973 | 31.258706 | N/A (prompted; not fine-tuned) | 76.245362 | 749,275,136 |
| Gemma | 12.040001 | 37.339210 | N/A (prompted; not fine-tuned) | 155.259526 | 651,005,056 |

BLEU and chrF are reported using SacreBLEU corpus metrics. The low scores of the Basic Transformer and mT5 indicate weak translation quality in this limited training setup. Gemma obtained the highest BLEU and chrF scores among the four models in this run. The Basic Transformer had the shortest measured inference time.

**Parameter-count verification required:** The parameter figures above are copied from the current notebook output. Before submission, verify that each figure represents the full model parameter count and not an artifact of quantized parameter storage or the loaded model implementation. Report quantization separately from total parameter count.

**Timing caveat:** Inference times are hardware- and runtime-dependent. Record the GPU/runtime used and ensure timing includes equivalent stages for all models. The current values are not hardware-independent benchmarks.

## 6. Model Comparison and Discussion

- **Basic Transformer:** It is the smallest model and has the lowest measured inference time, but its generated translations contain `<unk>` tokens and repetitions. Training from scratch on a small subset limits vocabulary coverage and language understanding.
- **mT5:** It benefits from multilingual pretraining, but the current fine-tuning setup is limited to one epoch and 1,000 examples. Its outputs contain incomplete text and `<extra_id_0>` artifacts, suggesting that the current adaptation/generation setup is inadequate.
- **LLaMA:** Instruction prompting produces more fluent translations than the two current encoder–decoder runs, but some outputs omit or alter source meaning. It was prompted rather than fine-tuned.
- **Gemma:** It achieved the highest current BLEU and chrF scores, but it is also the slowest of the four in the measured 100-example inference run. Its outputs can still contain mistranslations or semantic changes.

These results compare complete model-and-adaptation setups, not architecture alone. Model size, pretraining corpus, tokenizer, quantization, fine-tuning data, decoding settings, and runtime all affect the outcome.

## 7. Responsible and Ethical AI

### Hallucinations and faithfulness
A translation may sound fluent while changing, omitting, or inventing information. The current examples include incomplete outputs and semantic mistranslations. Compare translations against the source and reference, and use bilingual human review for important content.

### Bias
Models may reproduce gender, cultural, or linguistic biases present in their training data. Evaluate diverse examples and review potentially biased translations.

### Cultural misrepresentation
Idioms, literary references, and culturally specific expressions may not have direct equivalents. Use context-aware evaluation and bilingual reviewers familiar with the relevant cultures.

### Sensitive content
Errors in medical, legal, financial, or emergency translations can cause harm. Do not rely solely on automated translation for high-impact decisions; require qualified human verification and protect personal data.

### Mitigation
- Evaluate with both automatic metrics and human review.
- Test idioms, named entities, gendered language, and sensitive examples.
- Preserve the source text and disclose that output is machine-generated.
- Escalate high-risk translations to qualified translators.
- Avoid sending confidential text to services without appropriate privacy safeguards.

## 8. Requirements and Execution

The notebook was developed for Google Colab with a GPU runtime. It installs its Python dependencies in the first notebook cell.

### Execution steps

1. Open `English_French_Translation.ipynb` in Google Colab.
2. Select a GPU runtime (the notebook was documented for a T4 GPU).
3. Run the installation and import cells.
4. Ensure the account has been granted access to the gated LLaMA and Gemma model repositories and has accepted their terms.
5. Run all cells sequentially.
6. Download the generated CSV files and updated comparison graph from the Colab runtime.
7. Replace the repository's result files with the final outputs from the final notebook run.

Never commit Hugging Face access tokens, private credentials, or downloaded gated model weights.

### Main libraries

The notebook installs/uses PyTorch, Transformers, Hugging Face Datasets and Hub, SentencePiece, SacreBLEU, Evaluate, Accelerate, PEFT, BitsAndBytes, pandas, and Matplotlib. The exact pinned versions are specified in the notebook's installation cell.

## 9. Conclusion

For the current run, Gemma achieved the highest BLEU (12.040001) and chrF (37.339210), while the Basic Transformer had the shortest measured inference time (11.750217 seconds for 100 examples). The Basic Transformer and mT5 produced very weak outputs in this configuration. Gemma is therefore the best-performing model by the reported automatic translation metrics in this experiment, but this conclusion is limited to the selected 100-example test subset and current hardware/settings. Human review remains necessary because automatic scores do not guarantee faithful or culturally appropriate translations.
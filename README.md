# InlineCoder:

This repository contains the code and data for our * paper. It provides scripts for data preprocessing, model evaluation, and ablation studies on the DevEval benchmark.

## Dataset Preparation

Clone the DevEval dataset repository:

```bash
cd references
git clone git@github.com:seketeam/DevEval.git
```

Alternatively, download the original repositories from [HuggingFace](https://huggingface.co/datasets/LJ0815/DevEval/blob/main/Source_Code.tar.gz).  
After downloading, extract the contents and place them in:  
`references/DevEval/Source_Code`

## Environment Setup

Create and activate the Conda environment:

```bash
conda create -n inlineCoder python=3.8 -y
conda activate inlineCoder
```

Add the project root to your `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/InlineCoder:$PYTHONPATH"
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Data Preprocessing

Run the following script to preprocess the DevEval dataset:

```bash
python inline_coder/Preprocess/dev_eval_preprocess.py
```

### 2. API Connectivity Test

Configure your API key and base URL in `configs/CONFIGS.json`:

```json
{
    "api_key": "****your_API_key****",
    "base_url": "https://your_URL"
}
```

Test API connectivity:

```bash
python inline_coder/Models/services.py
```

### 3. InlineCoder Evaluation

Activate the environment and run the main evaluation script:

```bash
conda activate inlineCoder
python inline_coder/inline_coder_gen.py
```

Results will be automatically evaluated using BLEU and other metrics.  
**Note:** Path issues may occur; please ensure all paths are correctly configured.

### 4. Ablation Study

Run ablation experiments with:

```bash
bash inline_coder/Ablation/scripts/ablation_no_downstream.sh
```

Results will be evaluated automatically.

### 5. Pass@K Evaluation

Pass@K evaluation is performed separately due to its runtime:

```bash
conda activate inlineCoder
python inline_coder/DevEval/run_pass_k.py
```

---

If you have any questions or encounter issues, please refer to the documentation or contact
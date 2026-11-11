

#  **Norbert's Norwegian Nonsense Nullifier**
**Human centered AI product for Norwegian Language Learning**

**Degree:** Bachelor in Artificial Intelligence - development and usage\
**Instutution:** University of Innlandet/Inland\
**Subjects:**\
KIUA2000-H26 Machine Learning III\
KIUA2001-H26 Human Centered Desgign for AI systems

----

## **User manual**

The app displays various everyday situations where the user chooses one answer, based on the formality of the given context. This is done with the assistance of two models.

| Model | Role | Folder |
| --- | --- | --- |
|[facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | Translats answer alternatives (only infernce). | `models/nllb-600m` |
| [ltg/norbert3-base_sentence-sentiment](https://huggingface.co/ltg/norbert3-base_sentence-sentiment) | Use NORBERT's already built in classification system to further distinguish between different answering formats. | `models/norbert3-base-sentiment` |


*Both models are dowloaded through huggingface and run locally using the tranformers library.*

---

### **Techinal demands**
1. windows 10/11
2. Python 3.10
3. NVIDIA-GPU (tested on RTX 5060 Laptop, 8 GB). 

---

### **Instructions before running the program**
Create your virtual environment
```bash
python -m venv venv
.venv\Scripts\Activate.ps1
```

**Expected result:**\
It should now say "(.venv)" before the line in the terminal.

---

### **Pacakges that need to be installed before running**
1. Install pytorch with CUDA
```bash
python pip install torch --index-url https://download.pytorch.org/whl/cu128
```
2. Check that the GPU is working
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.randn(1).cuda())"
```
**Expected result:** \
A version that ends on "+cu..."True", name of grapic card and "device="cuda:0""

3. Install requirements
```bash
python pip install -r requirements.txt
```
**Important information:**\
Make sure that all packages are installed inside the venv.

---

### **Download the models (ca. 3 GB)**
1. Download the translator model (NLLB)
```bash
hf download facebook/nllb-200-distilled-600M --local-dir models/nllb-600m
```
2. Download the sequence model (NORBERT)
```bash
hf download ltg/norbert3-base_sentence-sentiment --local-dir models/norbert3-base-sentiment
```
3. Check that both the models are downloaded
```bash
dir models\nllb-600m
dir models\norbert3-base-sentiment
```
**Important information:**\
Check that NORBERT contains "model.safetensors", "modeling_norbert.py" and "configuration_norbert.py".\
The two python files are needed because the model is loaded with "trust_remote_code=True"

---


### **File architecture**
```
exam_code2026/

|---api/
|------main.py

|---backend/ 
|------__init__.py
|------classifier.py              
|------translator.py

|---frontend/
|------display.py  

|---models/
|------nllb-600m
|------norbert3-base-sentiment
|------beginner_model
|------formal_model
|------main_model
|------smalltalk_model

|---traning_testing
|------clean.py
|------raw.py

|---README.md
|---requirements.txt
```
---

### **License**
| Model | License | Notation |
| --- | --- | --- |
| facebook/nllb-200-distilled-600M | CC-BY-NC-4.0 | non-comercial. Researchmodel, not made for production, can translate wrong. |
| ltg/norbert3-base_sentence-sentiment | CC-BY-4.0 | Demands "trust_remote_code=True" |






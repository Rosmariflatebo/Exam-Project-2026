from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

pipe = pipeline("translation", model="models/nllb-200-distilled-600M")

tokenizer = AutoTokenizer.from_pretrained("models/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("models/nllb-200-distilled-600M", device_map="auto") 

"""
Description:
NLLB is a pre-trained translater model. What is special with this model
is that you never have to tell it which language you are translating from 
or to which language you are translating to. 

Langauge codes:
Norwegian/bokmål: 'nob-Latn'
English:          'eng_Latn'
Arabic:           'arb_Arab'
Ukrainian:        'ukr_Cyrl'
Somali:           'som_Latn'
Polish:           'pol_Latn'

How to use:
1. Download tokenizer and model.
    Use 'AutoTokenizer' and 'AutoModelForSeq2SeqLM' with the path
    'models/nllb-600m'
    Tell the tokenizer which language the text use are using is on
    f.ex: '(scr_lang, her nob_Latn)'.
    Move the model to the GPU with 'to.(DEVICE)'
2. Text -> numbers:
    Give the tokenizer a list with norwegian sentenences, and ask 
    for PyTorch-tensors and padding. Move this tensors to the GPU
    as well.
3. Translation:
    Call on 'generate', and ask it which langauge it is going to
    translate to. This is done with 'forced_bos_token_id' which
    is the token ID to the target languages code that you get from
    the tokenizer. Use also 'max_new_tokens' so that the answer
    dont get too long.
4. Numbers -> text:
    When first starting a new language it may take some time because
    of the memory. It becomes faster after one time. 
"""
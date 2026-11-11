
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/norbert3-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, trust_remote_code=True)

model.to(DEVICE)
print(f"The model is loaded on {DEVICE}")

pipe = transformers.pipeline( "text-classification",
                            model = MODEL_PATH,
                            trust_remote_code=MODEL_PATH.startswith(MODEL_PATH),
                            config= MODEL_PATH,
                            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            )
preds = pipe(["Ait",
             "Ja",
             "Ja, takk",
             "Ja, gjerne. Takk skal du ha."])
for p in preds:
    print(p)



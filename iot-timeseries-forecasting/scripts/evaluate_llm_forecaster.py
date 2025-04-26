# evaluate_llm_forecaster.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

MODEL_PATH = "models/llm_forecaster/"
VAL_FILE = "data/llm_preprocessed/val.csv"

# 1. Load
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

df = pd.read_csv(VAL_FILE)

predictions = []
ground_truths = []

for idx, row in df.iterrows():
    prompt = row['prompt']

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 10, num_beams=3, early_stopping=True)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the predicted number from the generated text
    try:
        predicted_numbers = eval(generated_text.split(":")[-1].strip())
        if isinstance(predicted_numbers, list):
            prediction = predicted_numbers[0]  # Only first number if list
        else:
            prediction = predicted_numbers
    except:
        prediction = np.nan

    try:
        true_numbers = eval(row['completion'])
        if isinstance(true_numbers, list):
            true_value = true_numbers[0]
        else:
            true_value = true_numbers
    except:
        true_value = np.nan

    predictions.append(prediction)
    ground_truths.append(true_value)

# 2. Metrics
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

mse = mean_squared_error(ground_truths, predictions)
mae = mean_absolute_error(ground_truths, predictions)
smape = np.mean(2.0 * np.abs(predictions - ground_truths) / (np.abs(predictions) + np.abs(ground_truths))) * 100

print(f"✅ Evaluation done!")
print(f"🔵 MSE: {mse:.4f}")
print(f"🟢 MAE: {mae:.4f}")
print(f"🟠 SMAPE: {smape:.2f}%")
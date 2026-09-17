import os
import re
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# BASE_PATH is the folder where all model's version will be saved
BASE_PATH = "./my_model_versions"
# DATASET is the dataset used to train the model
DATASET = "cardiffnlp/tweet_eval"
# DEFAULT_MODEL_PATH is the downloaded/original model
DEFAULT_MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def get_highest_version_number():
    if not os.path.exists(BASE_PATH):
        #os.makedirs(BASE_PATH)
        return 0

    existing = [
        d for d in os.listdir(BASE_PATH)
        if re.match(r"model_v_\d+", d)
    ]

    if not existing:
        return 0

    versions = [
        int(re.findall(r"\d+", d)[0])
        for d in existing
    ]

    return max(versions)


def get_highest_version_model():
    newer_version = get_highest_version_number()
    model_path = f"{BASE_PATH}/model_v_{newer_version}"
    # Check model path existence, if first time running or impossible to find the model use original model
    if not os.path.exists(model_path):
        model_path = DEFAULT_MODEL_PATH
    return model_path


def build_next_version(version):
    return version + 1


def new_version_path_builder():
    version = build_next_version(get_highest_version_number())
    save_path = os.path.join(BASE_PATH, f"model_v_{version}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saved version {version}")
    return save_path


# MODEL_PATH is the model that will be used
# In this first version it is the newer existing version of the model, if this version is not available the DEFAULT_MODEL_PATH will be used.
# Later on, it will be possible to select the best version instead of the newest one.
MODEL_PATH = get_highest_version_model()
LATEST_MODEL_PATH = "./my_latest_model"

# Function to save the model got from hf without changes or training
def save_default_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    try:
        final_path = new_version_path_builder()
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"New version saved in: {final_path}")
    except Exception as e:
        print(f"Error while saving new version: {e}")

    try:
        last_model_path = LATEST_MODEL_PATH
        if os.path.exists(last_model_path):
            shutil.rmtree(last_model_path)
        os.makedirs(last_model_path, exist_ok=True)
        
        model.save_pretrained(last_model_path)
        tokenizer.save_pretrained(last_model_path)
        print(f"New 'latest' saved in: {last_model_path}")
    except Exception as e:
        print(f"Error while saving latest model: {e}")



if __name__ == "__main__":
    print(f"Model in use: {MODEL_PATH}")
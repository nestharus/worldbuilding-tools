import stanza
import os

def install_stanza_model():
    model_name = 'en'
    required_version = '1.9.2'
    model_dir = os.path.join(stanza.resources.common.DEFAULT_MODEL_DIR, model_name)

    # Check if the model is installed and its version
    if not os.path.exists(model_dir) or required_version not in os.listdir(model_dir):
        stanza.download(model_name)

if __name__ == "__main__":
    install_stanza_model()
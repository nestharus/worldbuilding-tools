import spacy

def install_spacy_model():
    model_name = 'en_core_web_lg'
    model_installed = model_name in spacy.util.get_installed_models()
    if not model_installed:
        spacy.cli.download(model_name)

if __name__ == "__main__":
    install_spacy_model()
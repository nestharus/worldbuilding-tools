import logging
from logging import Logger
from pathlib import Path

from transformers import DebertaV2TokenizerFast, DebertaV2Model


class DebertaTokenizer:
    logger: Logger
    # model: DebertaV2Model
    tokenizer: DebertaV2TokenizerFast
    unknown_token_id: int

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        cache_dir = Path.home() / '.cache' / 'tokenizer_models'
        model_dir = cache_dir / 'microsoft-deberta-v3-xsmall'

        try:
            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False
            )

            # try:
            #     self.model = DebertaV2Model.from_pretrained(
            #         str(model_dir),
            #         local_files_only=True,
            #         trust_remote_code=False
            #     )
            #     self.logger.info("Successfully loaded model with auto-detected format")
            # except Exception as e:
            #     self.logger.error(f"Failed to load model: {e}")
            #     # Try explicit format loading
            #     safetensors_path = model_dir / "model.safetensors"
            #     pytorch_path = model_dir / "pytorch_model.bin"
            #
            #     if safetensors_path.exists():
            #         model_size = safetensors_path.stat().st_size / (1024 ** 3)  # Size in GB
            #         self.logger.info(f"Found safetensors model: {model_size:.2f}GB")
            #         self.model = DebertaV2Model.from_pretrained(
            #             str(model_dir),
            #             local_files_only=True,
            #             use_safetensors=True,
            #             trust_remote_code=False
            #         )
            #         self.logger.info(f"Successfully loaded model using safetensors")
            #     elif pytorch_path.exists():
            #         model_size = pytorch_path.stat().st_size / (1024 ** 3)  # Size in GB
            #         self.logger.info(f"Found PyTorch model: {model_size:.2f}GB")
            #         self.model = DebertaV2Model.from_pretrained(
            #             str(model_dir),
            #             local_files_only=True,
            #             use_safetensors=False,
            #             trust_remote_code=False
            #         )
            #         self.logger.info(f"Successfully loaded model using PyTorch format")
            #     else:
            #         raise FileNotFoundError("No valid model file found (tried safetensors and PyTorch formats)")
        except Exception as e:
            self.logger.error(f"Error initializing tokenizer: {e}")
            raise

        self.unknown_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)

        self.logger.info(f"Initialized with DeBERTa=microsoft-deberta-v3-xsmall")

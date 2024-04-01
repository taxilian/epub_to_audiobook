import logging
import math

import torch
from TTS.api import TTS

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.core.utils import set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)


class XTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        # Init TTS with the target model name

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.setLevel(config.log)

        # TTS provider specific config
        config.output_format = config.output_format or "wav"
        config.voice_rate = config.voice_rate or "+0%"
        config.voice_volume = config.voice_volume or "+0%"
        config.voice_pitch = config.voice_pitch or "+0Hz"
        config.proxy = config.proxy or None
        config.voice_sample_wav_path = (
            config.voice_sample_wav_path or "sample_voices/demo_speaker0.wav"
        )
        config.model_name = config.model_name or "tts_models/en/ljspeech/tacotron2-DDC"

        self.tts = TTS(
            # model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            model_name=config.model_name,
            progress_bar=True,
        ).to(device)

        self.get_supported_models()

        self.price = 0.000
        super().__init__(config)

    def __str__(self) -> str:
        return f"{self.config}"

    def validate_config(self):
        pass

    def text_to_speech(
        self,
        text: str,
        output_file: str,
        audio_tags: AudioTags,
    ):
        self.tts.tts_to_file(
            text,
            # speaker_wav=self.config.voice_sample_wav_path,
            # language="en",
            file_path=output_file,
        )

        set_audio_tags(output_file, audio_tags)

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price

    def get_break_string(self):
        return "    "

    def get_output_file_extension(self):
        if self.config.output_format.startswith("wav"):
            return "wav"
        else:
            raise NotImplementedError(
                f"Unknown file extension for output format: {self.config.output_format}"
            )

    def get_supported_models(self):
        print(self.tts.list_models())

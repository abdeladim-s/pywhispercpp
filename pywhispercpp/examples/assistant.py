#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example showcasing the use of `pywhispercpp` as an assistant.
The idea is to use a `VAD` to detect speech (in this example we used webrtcvad), and when speech is detected
we run the inference.
"""
import argparse
import importlib.metadata
import queue
import time
from typing import Callable
import numpy as np
import sounddevice as sd
import pywhispercpp.constants as constants
import webrtcvad
import logging
from pywhispercpp._logger import set_log_level
from pywhispercpp.model import Model

__version__ = importlib.metadata.version('pywhispercpp')

__header__ = f"""
=====================================
PyWhisperCpp
A simple assistant using Whisper.cpp
Version: {__version__}               
=====================================
"""


class Assistant:
    """
    Assistant class

    Example usage
    ```python
    from pywhispercpp.examples.assistant import Assistant

    my_assistant = Assistant(commands_callback=print, n_threads=8)
    my_assistant.start()
    ```
    """

    def __init__(self,
                 model='tiny',
                 input_device: int = None,
                 silence_threshold: int = 8,
                 q_threshold: int = 16,
                 block_duration: int = 30,
                 commands_callback: Callable[[str], None] = None,
                 model_log_level: int = logging.INFO,
                 **model_params):

        """
        :param model: whisper.cpp model name or a direct path to a`ggml` model
        :param input_device: The input device (aka microphone), keep it None to take the default
        :param silence_threshold: The duration of silence after which the inference will be running
        :param q_threshold: The inference won't be running until the data queue is having at least `q_threshold` elements
        :param block_duration: minimum time audio updates in ms
        :param commands_callback: The callback to run when a command is received
        :param model_log_level: Logging level
        :param model_params: any other parameter to pass to the whsiper.cpp model see ::: pywhispercpp.constants.PARAMS_SCHEMA
        """

        self.input_device = input_device
        self.sample_rate = constants.WHISPER_SAMPLE_RATE  # same as whisper.cpp
        self.channels = 1  # same as whisper.cpp
        self.block_duration = block_duration
        self.block_size = int(self.sample_rate * self.block_duration / 1000)
        self.q = queue.Queue()

        self.vad = webrtcvad.Vad()
        self.silence_threshold = silence_threshold
        self.q_threshold = q_threshold
        self._silence_counter = 0

        self.pwccp_model = Model(model,
                                 log_level=model_log_level,
                                 print_realtime=False,
                                 print_progress=False,
                                 print_timestamps=False,
                                 single_segment=True,
                                 no_context=True,
                                 **model_params)
        self.commands_callback = commands_callback

    def _audio_callback(self, indata, frames, time, status):
        """
        This is called (from a separate thread) for each audio block.
        """
        if status:
            logging.warning(F"underlying audio stack warning:{status}")

        assert frames == self.block_size
        audio_data = map(lambda x: (x + 1) / 2, indata)  # normalize from [-1,+1] to [0,1]
        audio_data = np.fromiter(audio_data, np.float16)
        audio_data = audio_data.tobytes()
        detection = self.vad.is_speech(audio_data, self.sample_rate)
        if detection:
            self.q.put(indata.copy())
            self._silence_counter = 0
        else:
            if self._silence_counter >= self.silence_threshold:
                if self.q.qsize() > self.q_threshold:
                    self._transcribe_speech()
                    self._silence_counter = 0
            else:
                self._silence_counter += 1

    def _transcribe_speech(self):
        logging.info(f"Speech detected ...")
        audio_data = np.array([])
        while self.q.qsize() > 0:
            # get all the data from the q
            audio_data = np.append(audio_data, self.q.get())
        # Appending zeros to the audio data as a workaround for small audio packets (small commands)
        audio_data = np.concatenate([audio_data, np.zeros((int(self.sample_rate) + 10))])
        # running the inference
        self.pwccp_model.transcribe(audio_data,
                                    new_segment_callback=self._new_segment_callback)

    def _new_segment_callback(self, seg):
        if self.commands_callback:
            self.commands_callback(seg[0].text)

    def start(self) -> None:
        """
        Use this function to start the assistant
        :return: None
        """
        logging.info(f"Starting Assistant ...")
        with sd.InputStream(
                device=self.input_device,  # the default input device
                channels=self.channels,
                samplerate=constants.WHISPER_SAMPLE_RATE,
                blocksize=self.block_size,
                callback=self._audio_callback):

            try:
                logging.info(f"Assistant is listening ... (CTRL+C to stop)")
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logging.info("Assistant stopped")

    @staticmethod
    def available_devices():
        return sd.query_devices()


def _main():
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('-m', '--model', default='tiny.en', type=str, help="Whisper.cpp model, default to %(default)s")
    parser.add_argument('-ind', '--input_device', type=int, default=None,
                        help=f'Id of The input device (aka microphone)\n'
                             f'available devices {Assistant.available_devices()}')
    parser.add_argument('-st', '--silence_threshold', default=16,
                        help=f"he duration of silence after which the inference will be running, default to %(default)s")
    parser.add_argument('-bd', '--block_duration', default=30,
                        help=f"minimum time audio updates in ms, default to %(default)s")

    args = parser.parse_args()

    my_assistant = Assistant(model=args.model,
                             input_device=args.input_device,
                             silence_threshold=args.silence_threshold,
                             block_duration=args.block_duration,
                             commands_callback=print)
    my_assistant.start()


if __name__ == '__main__':
    _main()

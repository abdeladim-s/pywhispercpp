#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick and dirty realtime livestream transcription.

Not fully satisfying though :)
You are welcome to make it better.
"""
import argparse
import logging
import queue
from multiprocessing import Process
import ffmpeg
import numpy as np
import pywhispercpp.constants as constants
import sounddevice as sd
from pywhispercpp.model import Model
import importlib.metadata


__version__ = importlib.metadata.version('pywhispercpp')

__header__ = f"""
========================================================
PyWhisperCpp
A simple Livestream transcription, based on whisper.cpp
Version: {__version__}               
========================================================
"""

class LiveStream:
    """
    LiveStream class

    ???+ note

        It heavily depends on the machine power, the processor will jump quickly to 100% with the wrong parameters.

    Example usage
    ```python
    from pywhispercpp.examples.livestream import LiveStream

    url = ""  # Make sure it is a direct stream URL
    ls = LiveStream(url=url, n_threads=4)
    ls.start()
    ```
    """

    def __init__(self,
                 url,
                 model='tiny.en',
                 block_size: int = 1024,
                 buffer_size: int = 20,
                 sample_size: int = 4,
                 output_device: int = None,
                 model_log_level=logging.CRITICAL,
                 **model_params):

        """
        :param url: Live stream url <a direct stream URL>
        :param model: whisper.cpp model
        :param block_size: block size, default to 1024
        :param buffer_size: number of blocks used for buffering, default to 20
        :param sample_size: sample size
        :param output_device: the output device, aka the speaker, leave it None to take the default
        :param model_log_level: logging level
        :param model_params: any other whisper.cpp params
        """
        self.url = url
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.output_device = output_device

        self.channels = 1
        self.samplerate = constants.WHISPER_SAMPLE_RATE

        self.q = queue.Queue(maxsize=buffer_size)
        self.audio_data = np.array([])

        self.pwccp_model = Model(model,
                                 log_level=model_log_level,
                                 print_realtime=True,
                                 print_progress=False,
                                 print_timestamps=False,
                                 single_segment=True,
                                 **model_params)

    def _transcribe_process(self):
        self.pwccp_model.transcribe(self.audio_data, n_processors=None)

    def _audio_callback(self, outdata, frames, time, status):
        assert frames == self.block_size
        if status.output_underflow:
            logging.error('Output underflow: increase blocksize?')
            raise sd.CallbackAbort
        assert not status
        try:
            data = self.q.get_nowait()
        except queue.Empty as e:
            logging.error('Buffer is empty: increase buffer_size?')
            raise sd.CallbackAbort from e
        assert len(data) == len(outdata)
        outdata[:] = data
        audio = np.frombuffer(data[:], np.float32)
        audio = audio.reshape((audio.size, 1)) / 2 ** 5
        self.audio_data = np.append(self.audio_data, audio)
        if self.audio_data.size > self.samplerate:
            # Create a separate process for transcription
            p1 = Process(target=self._transcribe_process,)
            p1.start()
            self.audio_data = np.array([])

    def start(self):
        process = ffmpeg.input(self.url).output(
            'pipe:',
            format='f32le',
            acodec='pcm_f32le',
            ac=self.channels,
            ar=self.samplerate,
            loglevel='quiet',
        ).run_async(pipe_stdout=True)

        out_stream = sd.RawOutputStream(
            device=self.output_device,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype='float32',
            callback=self._audio_callback)

        read_size = self.block_size * self.channels * self.sample_size

        logging.info('Buffering ...')
        for _ in range(self.buffer_size):
            self.q.put_nowait(process.stdout.read(read_size))

        with out_stream:
            logging.info('Starting Playback ... (CTRL+C) to stop')
            try:
                timeout = self.block_size * self.buffer_size / self.samplerate
                while True:
                    buffer_data = process.stdout.read(read_size)
                    self.q.put(buffer_data, timeout=timeout)
            except KeyboardInterrupt:
                logging.info("Interrupted!")

    @staticmethod
    def available_devices():
        return sd.query_devices()


def _main():
    print(__header__)
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('url', type=str, help=f"Stream URL")

    parser.add_argument('-nt', '--n_threads', type=int, default=3,
                        help="number of threads, default to %(default)s")
    parser.add_argument('-m', '--model', default='tiny.en', type=str, help="Whisper.cpp model, default to %(default)s")
    parser.add_argument('-od', '--output_device', type=int, default=None,
                        help=f'the output device, aka the speaker, leave it None to take the default\n'
                             f'available devices {LiveStream.available_devices()}')
    parser.add_argument('-bls', '--block_size', type=int, default=1024,
                        help=f"block size, default to %(default)s")
    parser.add_argument('-bus', '--buffer_size', type=int, default=20,
                        help=f"number of blocks used for buffering, default to %(default)s")
    parser.add_argument('-ss', '--sample_size', type=int, default=4,
                        help=f"Sample size, default to %(default)s")
    args = parser.parse_args()


    # url = "http://n03.radiojar.com/t2n88q0st5quv?rj-ttl=5&rj-tok=AAABhsR2u6MAYFxz69dJ6eQnww"  # VOA english
    ls = LiveStream(url=args.url,
                    model=args.model,
                    block_size=args.block_size,
                    buffer_size=args.buffer_size,
                    sample_size=args.sample_size,
                    output_device=args.output_device,
                    n_threads=args.n_threads)
    ls.start()


if __name__ == '__main__':
    _main()

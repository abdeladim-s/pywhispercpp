#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions
"""
import logging
import os
from pathlib import Path
import requests
from tqdm import tqdm
from pywhispercpp.constants import MODELS_BASE_URL, MODELS_PREFIX_URL, AVAILABLE_MODELS, MODELS_DIR


def _get_model_url(model_name: str) -> str:
    """
    Returns the url of the `ggml` model
    :param model_name: name of the model
    :return: URL of the model
    """
    return f"{MODELS_BASE_URL}/{MODELS_PREFIX_URL}-{model_name}.bin"


def download_model(model_name: str, download_dir=None, chunk_size=1024) -> str:
    """
    Helper function to download the `ggml` models
    :param model_name: name of the model, one of ::: constants.AVAILABLE_MODELS
    :param download_dir: Where to store the models
    :param chunk_size: size of the download chunk

    :return: Absolute path of the downloaded model
    """
    if model_name not in AVAILABLE_MODELS:
        logging.error(f"Invalid model name `{model_name}`, available models are: {AVAILABLE_MODELS}")
        return
    if download_dir is None:
        download_dir = MODELS_DIR
        logging.info(f"No download directory was provided, models will be downloaded to {download_dir}")

    os.makedirs(download_dir, exist_ok=True)

    url = _get_model_url(model_name=model_name)
    file_path = Path(download_dir) / os.path.basename(url)
    # check if the file is already there
    if file_path.exists():
        logging.info(f"Model {model_name} already exists in {download_dir}")
    else:
        # download it from huggingface
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))

        progress_bar = tqdm(desc=f"Downloading Model {model_name} ...",
                            total=total,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024)

        try:
            with open(file_path, 'wb') as file, progress_bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    progress_bar.update(size)
            logging.info(f"Model downloaded to {file_path.absolute()}")
        except Exception as e:
            # error download, just remove the file
            os.remove(file_path)
            raise e
    return str(file_path.absolute())


def to_timestamp(t: int, separator=',') -> str:
    """
    376 -> 00:00:03,760
    1344 -> 00:00:13,440

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :param separator: seprator between seconds and milliseconds
    :return: time representation in hh: mm: ss[separator]ms
    """
    # logic exactly from whisper.cpp

    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec = msec - hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec = msec - min * (1000 * 60)
    sec = msec // 1000
    msec = msec - sec * 1000
    return f"{int(hr):02,.0f}:{int(min):02,.0f}:{int(sec):02,.0f}{separator}{int(msec):03,.0f}"


def output_txt(segments: list, output_file_path: str) -> str:
    """
    Creates a raw text from a list of segments

    Implementation from `whisper.cpp/examples/main`

    :param segments: list of segments
    :return: path of the file
    """
    if not output_file_path.endswith('.txt'):
        output_file_path = output_file_path + '.txt'

    absolute_path = Path(output_file_path).absolute()

    with open(str(absolute_path), 'w') as file:
        for seg in segments:
            file.write(seg.text)
            file.write('\n')
    return absolute_path


def output_vtt(segments: list, output_file_path: str) -> str:
    """
    Creates a vtt file from a list of segments

    Implementation from `whisper.cpp/examples/main`

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith('.vtt'):
        output_file_path = output_file_path + '.vtt'

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, 'w') as file:
        file.write("WEBVTT\n\n")
        for seg in segments:
            file.write(f"{to_timestamp(seg.t0, separator='.')} --> {to_timestamp(seg.t1, separator='.')}\n")
            file.write(f"{seg.text}\n\n")
    return absolute_path


def output_srt(segments: list, output_file_path: str) -> str:
    """
    Creates a srt file from a list of segments

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith('.srt'):
        output_file_path = output_file_path + '.srt'

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, 'w') as file:
        for i in range(len(segments)):
            seg = segments[i]
            file.write(f"{i+1}\n")
            file.write(f"{to_timestamp(seg.t0, separator=',')} --> {to_timestamp(seg.t1, separator=',')}\n")
            file.write(f"{seg.text}\n\n")
    return absolute_path


def output_csv(segments: list, output_file_path: str) -> str:
    """
    Creates a srt file from a list of segments

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith('.csv'):
        output_file_path = output_file_path + '.csv'

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, 'w') as file:
        for seg in segments:
            file.write(f"{10 * seg.t0}, {10 * seg.t1}, \"{seg.text}\"\n")
    return absolute_path

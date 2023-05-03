# pywhispercpp
Python bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with a simple Pythonic API on top of it.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Wheels](https://github.com/abdeladim-s/pywhispercpp/actions/workflows/wheels.yml/badge.svg?branch=main&event=push)](https://github.com/abdeladim-s/pywhispercpp/actions/workflows/wheels.yml)
[![PyPi version](https://badgen.net/pypi/v/pywhispercpp)](https://pypi.org/project/pywhispercpp/)

whisper.cpp is:                       
<blockquote>

High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model:

- Plain C/C++ implementation without dependencies
- Apple silicon first-class citizen - optimized via Arm Neon and Accelerate framework
- AVX intrinsics support for x86 architectures
- VSX intrinsics support for POWER architectures
- Mixed F16 / F32 precision
- Low memory usage (Flash Attention)
- Zero memory allocations at runtime
- Runs on the CPU
- [C-style API](https://github.com/ggerganov/whisper.cpp/blob/master/whisper.h)

Supported platforms:

- [x] Mac OS (Intel and Arm)
- [x] [iOS](examples/whisper.objc)
- [x] [Android](examples/whisper.android)
- [x] Linux / [FreeBSD](https://github.com/ggerganov/whisper.cpp/issues/56#issuecomment-1350920264)
- [x] [WebAssembly](examples/whisper.wasm)
- [x] Windows ([MSVC](https://github.com/ggerganov/whisper.cpp/blob/master/.github/workflows/build.yml#L117-L144) and [MinGW](https://github.com/ggerganov/whisper.cpp/issues/168)]
</blockquote>

# Table of contents
<!-- TOC -->
* [Installation](#installation)
* [Quick start](#quick-start)
* [Examples](#examples)
  * [Main](#main)
  * [Assistant](#assistant)
  * [Recording](#recording)
  * [Live Stream Transcription](#live-stream-transcription)
* [Advanced usage](#advanced-usage)
* [Discussions and contributions](#discussions-and-contributions)
* [License](#license)
<!-- TOC -->

# Installation 

1. Install [ffmpeg](https://ffmpeg.org/)

 ```bash
 # on Ubuntu or Debian
 sudo apt update && sudo apt install ffmpeg

 # on Arch Linux
sudo pacman -S ffmpeg

 # on MacOS using Homebrew (https://brew.sh/)
 brew install ffmpeg

 # on Windows using Chocolatey (https://chocolatey.org/)
 choco install ffmpeg

 # on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

2. Once ffmpeg is installed, install `pywhispercpp`

```shell
pip install pywhispercpp
```

If you want to use the examples, you will need to install extra dependencies

```shell
pip install pywhispercpp[examples]
```

Or install the latest dev version from GitHub

```shell
pip install git+https://github.com/abdeladim-s/pywhispercpp
```

# Quick start

```python
from pywhispercpp.model import Model

model = Model('base.en', n_threads=6)
segments = model.transcribe('file.mp3', speed_up=True)
for segment in segments:
    print(segment.text)
```

You can also assign a custom `new_segment_callback`

```python
from pywhispercpp.model import Model

model = Model('base.en', print_realtime=False, print_progress=False)
segments = model.transcribe('file.mp3', new_segment_callback=print)
```


* The `ggml` model will be downloaded automatically.
* You can pass any `whisper.cpp` [parameter](https://abdeladim-s.github.io/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA) as a keyword argument to the `Model` class or to the `transcribe` function.
* The `transcribe` function accepts any media file (audio/video), in any format.
* Check the [Model](https://abdeladim-s.github.io/pywhispercpp/#pywhispercpp.model.Model) class documentation for more details.

# Examples

The [examples folder](https://github.com/abdeladim-s/pywhispercpp/tree/main/pywhispercpp/examples) contains several examples inspired from the original [whisper.cpp/examples](https://github.com/ggerganov/whisper.cpp/tree/master/examples).

## Main
Just a straightforward example with a simple Command Line Interface. 

Check the source code [here](https://github.com/abdeladim-s/pywhispercpp/blob/main/pywhispercpp/examples/main.py), or use the CLI as follows:

```shell
pwcpp file.wav -m base --output-srt --print_realtime true
```
Run ```pwcpp --help``` to get the help message

```shell
usage: pwcpp [-h] [-m MODEL] [--version] [--processors PROCESSORS] [-otxt] [-ovtt] [-osrt] [-ocsv] [--strategy STRATEGY]
             [--n_threads N_THREADS] [--n_max_text_ctx N_MAX_TEXT_CTX] [--offset_ms OFFSET_MS] [--duration_ms DURATION_MS]
             [--translate TRANSLATE] [--no_context NO_CONTEXT] [--single_segment SINGLE_SEGMENT] [--print_special PRINT_SPECIAL]
             [--print_progress PRINT_PROGRESS] [--print_realtime PRINT_REALTIME] [--print_timestamps PRINT_TIMESTAMPS]
             [--token_timestamps TOKEN_TIMESTAMPS] [--thold_pt THOLD_PT] [--thold_ptsum THOLD_PTSUM] [--max_len MAX_LEN]
             [--split_on_word SPLIT_ON_WORD] [--max_tokens MAX_TOKENS] [--speed_up SPEED_UP] [--audio_ctx AUDIO_CTX]
             [--prompt_tokens PROMPT_TOKENS] [--prompt_n_tokens PROMPT_N_TOKENS] [--language LANGUAGE] [--suppress_blank SUPPRESS_BLANK]
             [--suppress_non_speech_tokens SUPPRESS_NON_SPEECH_TOKENS] [--temperature TEMPERATURE] [--max_initial_ts MAX_INITIAL_TS]
             [--length_penalty LENGTH_PENALTY] [--temperature_inc TEMPERATURE_INC] [--entropy_thold ENTROPY_THOLD]
             [--logprob_thold LOGPROB_THOLD] [--no_speech_thold NO_SPEECH_THOLD] [--greedy GREEDY] [--beam_search BEAM_SEARCH]
             media_file [media_file ...]

positional arguments:
  media_file            The path of the media file or a list of filesseparated by space

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the `ggml` model, or just the model name
  --version             show program's version number and exit
  --processors PROCESSORS
                        number of processors to use during computation
  -otxt, --output-txt   output result in a text file
  -ovtt, --output-vtt   output result in a vtt file
  -osrt, --output-srt   output result in a srt file
  -ocsv, --output-csv   output result in a CSV file
  --strategy STRATEGY   Available sampling strategiesGreefyDecoder -> 0BeamSearchDecoder -> 1
  --n_threads N_THREADS
                        Number of threads to allocate for the inferencedefault to min(4, available hardware_concurrency)
  --n_max_text_ctx N_MAX_TEXT_CTX
                        max tokens to use from past text as prompt for the decoder
  --offset_ms OFFSET_MS
                        start offset in ms
  --duration_ms DURATION_MS
                        audio duration to process in ms
  --translate TRANSLATE
                        whether to translate the audio to English
  --no_context NO_CONTEXT
                        do not use past transcription (if any) as initial prompt for the decoder
  --single_segment SINGLE_SEGMENT
                        force single segment output (useful for streaming)
  --print_special PRINT_SPECIAL
                        print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
  --print_progress PRINT_PROGRESS
                        print progress information
  --print_realtime PRINT_REALTIME
                        print results from within whisper.cpp (avoid it, use callback instead)
  --print_timestamps PRINT_TIMESTAMPS
                        print timestamps for each text segment when printing realtime
  --token_timestamps TOKEN_TIMESTAMPS
                        enable token-level timestamps
  --thold_pt THOLD_PT   timestamp token probability threshold (~0.01)
  --thold_ptsum THOLD_PTSUM
                        timestamp token sum probability threshold (~0.01)
  --max_len MAX_LEN     max segment length in characters
  --split_on_word SPLIT_ON_WORD
                        split on word rather than on token (when used with max_len)
  --max_tokens MAX_TOKENS
                        max tokens per segment (0 = no limit)
  --speed_up SPEED_UP   speed-up the audio by 2x using Phase Vocoder
  --audio_ctx AUDIO_CTX
                        overwrite the audio context size (0 = use default)
  --prompt_tokens PROMPT_TOKENS
                        tokens to provide to the whisper decoder as initial prompt
  --prompt_n_tokens PROMPT_N_TOKENS
                        tokens to provide to the whisper decoder as initial prompt
  --language LANGUAGE   for auto-detection, set to None, "" or "auto"
  --suppress_blank SUPPRESS_BLANK
                        common decoding parameters
  --suppress_non_speech_tokens SUPPRESS_NON_SPEECH_TOKENS
                        common decoding parameters
  --temperature TEMPERATURE
                        initial decoding temperature
  --max_initial_ts MAX_INITIAL_TS
                        max_initial_ts
  --length_penalty LENGTH_PENALTY
                        length_penalty
  --temperature_inc TEMPERATURE_INC
                        temperature_inc
  --entropy_thold ENTROPY_THOLD
                        similar to OpenAI's "compression_ratio_threshold"
  --logprob_thold LOGPROB_THOLD
                        logprob_thold
  --no_speech_thold NO_SPEECH_THOLD
                        no_speech_thold
  --greedy GREEDY       greedy
  --beam_search BEAM_SEARCH
                        beam_search

```

## Assistant

This is a simple example showcasing the use of `pywhispercpp` as an assistant.
The idea is to use a `VAD` to detect speech (in this example we used webrtcvad), and when some speech is detected,
we run the transcription.  
It is inspired from the [whisper.cpp/examples/command](https://github.com/ggerganov/whisper.cpp/tree/master/examples/command) example.

You can check the source code [here](https://github.com/abdeladim-s/pywhispercpp/blob/main/pywhispercpp/examples/assistant.py) 
or you can use the class directly to create your own assistant:


```python
from pywhispercpp.examples.assistant import Assistant

my_assistant = Assistant(commands_callback=print, n_threads=8)
my_assistant.start()
```
Here we set the `commands_callback` to a simple `print`, so the commands will just get printed on the screen.

You can run this example from the command line as well

```shell
$ pwcpp-assistant --help

usage: pwcpp-assistant [-h] [-m MODEL] [-ind INPUT_DEVICE] [-st SILENCE_THRESHOLD] [-bd BLOCK_DURATION]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Whisper.cpp model, default to tiny.en
  -ind INPUT_DEVICE, --input_device INPUT_DEVICE
                        Id of The input device (aka microphone)
  -st SILENCE_THRESHOLD, --silence_threshold SILENCE_THRESHOLD
                        he duration of silence after which the inference will be running, default to 16
  -bd BLOCK_DURATION, --block_duration BLOCK_DURATION
                        minimum time audio updates in ms, default to 30
```

## Recording 
Another simple [example](https://github.com/abdeladim-s/pywhispercpp/blob/main/pywhispercpp/examples/recording.py) to transcribe your own recordings.

You can use it from Python as follows:

```python
from pywhispercpp.examples.recording import Recording

myrec = Recording(5)
myrec.start()
```

Or from the command line:
    
```shell
$ pwcpp-recording --help

usage: pwcpp-recording [-h] [-m MODEL] duration

positional arguments:
  duration              duration in seconds

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Whisper.cpp model, default to tiny.en
```
## Live Stream Transcription
This [example](https://github.com/abdeladim-s/pywhispercpp/blob/main/pywhispercpp/examples/livestream.py) is an attempt to transcribe a livestream in realtime, but the results are not quite satisfactory yet, the CPU jumps quickly to 100% and I cannot use huge models on my descent machine.
(Or maybe I am doing something wrong!) :sweat_smile:

If you have a powerful machine, give it a try.

From python :
```python
from pywhispercpp.examples.livestream import LiveStream

url = ""  # Make sure it is a direct stream URL
ls = LiveStream(url=url, n_threads=4)
ls.start()
```

From the command line:

```shell
$ pwcpp-livestream --help

usage: pwcpp-livestream [-h] [-nt N_THREADS] [-m MODEL] [-od OUTPUT_DEVICE] [-bls BLOCK_SIZE] [-bus BUFFER_SIZE] [-ss SAMPLE_SIZE] url

positional arguments:
  url                   Stream URL

options:
  -h, --help            show this help message and exit
  -nt N_THREADS, --n_threads N_THREADS
                        number of threads, default to 3
  -m MODEL, --model MODEL
                        Whisper.cpp model, default to tiny.en
  -od OUTPUT_DEVICE, --output_device OUTPUT_DEVICE
                        the output device, aka the speaker, leave it None to take the default
  -bls BLOCK_SIZE, --block_size BLOCK_SIZE
                        block size, default to 1024
  -bus BUFFER_SIZE, --buffer_size BUFFER_SIZE
                        number of blocks used for buffering, default to 20
  -ss SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        Sample size, default to 4
```

# Advanced usage
* First check the [API documentation](https://abdeladim-s.github.io/pywhispercpp/) for more advanced usage.
* If you are a more experienced user, you can access the [C-Style API](https://github.com/ggerganov/whisper.cpp/blob/master/whisper.h) directly, almost all functions from `whisper.h`
are exposed with the binding module `_pywhispercpp`.

```python
import _pywhispercpp as pwcpp

ctx = pwcpp.whisper_init_from_file('path/to/ggml/model')
```

# Discussions and contributions
If you find any bug, please open an [issue](https://github.com/abdeladim-s/pywhispercpp/issues).

If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions](https://github.com/abdeladim-s/pywhispercpp/discussions) and open a new topic.

# License

This project is licensed under the same license as [whisper.cpp](https://github.com/ggerganov/whisper.cpp/blob/master/LICENSE) (MIT  [License](./LICENSE)).


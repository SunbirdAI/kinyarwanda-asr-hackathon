# Kinyarwanda ASR Hackathon entry: Sunbird AI

## Team

[Sunbird AI](https://sunbird.ai) is a non-profit research team based in Uganda, where we create open models and systems for social benefit.
Our main focus at the moment is in developing models to understand local languages (ASR, translation, LLMs),
so we were excited to see this hackathon. Thanks to [Digital Umuganda](https://digitalumuganda.com/) for making it happen!

The team within Sunbird who worked on the hackathon (in alphabetical order) were Benjamin Akera, Evelyn Nafula Ouma, John Quinn and Patrick Walukagga.

## Overview

 ✅ What worked:

- openai/whisper-large-v3 as a base model.
- Using the model to remove as many mislabelled training examples as we could and then re-training.
- Augmenting audio by adding noise and changing speed.
- Training the model to make lower-case, unpunctuated predictions.
- Recompressing the audio files to make them easier to work with.

❌ What didn't work:

- Using external datasets for track C. We trained on all of Common Voice Kinyarwanda (a million examples!) but didn't see any improvement. Hypothesis: the hackathon data was all in a similar style, with people reacting to image prompts; mixing in other types of data (Common Voice is prompted speech, where people read out sentences) probably helped to get a more robust model for practical purposes, but was out of domain for this test set. One possibility would have been to first train on Common Voice, as a kind of pretraining, and then the hackathon data, but we ran out of time.
- MMS as a base model. We thought that a CTC model might be suited to this competition, as it produces lower-case, unpunctuated text, and had previous experience of good WER and CER metrics for other languages, but in practice couldn't get WER below 0.2.
- Phi-4 as a base model. This seems promising, but we only had time for a brief exploration.
  
❓ Not sure/not enough time to complete:

- Trimming silences from the beginning and end of the audio.
- Using the image categories as text prompts in Whisper.
- Using the unlabelled Track C audio for semi-supervised learning.

## Training details

### Text preprocessing

Since the competition metrics are based on WER and CER, and these don't depend on capital letters or punctuation such as full stops,
we used lower case unpunctuated text to train the model, so that all the model's capacity is spent on getting the characters/words right.

### Silence removal

We used Silero VAD to find non-speech parts of the audio. Whisper is essentially an LLM conditioned on audio,
so for inputs with long periods of silence or non-speech, it loses this conditioning and has a tendency to hallucinate.

### Training schedule

We used learning rate 1e-5, batch size 32, and began from [openai/whisper-large-v3](htts://hf.co/openai/whisper-large-v3), a model with no Kinyarwanda understanding.
Later we saw clarification from the organisers that it was allowed to fine-tune existing Kinyarwanda models, and
so we could have used our own [Sunbird/asr-whisper-large-v3-salt](https://hf.co/Sunbird/asr-whisper-large-v3-salt) as a starting point,
which may have improved the results - but by this time we'd already done the main training and it seemed to work OK.

The supplied dev_test validation split was huge, with more than 9000 examples.
We used only a few hundred of these for validation, and the rest added to the training set.

After three epochs, validation loss did not improve.
We did full parameter fine tuning on single H100 GPUs
(not necessary though: we've found this model also to be trainable on 48GB cards such as RTX 6000 Ada; LoRA training is also possible).

## Filtering the training set

We estimate around 3% of the examples had the wrong label, i.e. the text transcription seemed to be for a different audio file.
Therefore we ran an earlier version of the model on as many of the training examples as we could, and looked for major discrepancies
between the model prediction and the text label (WER > 90%).
We had time to process 125,000 of the training examples and removed about 4,000 examples this way.

## GPU compute time

The longest training run was 30 hours on a single H100 GPU.

## Model size and inference speed

Model parameter count: 1.3B

TODO: Get A100 real time factor

# Kinyarwanda ASR Hackathon: 1st place solution

## Team

[Sunbird AI](https://sunbird.ai) is a non-profit research team based in Uganda, where we create open models and systems for social benefit.
Our main focus at the moment is in developing models to understand African languages (ASR, translation, LLMs),
so we were excited to see this hackathon. Thanks to [Digital Umuganda](https://digitalumuganda.com/) for making it happen!

The team within Sunbird who worked on the hackathon (in alphabetical order) were Benjamin Akera, Evelyn Nafula Ouma, Gilbert Yiga and John Quinn.

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

The model used for making the predictions is [here](https://huggingface.co/jq/whisper-large-v3-kin-track-b). For practical applications involving Kinyarwanda ASR, a better model is likely [this one](jq/whisper-large-v3-kin-nyn-lug-xog), which was an earlier Track C submission that is probably more robust in real-world conditions, being trained on other datasets including Common Voice and some related languages from Uganda with different speaking styles, and more realistic audio augmentation including representative background noise.

## Filtering the training set

Because the dataset was pretty big, and with some variation in file types (mostly webm but a few mp3s and other formats) we started by recompressing it to .ogg, which kept the quality but halved the size of the dataset and made it easier to experiment with. The size could actually be halved again by reducing the sample rate, since all the models we tried used 16 KHz audio, but we kept it at the original sample rates.

While this hackathon dataset is an amazing resource and overall seemed to be well transcribed, there was some label noise that would limit the accuracy of any model trained on it. Most importantly, we noticed that many of the examples seemed to have the wrong label, i.e. the text transcription seemed to be for a different audio file.
To try to mitigate this, we ran an earlier version of the model on as many of the training examples as we could, and looked for major discrepancies
between the model prediction and the text label. If WER was greater than 90% then we dropped that example.
We had time to process 125,000 of the training examples in this way and removed about 4,000 examples.
If we'd been able to run this on the rest of the training data then we'd have had a bit more improvement - and it would have been better still if we could have worked out which audio samples the mismatched labels belonged to so that we could have fixed them instead of dropping them.
We also excluded examples where the ratio of the text length to the audio duration was extremely high or low.

## Training details

### Text preprocessing

Since the competition metrics are based on WER and CER, and these don't depend on capital letters or punctuation such as full stops,
we used lower case unpunctuated text to train the model, so that all the model's capacity is spent on getting the characters/words right.

### Silence removal

We used Silero VAD to find non-speech parts of the audio. Whisper is essentially an LLM conditioned on audio,
so for inputs with long periods of silence or non-speech, it loses this conditioning and has a tendency to hallucinate.
We're still unsure if this helps though - we got better validation results when removing non-speech segments, but nothing changed on the public leaderboard.

### Audio augmentation

We add random noise to the audio samples to add some variation and make the training task a little more difficult. In earlier experiments we added realistic background noise from our open [Urban Noise Uganda](https://huggingface.co/datasets/Sunbird/urban-noise-uganda-61k) dataset collected in Kampala, but changed this later to just add white noise, as the test examples are all fairly free of background noise.

### Training schedule

We used learning rate 1e-5, batch size 32, and began from [openai/whisper-large-v3](htts://hf.co/openai/whisper-large-v3), a model with no Kinyarwanda understanding.
Later we saw clarification from the organisers that it was allowed to fine-tune existing Kinyarwanda models, and
so we could have used our own [Sunbird/asr-whisper-large-v3-salt](https://hf.co/Sunbird/asr-whisper-large-v3-salt) as a starting point,
which may have improved the results - but by this time we'd already done the main training and it seemed to work OK.

The supplied `dev_test` validation split was huge, with more than 9000 examples.
We used only a few hundred of these for validation, and the rest added to the training set.

After three epochs, validation loss stopped improving.
We did full parameter fine tuning on single H100 GPUs
(not necessary though: we've found this model also to be trainable on 48GB cards such as RTX 6000 Ada; LoRA training is also possible).

Inference was using beam search (`num_beams=5`), using a pipeline for examples greater than 30 seconds, and directly with `model.generate()` otherwise.
On validation data, we got a WER of 8.8% and CER of 2.2%.

## GPU compute time

The longest training run was 30 hours on a single H100 GPU. We used <150 GPU hours (with different specs, including smaller GPUs for explorations) for the whole hackathon.

## Model size and inference speed

Model parameter count: 1.3B

A100 real time factor is about 0.3 (processing 10 minutes of audio takes about 3 minutes). This is with batch size 1, and with no torch.compile, so can be speeded up. On an H100 we get about 10x real time.

# QuantizedLSTM

This repository contains code and models for quantized LSTMs based on [Hello Edge: Keyword Spotting on Microcontrollers](https://arxiv.org/abs/1711.07128) and [PACT: Parameterized Clipping Activation for Quantized Neural Networks}(https://arxiv.org/abs/1805.06085). Results are obtained on the [Google Speech Commands v0.2 data set](https://arxiv.org/abs/1804.03209).

## Training

Specify your parameters either in KWS_LSTM.py or as command line parameters and then run. Note all weights are stored in full precision but are quantized during training and inference.

```
python KWS_LSTM.py
```

## Results

|                   | #1       | #2       | #3       | Avg.     |
|-------------------|----------|----------|----------|----------|
| 118, 8W, 8A, PACT | 88.3500% | 89.7700% | 89.1800% | 89.1000% |
| 118, 8W, 8A       | 89.3000% | 89.4700% | 90.2800% | 89.6833% |
| 344, 8W, 8A, PACT | 91.3000% | 91.0300% | 91.9500% | 91.4267% |
| 344, 8W, 8A       | 91.7700% | 91.1000% | 91.4600% | **91.4433%** |

The first numbers shows the number of hidden LSTM units. PACT indicates whether the ranges of the quantized variables were trained or not. All models are trained with 8 bit weights and activations.
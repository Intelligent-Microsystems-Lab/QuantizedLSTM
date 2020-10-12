# QuantizedLSTM

This repository contains code and models for quantized LSTMs based on [Hello Edge: Keyword Spotting on Microcontrollers](https://arxiv.org/abs/1711.07128) and [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085). Results are obtained on the [Google Speech Commands v0.2 data set](https://arxiv.org/abs/1804.03209).

## Training

Specify your parameters either in KWS_LSTM.py or as command line parameters and then run. Note all weights are stored in full precision but are quantized during training and inference.

```
python KWS_LSTM.py
```

## Results

| Config            | #1       | #2       | #3       | Avg.     |
|-------------------|----------|----------|----------|----------|
| 118, 8W, 8A, PACT | 88.3500% | 89.7700% | 89.1800% | 89.1000% |
| 118, 8W, 8A       | 89.3000% | 89.4700% | 90.2800% | 89.6833% |
| 344, 8W, 8A, PACT | 91.3000% | 91.0300% | 91.9500% | 91.4267% |
| 344, 8W, 8A       | 91.7700% | 91.1000% | 91.4600% | **91.4433%** |

The first number of the config shows the number of hidden LSTM units. PACT indicates whether the ranges of the quantized variables were trained or not. All models are trained with 8 bit weights and activations.

| Config            | Trial       | UUID of checkpoint file      |
|-------------------|-------------|------------------------------|
| 118, 8W, 8A, PACT | 1           | 2fc3aec6-b7e3-4deb-baf9-5857c31fa0ac|
| 	                | 2 	      | ea886fba-b40f-4a23-afea-f4bd113ab667 |
| 	                | 3           | 4896e49d-eed6-4e82-b28c-480e5c3d5269 |
| 118, 8W, 8A       | 1           | 63223345-9304-4f79-926d-24a23b5cbe44 |
| 	                | 2           | 5de899ec-a086-4025-b44d-54944b2e576f |
| 	                | 3           | 1d08d7a6-71fe-4843-a777-987e2d0bf724 |
| 344, 8W, 8A, PACT | 1           | 01bcd8cd-0e36-466d-8892-4772b2a21aec |
| 	                | 2           | ce893f32-9fb9-4c8f-a7a2-e3529e58cd90 |
| 	                | 3           | 01315410-c3a8-4fa1-83af-cf43c6bb2bf8 |
| 344, 8W, 8A       | 1           | 4e382bdd-053a-4c27-9ff5-0e1a8f80785c |
| 	                | 2           | 2f27fb25-01dc-4bdd-ade4-062d452cf913 |
| 	                | 3           | 993ac3f4-805a-41bc-a768-a28baea9893b |
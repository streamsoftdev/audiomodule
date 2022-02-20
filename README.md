# AudioModule

The base class for audio modules. Generally, an audio module receives signals on its inputs and produces signals on its outputs.

## Inputs and Outputs

Audio modules have a number of inputs and outputs. Each input and output has a number of channels, where each channel is a signal. Each output can be connected to any number of inputs of other audio modules. An input can only have one output connected to it. All inputs and outputs buffer data using the `Buffer` class.

## Fundamental parameters

Every audio module has a sampling rate, chunk size and data type. In practice however all of these parameters will be the same for all audio modules that are connected together.

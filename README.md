# Sim agents MTR-AutoBots (sima_agents_mtr_autobots_gr25)

MTR-AutoBots is a project that combines the Motion Transformer Model (MTR) and AutoBots to enhance motion forecasting in autonomous driving scenarios. This repository aims to leverage the strengths of both models to improve the accuracy and performance of motion prediction.

## Features

- **Data Format**: The code utilizes the data format from the Motion Transformer Model (MTR). For detailed explanations and data format, please refer to the `mtr/datasets` directory.

- **Preprocessing**: Prior to the encoder, the agent and map data undergo a preprocessing step using two polyline encoders, similar to the implementation in MTR. However, the output layer has been modified to align with the inputs of AutoBots. MTR combines all timestamps with a max pool, which has been removed in MTR-AutoBots to retain the temporal dimension, similar to AutoBots.

- **Masking**: AutoBots and MTR employ different masking techniques, where setting `False` in MTR corresponds to `True` in AutoBots. To account for this difference, the masks are inverted before the AutoBots encoder. Additionally, the attention layer encoder for the map is retained to preserve the map feature's shape before the decoder.

- **Loss Function**: The loss function used in MTR-AutoBots is derived from AutoBots. The model output consists of x, y, vx, and vy values. It is important to note that the sim agents metric utilizes x, y, z, and heading for evaluation. While we initially attempted to directly modify the model output to include these values, we encountered issues with the loss function. As a result, we decided to keep the output with x, y, vx, and vy and compute the heading from the x and y values. The z value is set to the same value as the last observed trajectory.

- **Model Implementation**: The model encoder and decoder in MTR-AutoBots are based on the AutoBots model. The shape of the tensors and variables has been adjusted to be compatible with Waymo.

## Sim Agents Metric

The implementation of the sim agents metric can be found in `tools/sim_agents_metric.py`. This file utilizes the preprocessed input data to recreate scenarios for the sim agents metric. The main challenge in this implementation was understanding the Waymo data format and correctly assigning values to the scenario variables. We have implemented the necessary functions to achieve this.

## Links

- AutoBots GitHub: [https://github.com/roggirg/AutoBots](https://github.com/roggirg/AutoBots)
- MTR GitHub: [https://github.com/sshaoshuai/MTR](https://github.com/sshaoshuai/MTR)

author: Alicia Mauroux and Xavier Nal

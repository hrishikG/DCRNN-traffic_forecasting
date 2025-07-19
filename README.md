# DCRNN for Spatio-Temporal Traffic Forecasting

This project implements a Diffusion Convolutional Recurrent Neural Network (DCRNN) to predict future traffic patterns based on historical speed data from the **METR-LA** dataset. The model combines spatial dependencies (via diffusion convolution over road networks) and temporal dynamics (via GRU-style recurrent units) to forecast traffic congestion.

## 📊 Dataset

- **METR-LA**: Contains traffic data from 207 loop detectors on highways in Los Angeles, collected every 5 minutes over several months.
- Provided files:
  - `METR-LA.h5`: Sensor speed measurements.
  - `adj_METR-LA.pkl`: Road network adjacency matrix (used to compute diffusion supports).

## 🚀 Model Overview

The DCRNN model uses:
- **Diffusion Convolution** to model spatial flow over the graph.
- **GRU-like Recurrent Units** to capture temporal dependencies.
- **Encoder-Decoder** architecture to predict multiple future time steps.

## 🔧 Key Implementation Details

- Custom implementation of **scaled symmetric Laplacian** with memory-safe element-wise operations to avoid kernel crashes.
- Both **forward** and **reverse** Laplacians are used to enable bidirectional diffusion.
- Uses teacher forcing during training and supports inference without ground truth.

## 🛠 Installation

```bash
pip install -r requirements.txt
 Sample Output
Encoder output shape: (batch_size, num_nodes, hidden_dim)

Final prediction shape: (batch_size, forecast_horizon, num_nodes, 1)

📚 References
[Yu et al., 2018] Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. arXiv:1709.04875

Original repo: liyaguang/DCRNN

🧠 Author
Hrishik Guha
Aspiring Data Scientist | ML & Time Series Enthusiast
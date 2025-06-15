# Lotto Number Predictor

This project uses deep learning (Keras/TensorFlow) to predict German Lotto numbers based on historical draw data, leveraging a Temporal Fusion Transformer (TFT) architecture with mixed precision for Apple Silicon (M1/M2/M4) acceleration.

---

## Features

- **Automatic download and parsing** of the complete Lotto draw archive.
- **Sequence modeling** using all available variables (not just the main numbers).
- **Date feature encoding** for temporal context.
- **Temporal Fusion Transformer** (TFT) model with LSTM and attention layers.
- **Mixed precision** support for Apple Silicon (Metal backend).
- **Model checkpointing, early stopping, and learning rate scheduling**.
- **Prediction with temperature scaling** for more diverse outputs.

---

## Requirements

- Python 3.8+
- [TensorFlow 2.x](https://www.tensorflow.org/install)
- numpy
- pandas

Install dependencies with:

```sh
pip install tensorflow numpy pandas
```

---

## Usage

Run the main script:

```sh
python3 lotto.py
```

The script will:

1. Download and parse the Lotto archive.
2. Prepare the data for sequence modeling.
3. Build and train the TFT model (or load a saved model if available).
4. Predict the next set of Lotto numbers based on the latest draws.

---

## Project Structure

- `lotto.py` — Main script for data processing, model training, and prediction.
- `best_lotto_model_all_vars_with_date.keras` — Saved model checkpoint (created after training).

---

## Data Details

- Data is loaded from [Johannes Friedrich's LottoNumberArchive](https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json).
- Each draw is grouped by ID and variable (e.g., "Lottozahl").
- Only draws with exactly 6 main numbers are used for training.

---

## Model Details

- **Inputs:** Sequences of past draws for all variables, plus encoded date features.
- **Architecture:** Embedding layers → LSTM → 2x TFT blocks → Dense layers.
- **Output:** Multi-label (multi-hot) prediction for the next 6 Lotto numbers.
- **Loss:** Binary cross-entropy with label smoothing.

---

## Apple Silicon Support

- Mixed precision is enabled for optimal performance on Apple M1/M2/M4 chips.
- GPU/Metal memory growth is configured automatically.

---

## Example Output

```
Variables found in dataset: ['Lottozahl', ...]
Total usable draws with Lottozahl (6 numbers): 32400
Total training samples prepared: 32380

📅 Example Lotto training samples with dates:
09.10.1955 -> [3, 12, 13, 16, 23, 41]
...

 Predicted next Lotto numbers:
[2, 7, 14, 23, 36, 44]
```

---

## Customization

- **Change model parameters** (timesteps, embedding size, etc.) at the top of `lotto.py`.
- **Adjust temperature** in `predict_numbers()` for more or less randomness in predictions.

---

## License
```
This is a fun project, Do whatever you want XD
```
---

## Disclaimer

**This project does not guarantee winning Lotto numbers.**  
It is a demonstration of sequence modeling and deep learning on real-world data.

---

## Author

- [Soumyadip Banerjee](https://philosopherscode.de/)
- Dataset Provider: [Johannes Friedrich](https://johannesfriedrich.github.io/).

---

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [LottoNumberArchive](https://johannesfriedrich.github.io/LottoNumberArchive/)

---

Feel free to open issues or contribute improvements!

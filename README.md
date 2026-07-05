# LDPC Encoder and Decoder in Python

A Python implementation of an **LDPC (Low-Density Parity-Check) Encoder and Decoder** built from scratch using **NumPy** and **Matplotlib**, without any external LDPC libraries. The project demonstrates the complete digital communication chain, from encoding to decoding over an AWGN channel.

---

## Features

* LDPC encoding using a parity-check matrix
* GF(2) arithmetic and Gaussian elimination
* Generator matrix construction
* BPSK modulation
* AWGN channel simulation
* LLR computation
* Min-Sum, Sum-Product, and Bit-Flipping decoders
* BER vs. SNR performance analysis
* Modular code structure for easy understanding and future FPGA/Verilog implementation

---

## Project Structure

```text
LDPC_Project/
│── encoder.py
│── decoder.py
│── README.md
└── results/
    ├── ber_curve.png
    ├── encoder_output.png
    └── decoder_output.png
```

---

## Working Flow

```text
Message
   ↓
LDPC Encoder
   ↓
BPSK Modulation
   ↓
AWGN Channel
   ↓
LLR Computation
   ↓
LDPC Decoder
   ↓
Recovered Message
```

---

## Requirements

* Python 3.x
* NumPy
* Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

Run the project:

```bash
python encoder.py
python decoder.py
```

---

## Results

Include the following in the `results/` folder:

* **BER vs. SNR Curve** (`ber_curve.png`)
* **Encoder Output Screenshot**
* **Decoder Output Screenshot**

Display the BER curve in the README:

```markdown
## BER Performance

![BER Curve](results/ber_curve.png)
```

You can also add screenshots:

```markdown
## Encoder Output

![Encoder](results/encoder_output.png)

## Decoder Output

![Decoder](results/decoder_output.png)
```

To save the BER graph automatically:
<img width="1366" height="655" alt="BER" src="https://github.com/user-attachments/assets/8046343b-c314-47cd-931e-2174de520dab" />


---

## Future Scope

* IEEE standard LDPC matrices
* Layered and Offset Min-Sum decoding
* Hardware (FPGA/Verilog) implementation
* Performance optimization for large block lengths

---

## Author

**Rohit Sharma**

This project was developed as a learning-oriented implementation of LDPC encoding and decoding, with a focus on understanding modern error-correcting codes and their practical performance over noisy communication channels.

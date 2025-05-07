# CPA Power Analysis on AES Hardware Implementations (Unprotected, MS1, MS2)
# Authors: Nilay Jadav, Zachary Weinstein
 
This project demonstrates **Correlation Power Analysis (CPA)** attacks on AES-128 hardware implementations using the AESPTv2 dataset (STM32, D1). We provide ready-to-run code and methodology for:
- **Unprotected AES** (classic CPA)
- **MS1 (masked, weak) AES** (first-order CPA)
- **MS2 (masked, robust) AES** (second-order CPA, with discussion)

> **Note:** For robust masking (MS2), classic and second-order CPA are generally ineffective. Please take a look below for details and alternatives.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset Used](#dataset-used)
- [Attack Methods](#attack-methods)
  - [Unprotected (Classic CPA)](#unprotected-classic-cpa)
  - [MS1 (Masked, Weak, First-Order CPA)](#ms1-masked-weak-first-order-cpa)
  - [MS2 (Masked, Robust, Second-Order CPA)](#ms2-masked-robust-second-order-cpa)
- [How to Run](#how-to-run)
- [References](#references)

---

## About the Project

This repository contains code and instructions to perform side-channel power analysis attacks on AES-128 hardware implementations. It is based on the [AESPTv2 dataset](https://github.com/urioja/AESPTv2) and demonstrates the practical insecurity of unprotected and weakly-masked cryptographic hardware against physical attacks.

---

## Dataset Used

We use the **AESPTv2 D1** dataset, which contains:
- **Unprotected**: Standard AES, no countermeasures.
- **MS1**: First-order masked AES (less robust).
- **MS2**: Strongly masked AES (robust, designed to defeat classic CPA).
- Each group contains `Attack` and `Profiling` sets. We use the `Attack` set for CPA.

**File structure example:**
D1/Unprotected/Attack/Traces
D1/Unprotected/Attack/MetaData
D1/MS1/Attack/Traces
D1/MS1/Attack/MetaData
D1/MS2/Attack/Traces
D1/MS2/Attack/MetaData


---

## Attack Methods

### Unprotected (Classic CPA)

- **Attack:** First-order CPA on the Hamming Weight of the S-box output.
- **Approach:** For each key byte, guess all 256 values, compute HW(Sbox[PT^K]), and correlate with measured traces.
- **Expected result:** Recovers the full AES key with a few thousand traces.

### MS1 (Masked, Weak, First-Order CPA)

- **Attack:** First-order CPA as above.
- **Approach:** Same as unprotected, but masking makes the attack harder. More traces are needed.
- **Expected result:** Key recovery is possible, but may require tens of thousands of traces.

### MS2 (Masked, Robust, Second-Order CPA)

- **Attack:** Second-order CPA using the centered product of two Points of Interest (POIs).
- **Approach:** Compute centered product of two trace samples per trace, then correlate with HW(Sbox[PT^K]).
- **Expected result:** Key recovery is generally **not possible** with classic or second-order CPA due to strong masking. For MS2, profiling attacks (template or deep learning) or specialized attacks (e.g., MixColumns, as in the NAE0010 whitepaper) are required.

---

## How to Run

### 1. **Prepare the Dataset**

- Download and extract the D1 `.h5` file from [AESPTv2](https://github.com/urioja/AESPTv2).
- Upload the `.h5` file to your Google Drive.

### 2. **Open and Run the Colab Notebook**

- Mount your Google Drive in Colab.
- Use the code snippets below for each attack type.
- **Update the `h5_path` variable** to point to your `.h5` file.

#### **Sample: Load Unprotected Attack Set**

import h5py
import numpy as np

h5_path = '/content/drive/MyDrive/path_to_your_folder/D1.h5' # Update path
with h5py.File(h5_path, 'r') as f:
traces = f['D1/Unprotected/Attack/Traces'][:]
metadata = f['D1/Unprotected/Attack/MetaData'][:]
plaintexts = np.array([row['plaintext'] for row in metadata])
keys = np.array([row['key'] for row in metadata])


#### **Sample: Classic CPA Attack**

AES_SBOX = np.array([...], dtype=np.uint8) # Full S-box here

def hamming_weight(x): return bin(x).count('1')
def leakage_model(pt_byte, key_guess): return hamming_weight(AES_SBOX[pt_byte ^ key_guess])

def cpa_attack(traces, plaintexts, target_byte, num_traces=5000):
traces = traces[:num_traces]
plaintexts = plaintexts[:num_traces, target_byte]
hypotheses = np.zeros((256, num_traces))
for kg in range(256):
hypotheses[kg] = [leakage_model(pt, kg) for pt in plaintexts]
traces_norm = (traces - traces.mean(axis=0)) / traces.std(axis=0)
hyp_norm = (hypotheses - hypotheses.mean(axis=1, keepdims=True)) / hypotheses.std(axis=1, keepdims=True)
corr_matrix = np.dot(hyp_norm, traces_norm) / num_traces
best_guess = np.argmax(np.max(np.abs(corr_matrix), axis=1))
return best_guess, corr_matrix



#### **Sample: Second-Order CPA for MS2**

def centered_product(x, y): return (x - np.mean(x)) * (y - np.mean(y))

def second_order_cpa(traces, plaintexts, target_byte, poi1, poi2, num_traces=10000):
x = traces[:num_traces, poi1]
y = traces[:num_traces, poi2]
cp = centered_product(x, y)
plaintexts = plaintexts[:num_traces, target_byte]
hypotheses = np.zeros((256, num_traces))
for kg in range(256):
sbox_out = AES_SBOX[plaintexts ^ kg]
hypotheses[kg] = [hamming_weight(val) for val in sbox_out]
hyp_norm = (hypotheses - hypotheses.mean(axis=1, keepdims=True)) / hypotheses.std(axis=1, keepdims=True)
cp_norm = (cp - np.mean(cp)) / np.std(cp)
corrs = np.array([np.corrcoef(hyp_norm[kg], cp_norm) for kg in range(256)])
best_guess = np.argmax(np.abs(corrs))
return best_guess, corrs



---

## Limitations and Notes

- **Unprotected and MS1:** Classic CPA works, but MS1 may need more traces.
- **MS2:** Classic and second-order CPA are generally ineffective. For MS2, use profiling attacks (template, deep learning) or specialized MixColumns attacks (see [NAE0010 Whitepaper](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/35295567/086907eb-00b9-4fe8-bdf6-e3dc08b636ff/NAE0010_Whitepaper_CW305_AES_SCA_Attack.pdf)).
- **Profiling attacks** require using the `/Profiling/Traces` and `/Profiling/MetaData` sets for training.

---

## References

- [AESPTv2 Dataset](https://github.com/urioja/AESPTv2)
- [NAE0010 Whitepaper: Power Analysis on FPGA Implementation of AES Using CW305 & ChipWhisperer®](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/35295567/086907eb-00b9-4fe8-bdf6-e3dc08b636ff/NAE0010_Whitepaper_CW305_AES_SCA_Attack.pdf)
- [ChipWhisperer Docs](https://chipwhisperer.readthedocs.io/en/latest/)
- [eshard blog on masked SCA datasets](https://eshard.com/posts/masked-and-shuffled-dataset-for-sca-part2)

---

## License

This project is for academic and educational use. Please cite the AESPTv2 dataset and the NAE0010 whitepaper if you use this material in your research.

---

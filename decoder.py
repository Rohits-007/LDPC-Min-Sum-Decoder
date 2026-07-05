# Decoder

import numpy as np
import matplotlib.pyplot as plt

from encoder import (
    create_example_parity_matrix,
    systematic_form,
    generator_matrix,
    encode,
    restore_order,
    gf2_dot,
    print_title,
    print_matrix
)

def print_title(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def print_vector(name, vector):
    print(f"\n{name} =")
    print(vector)

def bpsk_modulate(bits):
    return 1 - 2 * bits

def hard_decision(symbols):
    return (symbols < 0).astype(int)

def awgn_channel(symbols, snr_db):

    snr = 10 ** (snr_db / 10)

    sigma = np.sqrt(1 / (2 * snr))

    noise = sigma * np.random.randn(len(symbols))

    received = symbols + noise

    return received, sigma

def compute_llr(received, sigma):

    variance = sigma ** 2

    llr = (2 * received) / variance

    return llr

def build_graph(H):

    check_nodes = []
    variable_nodes = []

    rows, cols = H.shape

    for i in range(rows):
        check_nodes.append(np.where(H[i] == 1)[0])

    for j in range(cols):
        variable_nodes.append(np.where(H[:, j] == 1)[0])

    return check_nodes, variable_nodes

def min_sum_decode(H, llr, iterations=20):

    rows, cols = H.shape

    q = np.zeros((rows, cols))
    r = np.zeros((rows, cols))

    q[H == 1] = llr[np.where(H == 1)[1]]

    for _ in range(iterations):

        for i in range(rows):

            cols_i = np.where(H[i])[0]

            for j in cols_i:

                others = cols_i[cols_i != j]

                sign = np.prod(np.sign(q[i, others]))

                minimum = np.min(np.abs(q[i, others]))

                r[i, j] = sign * minimum

        L = llr.copy()

        for j in range(cols):

            rows_j = np.where(H[:, j])[0]

            L[j] += np.sum(r[rows_j, j])

            for i in rows_j:
                q[i, j] = L[j] - r[i, j]

        bits = (L < 0).astype(int)

        if np.all(H @ bits % 2 == 0):
            break

    return bits

def sum_product_decode(H, llr, iterations=20):

    rows, cols = H.shape

    q = np.zeros((rows, cols))
    r = np.zeros((rows, cols))

    q[H == 1] = llr[np.where(H == 1)[1]]

    for _ in range(iterations):

        for i in range(rows):

            cols_i = np.where(H[i])[0]

            for j in cols_i:

                others = cols_i[cols_i != j]

                product = np.prod(np.tanh(q[i, others] / 2))

                product = np.clip(product, -0.999999, 0.999999)

                r[i, j] = 2 * np.arctanh(product)

        L = llr.copy()

        for j in range(cols):

            rows_j = np.where(H[:, j])[0]

            L[j] += np.sum(r[rows_j, j])

            for i in rows_j:
                q[i, j] = L[j] - r[i, j]

        bits = (L < 0).astype(int)

        if np.all(H @ bits % 2 == 0):
            break

    return bits

def bit_flipping_decode(H, received, iterations=20):

    bits = hard_decision(received)

    for _ in range(iterations):

        syndrome = (H @ bits) % 2

        if np.all(syndrome == 0):
            break

        metric = H.T @ syndrome

        flip = np.argmax(metric)

        bits[flip] ^= 1

    return bits

def simulate(H, G, snr_range, frames=100):

    ber_ms = []
    ber_sp = []
    ber_bf = []

    k = G.shape[0]

    for snr in snr_range:

        err_ms = 0
        err_sp = 0
        err_bf = 0

        total = 0

        for _ in range(frames):

            msg = np.random.randint(0, 2, k)

            code = (msg @ G) % 2

            tx = bpsk_modulate(code)

            rx, sigma = awgn_channel(tx, snr)

            llr = compute_llr(rx, sigma)

            ms = min_sum_decode(H, llr)
            sp = sum_product_decode(H, llr)
            bf = bit_flipping_decode(H, rx)

            err_ms += np.sum(code != ms)
            err_sp += np.sum(code != sp)
            err_bf += np.sum(code != bf)

            total += len(code)

        ber_ms.append(err_ms / total)
        ber_sp.append(err_sp / total)
        ber_bf.append(err_bf / total)

    return ber_ms, ber_sp, ber_bf

def plot_results(snr, ber_ms, ber_sp, ber_bf):

    plt.figure(figsize=(8, 5))

    plt.semilogy(snr, ber_ms, "o-", label="Min-Sum")
    plt.semilogy(snr, ber_sp, "s-", label="Sum-Product")
    plt.semilogy(snr, ber_bf, "^-", label="Bit-Flipping")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    print_title("LDPC Decoder")

    H = create_example_parity_matrix()

    H_sys, _ = systematic_form(H)

    G = generator_matrix(H_sys)

    snr = np.arange(0, 6)

    ber_ms, ber_sp, ber_bf = simulate(H, G, snr)

    plot_results(snr, ber_ms, ber_sp, ber_bf)
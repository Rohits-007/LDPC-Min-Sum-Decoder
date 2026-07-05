"""
=========================================================
encoder.py

LDPC Encoder implemented completely from scratch.

This file will gradually grow throughout the project.

Current Chapter:
    1. Imports
    2. Utility functions
    3. Parity-check matrix creation (coming next)
    4. Generator matrix construction
    5. Encoding
    6. Verification
=========================================================
"""

# =========================================================
# Import required libraries
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Global Settings
# =========================================================

# Printing matrices becomes much easier
np.set_printoptions(
    linewidth=150,
    suppress=True
)

# =========================================================
# Utility Functions
# =========================================================

def print_title(title):
    """
    Prints a formatted title.

    Example:
    ----------------------------
    Creating Parity Matrix
    ----------------------------
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def print_matrix(name, matrix):
    """
    Nicely prints any matrix.

    Parameters
    ----------
    name : str
        Matrix name

    matrix : numpy array
        Matrix to display
    """

    print(f"\n{name} :")
    print(matrix)

# =========================================================
# Parity check matrix functions
# =========================================================

def create_example_parity_matrix():
    H=np.array([       
        [1,1,0,1,0,0],
        [0,1,1,0,1,0],
        [1,0,1,0,0,1],
    ], dtype=int)
    return H

def is_binary_matrix(H):
    return np.all((H==0)|(H==1))

def matrix_dimensions(H):
    rows, cols = H.shape
    return rows, cols

def display_matrix_information(H):
    rows,cols = matrix_dimensions(H)
    print_title("Parity check Matrix")
    print_matrix("H", H)
    print(f"\nNumber of parity equations: {rows}")
    print(f"\nCodeword length(n): {cols}")
    print(f"\nMessage length estimate: {cols-rows}")
    if is_binary_matrix(H):
        print("\nMatrix validation: PASSED")
    else:
        print("\nMatrix validation: FAILED")

# =========================================================
# GF(2) Matrix Operations
# =========================================================

def gf2_add(a, b):
    """
    Performs addition over GF(2).

    In GF(2):
        0 + 0 = 0
        0 + 1 = 1
        1 + 0 = 1
        1 + 1 = 0

    This is equivalent to the XOR operation.

    Parameters
    ----------
    a : numpy.ndarray
    b : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    return np.bitwise_xor(a, b)


def gf2_dot(A, B):
    """
    Matrix multiplication over GF(2).

    Parameters
    ----------
    A : numpy.ndarray
    B : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    # Ordinary matrix multiplication
    product = np.dot(A, B)

    # Reduce every element modulo 2
    return product % 2


def swap_rows(matrix, row1, row2):
    """
    Swaps two rows of a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
    row1 : int
    row2 : int
    """

    matrix[[row1, row2]] = matrix[[row2, row1]]


def swap_columns(matrix, col1, col2):
    """
    Swaps two columns of a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
    col1 : int
    col2 : int
    """

    matrix[:, [col1, col2]] = matrix[:, [col2, col1]]


def xor_rows(matrix, source_row, target_row):
    """
    Performs

        target_row = target_row XOR source_row

    Parameters
    ----------
    matrix : numpy.ndarray
    source_row : int
    target_row : int
    """

    matrix[target_row] = np.bitwise_xor(
        matrix[target_row],
        matrix[source_row]
    )

def gaussian_elimination_gf2(H):
    """ performs Gaussian elimination on a binary matrix over GF(2) 
    Parameters ----
    H: numpy.ndarray
    
    Returns -----
    H_sys: numpy.ndarray
    Row echelon form of H. """

    H_sys = H.copy() 
    rows, cols = H_sys.shape 

    pivot_row = 0
    pivot_columns=[]

    for col in range(cols):
        if pivot_row>=rows:
            break
        pivot_found = False
    
        for r in range(pivot_row, rows):
            if H_sys[r, col] ==1:
                swap_rows(H_sys, pivot_row, r)
                pivot_found = True
                break

        if not pivot_found:
            continue

        pivot_columns.append(col)

        for r in range(rows):
            if r!=pivot_row and H_sys[r, col]==1:
                xor_rows(H_sys, pivot_row, r)
        pivot_row+=1

    return H_sys, pivot_columns

def systematic_form(H):
    """Systematic form [P|I]
    """
    H_sys=H.copy()
    rows, cols = H_sys.shape
    column_order=list(range(cols))

    pivot_row=0

    for col in range(cols):
        if pivot_row>=rows:
            break
        pivot=0

        for r in range(pivot_row, rows):
            if H_sys[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        swap_rows(H_sys, pivot_row, pivot)

        for r in range(rows):
            if r!=pivot_row and H_sys[r, col]:
                xor_rows(H_sys, pivot_row, r)
        pivot_row+=1
    _, pivot_cols=gaussian_elimination_gf2(H_sys)
    target=cols-rows
    for i,p in enumerate(pivot_cols):
        desired=target+i
        if p!=desired:
            swap_columns(H_sys, p, desired)
            column_order[p],column_order[desired]= column_order[desired],column_order[p]
            pivot_cols = [
                desired if x == p else 
                p if x == desired else
                x for x in pivot_cols
            ]
    return H_sys, column_order

def generator_matrix(H_sys):

    rows, cols = H_sys.shape

    k = cols - rows

    P = H_sys[:, :k]

    I = np.eye(k, dtype=int)

    G = np.concatenate((I, P.T), axis=1)

    return G

def encode(message, G):

    codeword = gf2_dot(message, G)

    return codeword

def restore_order(codeword, column_order):

    restored = np.zeros_like(codeword)

    for new, old in enumerate(column_order):

        restored[old] = codeword[new]

    return restored

def verify_codeword(H, codeword):

    syndrome = gf2_dot(H, codeword.reshape(-1, 1))

    syndrome = syndrome.flatten()

    print("\nSyndrome =", syndrome)

    if np.all(syndrome == 0):

        print("Valid Codeword")

    else:

        print("Invalid Codeword")

    return syndrome




# =========================================================
# Main Program
# =========================================================

if __name__ == "__main__":

    print_title("LDPC Encoder")

    H = create_example_parity_matrix()

    H_sys, column_order = systematic_form(H)

    print_matrix("Systematic H", H_sys)

    G = generator_matrix(H_sys)

    print_matrix("Generator Matrix G", G)

    message = np.array([1, 0, 1])

    print("\nMessage =", message)

    codeword = encode(message, G)

    codeword = restore_order(codeword, column_order)

    print("\nEncoded Codeword =", codeword)

    verify_codeword(H, codeword)


    # print_title("LDPC Encoder")
    # H = create_example_parity_matrix()
    # display_matrix_information(H)
    # print("Project Initialized Successfully.")
    # print_title("Testing GF(2) Operations")

# # -----------------------------------------
# # Test XOR Addition
# # -----------------------------------------

# a = np.array([1, 0, 1, 1])
# b = np.array([1, 1, 0, 1])

# print("\nVector A :", a)
# print("Vector B :", b)

# print("A XOR B :", gf2_add(a, b))

# # # -----------------------------------------
# # # Test Matrix Multiplication
# # # -----------------------------------------

# A = np.array([
#     [1, 0],
#     [1, 1]
# ])

# B = np.array([
#     [1, 1],
#     [0, 1]
# ])

# print("\nMatrix A")
# print(A)

# print("\nMatrix B")
# print(B)

# print("\nGF(2) Product")
# print(gf2_dot(A, B))

# # # -----------------------------------------
# # # Test Row Swap
# # # -----------------------------------------

# test_matrix = H.copy()

# swap_rows(test_matrix, 0, 2)

# print("\nAfter Row Swap")
# print(test_matrix)

# # -----------------------------------------
# # Test Column Swap
# # -----------------------------------------

# test_matrix = H.copy()

# swap_columns(test_matrix, 1, 4)

# print("\nAfter Column Swap")
# print(test_matrix)

# # # -----------------------------------------
# # # Test XOR Row Operation
# # # -----------------------------------------

# test_matrix = H.copy()

# xor_rows(test_matrix, 0, 1)

# print("\nAfter Row XOR")
# print(test_matrix)

# print_title("Gaussian Elimination over GF(2)")

# H_systematic, pivots = gaussian_elimination_gf2(H)

# print("\nOriginal Matrix")
# print(H)
# print("\nRow Echelon Matrix")
# print(H_systematic)
# print("\nPivot Columns")
# print(pivots)

    
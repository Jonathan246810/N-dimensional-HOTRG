import numpy as np
from numpy.linalg import eigh
from ncon import ncon

max_dim = 8
alphabet = "abcdefghijklmnopqrstuvwxyz"
delta_energy_factor = 0.00001

def initial_delta_tensor(total_dimension):

    dim = 2 * total_dimension
    shape = tuple(dim * [2])
    out = np.zeros(shape)
    np.fill_diagonal(out, 1)

    return out

def initial_tensor(total_dimension, coupling_constant):

    a, b, c = np.sqrt(np.cosh(coupling_constant)), np.sqrt(np.sinh(coupling_constant)), -1 * np.sqrt(np.sinh(coupling_constant))
    c_1, c_2 = np.exp(delta_energy_factor * coupling_constant), np.exp(-delta_energy_factor * coupling_constant )
    W = [[a*c_1, b*c_1], [a*c_2, c*c_2]]
    delta = initial_delta_tensor(total_dimension)
    d = 2 * total_dimension

    tensor_array = (d + 1) * [np.array(W)]
    tensor_array[0] = delta
    contraction_array = 1 * [np.arange(1, d+1)]

    for i in range(1, d + 1):
        contraction_array.append(np.array([i, -1*i]))

    out = ncon(tuple(tensor_array), tuple(contraction_array))
    return out

def initial_Mtensor(total_dimension, coupling_constant):

    a, b, c = np.sqrt(np.cosh(coupling_constant)), np.sqrt(np.sinh(coupling_constant)), -1 * np.sqrt(np.sinh(coupling_constant))
    c_1, c_2 = np.exp(delta_energy_factor * coupling_constant), np.exp(-delta_energy_factor * coupling_constant )
    W = [[a * c_1, b * c_1], [a * c_2, c * c_2]]
    delta = initial_delta_tensor(total_dimension)
    d = 2 * total_dimension
    WM = [[a * c_1, b * c_1], [-a * c_2, -c * c_2]]

    tensor_array = (d + 1) * [np.array(W)]
    tensor_array[0] = delta
    tensor_array[d] =np.array(WM)
    contraction_array = 1 * [np.arange(1, d+1)]

    for i in range(1, d + 1):
        contraction_array.append(np.array([i, -1*i]))

    out = ncon(tuple(tensor_array), tuple(contraction_array))
    return out

def give_matrices(tensor, total_dimension):
    """
    :param tensor: tensor from which the hotrg step is started, the first index and the "total_dimension"-th index are the only indices for which no squared matrix needs to be determined
    :param total_dimension: total dimension of the system
    :return: an array of matrices, the first matrix is the (squared) matrix for the second index of tensor and so on
    """

    matrices = []
    for i in range(1, 2*total_dimension):
        if i != total_dimension:
            matrices.append(give_matrix(tensor, total_dimension, i))

    return matrices


def give_matrix(tensor, total_dimension, index):
    """
    :param tensor: see give_matrices
    :param total_dimension: total dimension of the system
    :param index: the number of the index for which the squared matrix needs to be determined
    :return: the squared matrix_index obtained from tensor
    """
    dim = list(np.shape(tensor))[index]
    indices = generate_contraction_indices(total_dimension, index)
    Q = ncon((tensor, tensor), indices[0])
    R = ncon((tensor, tensor), indices[1])
    matrix = ncon((Q, R), ([-1, 1, -3, 2], [1, -2, 2, -4]))
    matrix = matrix.reshape((dim**2, dim**2))

    return matrix

def SVD_Data(matrix):
    """
    :param matrix: matrix obtained after conversion of a tensor
    :return: the U matrix of the SVD and the squared singular values
    """

    S, U = eigh(matrix)
    order = np.argsort(-S)
    S = S[order]
    U = U[:, order]
    S = np.diag(S)

    return U, S

def truncation_tensors(total_dimension, matrices):
    """
    :param total_dimension: dimension of the system
    :param matrices: list of squared matrices obtained for 'give_matrices'
    :return: list of truncated U tensors (last index reduced to max_dim) of length max_dim - 1
    """

    out = []

    for i in range(total_dimension - 1):

        M_1 = matrices[i]
        M_2 = matrices[i - 1 + total_dimension]
        dim = int(np.sqrt(len(M_1)))

        if len(M_1) <= max_dim:
            U, S = SVD_Data(M_1)
            out.append(U.reshape((dim, dim, dim**2)))

        else:
            U_1, S_1 = SVD_Data(M_1)
            U_2, S_2 = SVD_Data(M_2)

            S_1, S_2 = S_1[max_dim:, max_dim:], S_2[max_dim:, max_dim:]
            e_1, e_2 = np.trace(S_1), np.trace(S_2)

            if e_1 < e_2:
                U_1 = U_1[:, :max_dim]
                U_1 = U_1.reshape((dim, dim, max_dim))
                out.append(U_1)
            else:
                U_2 = U_2[:, :max_dim]
                U_2 = U_2.reshape((dim, dim, max_dim))
                out.append(U_2)

    return out


def generate_contraction_indices(total_dimension, index):
    """
    :param total_dimension: total dimension of the system
    :param index: index for which the contraction needs to be configured
    :return: array of  2 tuples of 2 arrays needed for the correct ncon calls, used to obtain the correct squared matrix
    """
    a1, a2, b1, b2 = np.zeros(2*total_dimension), np.zeros(2*total_dimension), np.zeros(2*total_dimension), np.zeros(2*total_dimension)

    a1[index] = -1
    a1[total_dimension] = -2
    a2[index] = -3
    a2[total_dimension] = -4

    b1[0] = -1
    b1[index] = -2
    b2[0] = -3
    b2[index] = -4

    for i in range(2*total_dimension):
        if a1[i] == 0.:
            a1[i] = int(i)
            a2[i] = int(i)

        if b2[i] == 0.:
            b1[i] = int(i)
            b2[i] = int(i)

    return [(a1, a2), (b1, b2)]

def generate_final_contraction_indices(total_dimension):

    # first contraction of the 2 (starting) tensors with simultaneous truncation of the first N indices

    T_1, T_2 = [-1], [total_dimension]
    U_array1 = []

    for i in range(1, total_dimension):
        T_1.append(i)
        T_2.append(total_dimension + i)
        U_array1.append([i, total_dimension + i, -(i + 1)])

    T_1.append(total_dimension)
    T_2.append(-1 * total_dimension - 1)

    for i in range(1, 2 * total_dimension - 1):
        if i % 2 == 1:
            T_1.append(-(total_dimension + i + 1))
        else:
            T_2.append(-(total_dimension + i + 1))

    U_array1.insert(0, T_2)
    U_array1.insert(0, T_1)

    # _________________________________________________________________________________________________________
    # start of the second contraction

    T = list(-1 * np.arange(1, total_dimension + 2))
    U_array2 = []

    for i in range(1, total_dimension):
        T.append(2 * i - 1)
        T.append(2 * i)
        U_array2.append([2 * i - 1, 2 * i, -(total_dimension + i + 1)])

    U_array2.insert(0, T)

    return [tuple(U_array1), tuple(U_array2)]

def rotate_indices(total_dimension, index):
    """
    :param total_dimension: rotates the matrix indices
    :param index:
    :return:indices rotated starting from the standard alphabet, places the 'index-th' letter on place 0 and the 'index + total dimension'-th letter on place total dimension
            (used to change the main contraction axis)
    """
    if index >= total_dimension:
        raise ValueError("Parameter 'index' has to be an integer between 0 and 'total_dimension', excluding 'total_dimension'.")

    a = alphabet[:total_dimension*2]

    if index == 0:
        return a

    t = int(index)
    d = int(index + total_dimension) % int(2*total_dimension)

    out = a[t] + a[1:min(t, d)] + a[0] + a[min(t, d) + 1: total_dimension] + a[d] + a[total_dimension + 1:max(t, d)] + a[total_dimension] + a[max(t, d) + 1:]
    return out

def HOTRG_step(total_dimension, tensor, mtensor):

    start = alphabet[: 2*total_dimension]

    for index in range(total_dimension):

        finish = rotate_indices(total_dimension, index)
        t = np.einsum(start + " -> " + finish, tensor)
        m = np.einsum(start + " -> " + finish, mtensor)
        matrices = give_matrices(t, total_dimension)
        U_list = truncation_tensors(total_dimension, matrices)
        contraction_indices = generate_final_contraction_indices(total_dimension)
        m = ncon(tuple([m] + [t] + U_list), contraction_indices[0])
        m = ncon(tuple([m] + U_list), contraction_indices[1])
        t = ncon(tuple([t] + [t] + U_list), contraction_indices[0])
        t = ncon(tuple([t] + U_list), contraction_indices[1])
        tensor = np.einsum(finish + " -> " + start, t)
        mtensor = np.einsum(finish + " -> " + start, m)

    return tensor, mtensor

def Calculate_Z(total_dimension, coupling_constant, steps):

    tensor = initial_tensor(total_dimension, coupling_constant)
    mtensor = initial_Mtensor(total_dimension, coupling_constant)
    c = 0

    for step in range(steps):

        tensor, mtensor = HOTRG_step(total_dimension, tensor, mtensor)
        c *= (2**total_dimension)
        check = abs(np.amax(tensor))

        if check > 0:
            tensor = tensor/check
            mtensor = mtensor/check
            c += np.log(check)

    a = alphabet[:total_dimension]

    Z = np.einsum(a + a, tensor)
    ZM = np.einsum(a + a, mtensor)

    return Z, c, ZM

def Calculate_m(total_dimension, coupling_constant, steps):

    Z, c, ZM = Calculate_Z(total_dimension, coupling_constant, steps)
    Z_2 = ZM/Z
    return Z_2




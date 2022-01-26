import numpy as np
from scipy.ndimage import zoom
from time import time

LABELS_SHAPE = [256, 256, 256]


def print_shapes(input, factors):
    in_shape = input.shape
    out_shape = np.multiply(input.shape, factors)
    print(f" Going from {in_shape} --> {out_shape}")

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'\tFunction {func.__name__!r}: {(t2-t1):.4f}s')
        return result
    return wrap_func


def generate_small_def(spacing):
    nonlin_shape_factor = [0.0625, 1/spacing, 0.0625]

    def_field = None
    factors = None

    return def_field, factors


def generate_small_bias(spacing):
    bias_shape_factor = [0.025, 1/spacing, 0.025]
    bias_field_std = 0.5

    small_bias_size = np.ceil(
        np.multiply(LABELS_SHAPE, bias_shape_factor)).astype(np.int)
    
    small_bias = bias_field_std * np.random.uniform(
        size=[1]) * np.random.normal(size=small_bias_size)
    factors = np.floor_divide(LABELS_SHAPE, small_bias_size)

    return small_bias, factors


@timer_func
def bias_scipy_zoom(small_bias, factors):
    bias_field = np.exp(zoom(small_bias, factors, order=1))
    return bias_field


@timer_func
def bias_jei_zoom(small_bias, factors):
    # return bias_field
    Sx, Sy, Sz = small_bias.shape
    Bx, By, Bz = np.multiply(small_bias.shape, factors)

    I = small_bias
    A = np.zeros((Bx, Sy, Sz))
    Xs = np.arange(Bx)
    Xs_prime = Xs / factors[0]
    f = np.floor(Xs_prime).astype(np.int)
    c = np.minimum(f + 1, Sx - 1)

    # w1 = c - Xs_prime
    w2 = Xs_prime - f
    w1 = 1 - w2

    for i in range(Bx):
        k = w1[i] * I[f[i], :, :] + w2[i] * I[c[i], :, :]
        A[i, :, :] = k

    I = A
    A = np.zeros((Bx, By, Sz))
    Ys = np.arange(By)
    Ys_prime = Ys / factors[1]
    f = np.floor(Ys_prime).astype(np.int)
    c = np.minimum(f + 1, Sy - 1)

    # w1 = c - Xs_prime
    w2 = Ys_prime - f
    w1 = 1 - w2

    for i in range(By):
        k = w1[i] * I[:, f[i], :] + w2[i] * I[:, c[i], :]
        A[:, i, :] = k

    I = A
    A = np.zeros((Bx, By, Bz))
    Zs = np.arange(Bz)
    Zs_prime = Zs / factors[2]
    f = np.floor(Zs_prime).astype(np.int)
    c = np.minimum(f + 1, Sz - 1)

    # w1 = c - Xs_prime
    w2 = Zs_prime - f
    w1 = 1 - w2

    for i in range(Bz):
        k = w1[i] * I[:, :, f[i]] + w2[i] * I[:, :, c[i]]
        A[:, :, i] = k

    return np.exp(A)


@timer_func
def bias_einsum_zoom(small_bias, factors):
    Bx, By, Bz = np.multiply(small_bias.shape, factors)

    I = small_bias
    letter = {0: 'i', 1: 'j', 2: 'k'}
    for idx, element in enumerate((Bx, By, Bz)):
        shape_list = list(I.shape)
        some_factor = shape_list[idx]
        Xs = np.arange(element)
        Xs_prime = Xs / factors[idx]
        f = np.floor(Xs_prime).astype(np.int)
        c = np.minimum(f + 1, some_factor - 1)
        w2 = Xs_prime - f
        w1 = 1 - w2

        summand1 = np.einsum(f'{letter[idx]}, ijk->ijk', w1, np.take(I, f, idx))
        summand2 = np.einsum(f'{letter[idx]}, ijk->ijk', w2, np.take(I, c, idx))
        I = summand1 + summand2
    
    return np.exp(I)


@timer_func
def einsum_zoom(small_bias, factors, flag='bias'):
    small_shape = small_bias.shape

    if len(small_shape) == 4:
        small_shape = small_shape[:-1]
        range_k = 4
    else:
        range_k = 3

    Bx, By, Bz = np.multiply(small_shape, factors)

    I = small_bias
    letter = {0: 'i', 1: 'j', 2: 'k'}
    for idx, element in enumerate((Bx, By, Bz)):
        some_factor = small_shape[idx]
        Xs = np.arange(element)
        Xs_prime = Xs / factors[idx]
        f = np.floor(Xs_prime).astype(np.int)
        c = np.minimum(f + 1, some_factor - 1)
        w2 = Xs_prime - f
        w1 = 1 - w2

        if flag == 'bias':
            ein_str = f'{letter[idx]}, ijk->ijk'
        else:
            ein_str = f'{letter[idx]}, ijkl->ijkl'
        
        summand1 = np.einsum(ein_str, w1, np.take(I, f, idx))
        summand2 = np.einsum(ein_str, w2, np.take(I, c, idx))
        I = summand1 + summand2
    
    if flag == 'def':
        return I
    else:
        return np.exp(I)


@timer_func
def bias_prod_zoom(small_bias, factors, flag='bias'):
    small_shape = small_bias.shape

    if len(small_shape) == 4:
        small_shape = small_shape[:-1]
        range_k = 4
    else:
        range_k = 3

    Bx, By, Bz = np.multiply(small_shape, factors)

    I = small_bias
    for idx, element in enumerate((Bx, By, Bz)):
        some_factor = small_shape[idx]
        Xs = np.arange(element)
        Xs_prime = Xs / factors[idx]
        f = np.floor(Xs_prime).astype(np.int)
        c = np.minimum(f + 1, some_factor - 1)
        w2 = Xs_prime - f
        w1 = 1 - w2

        my_list = list(range(range_k))
        my_list.remove(idx)
        summand1 = np.expand_dims(w1, my_list) * np.take(I, f, idx)
        summand2 = np.expand_dims(w2, my_list) * np.take(I, c, idx)
        I = summand1 + summand2
    
    if flag == 'def':
        return I
    else:
        return np.exp(I)

@timer_func
def def_prod_zoom(small_bias, factors):
    Bx, By, Bz = np.multiply(small_bias.shape[:-1], factors)

    I = small_bias
    for idx, element in enumerate((Bx, By, Bz)):
        shape_list = list(I.shape[:-1])
        some_factor = shape_list[idx]
        Xs = np.arange(element)
        Xs_prime = Xs / factors[idx]
        f = np.floor(Xs_prime).astype(np.int)
        c = np.minimum(f + 1, some_factor - 1)
        w2 = Xs_prime - f
        w1 = 1 - w2

        my_list = list(range(4))
        my_list.remove(idx)
        summand1 = np.expand_dims(w1, my_list) * np.take(I, f, idx)
        summand2 = np.expand_dims(w2, my_list) * np.take(I, c, idx)
        I = summand1 + summand2
    
    return I

@timer_func
def def_scipy_zoom():
    pass


@timer_func
def def_jei(small_def, factors):
    # return bias_field
    Sx, Sy, Sz, dims = small_def.shape
    # dims will always be 3... if not, maybe throw an error?
    Bx, By, Bz = np.multiply([Sx, Sy, Sz], factors)

    I = small_def
    A = np.zeros((Bx, Sy, Sz, dims))
    Xs = np.arange(Bx)
    Xs_prime = Xs / factors[0]
    f = np.floor(Xs_prime).astype(np.int)
    # c = np.ceil(Xs_prime).astype(np.int)
    c = np.minimum(f + 1, Sx - 1)

    # w1 = c - Xs_prime
    w2 = Xs_prime - f
    w1 = 1 - w2

    for i in range(Bx):
        k = w1[i] * I[f[i], :, :, :] + w2[i] * I[c[i], :, :, :]
        A[i, :, :, :] = k

    I = A
    A = np.zeros((Bx, By, Sz, 3))
    Ys = np.arange(By)
    Ys_prime = Ys / factors[1]
    f = np.floor(Ys_prime).astype(np.int)
    # c = np.ceil(Xs_prime).astype(np.int)
    c = np.minimum(f + 1, Sy - 1)

    # w1 = c - Xs_prime
    w2 = Ys_prime - f
    w1 = 1 - w2

    for i in range(By):
        k = w1[i] * I[:, f[i], :, :] + w2[i] * I[:, c[i], :, :]
        A[:, i, :, :] = k

    I = A
    A = np.zeros((Bx, By, Bz, 3))
    Zs = np.arange(Bz)
    Zs_prime = Zs / factors[2]
    f = np.floor(Zs_prime).astype(np.int)
    # c = np.ceil(Xs_prime).astype(np.int)
    c = np.minimum(f + 1, Sz - 1)

    # w1 = c - Xs_prime
    w2 = Zs_prime - f
    w1 = 1 - w2

    for i in range(Bz):
        k = w1[i] * I[:, :, f[i], :] + w2[i] * I[:, :, c[i], :]
        A[:, :, i, :] = k

    return A


if __name__ == '__main__':
    print(f'Labels Shape is: {LABELS_SHAPE}')
    for spacing in [2, 6, 8, 10, 12]:
        print(f'Spacing = {spacing}', end=' ')
        
        small_bias, bias_factors = generate_small_bias(spacing)
        print_shapes(small_bias, bias_factors)
        
        bias_field_scipy = bias_scipy_zoom(small_bias, bias_factors)
        bias_field_jei = bias_jei_zoom(small_bias, bias_factors)
        bias_field_einsum = bias_einsum_zoom(small_bias, bias_factors)
        bias_field_prod = bias_prod_zoom(small_bias, bias_factors)
        
        print('check sum of final matrix')
        print(f"1: {np.sum(bias_field_scipy)}")
        print(f"2: {np.sum(bias_field_jei)}")
        print(f"3: {np.sum(bias_field_einsum)}")
        print(f"4: {np.sum(bias_field_prod)}")
        
        print()

        small_def = np.stack([small_bias, small_bias, small_bias], axis=-1)
        def_field_jei = def_jei(small_def, bias_factors)
        def_field_prod = def_prod_zoom(small_def, bias_factors)
        def_field_prod1 = bias_prod_zoom(small_def, bias_factors, 'def')
        def_field_prod2 = einsum_zoom(small_def, bias_factors, 'def')
        
        print('check sum of final matrix')
        print(f"1: {np.sum(def_field_jei)}")
        print(f"2: {np.sum(def_field_prod)}")
        print(f"3: {np.sum(def_field_prod1)}")
        print(f"3: {np.sum(def_field_prod2)}")

        print()
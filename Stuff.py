
import numpy
import numpy as np
from numpy import convolve
import scipy
import math


def create_mask(n_max,n_out):
    n_max_calc = n_max+1
    lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
    mask = np.ones(lengh)
    for m1 in range(0, n_max+1):
        m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
        count=0
        for n1 in range(m1,n_max+1,2):
            if m1>n_out or n1>n_out:
                mask[m1_lengh+count] -= 1
            count+=1
    # mask = mask.bool()
    return mask

print(create_mask(31,2))
dackel
def Radial_function_matrix(m, n_max):
    scaling = []
    matrix = None
    matrix = []
    empty = np.zeros(n_max+1)
    for i in range(m):
        empty *=0
        empty[n_max-i] = 1
        ##print('hi',empty)
        matrix.append(empty.copy())
    ##print(matrix)
    for i in range(n_max+n_max+1):
        scaling.append(1/((2*n_max-i)**2+2))
    for n in range(m,n_max+1,2):
        faktor = []
        for i in range(int((n_max-n))):
            faktor.append(0)
        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k !=int((n-m)/2):
                faktor.append(0)

        for i in range(m):
            faktor.append(0)
        scale = convolve(faktor,faktor)
        scale = np.einsum('i,i', scaling,scale)

        #faktor = faktor/scale

        matrix.append((faktor.copy()))

        if n != n_max:
            empty *=0
            empty[n_max-n-1] = 1
            matrix.append(empty.copy())
    ##print(matrix)

    ##print(scaling)
    matrix = (np.rot90(numpy.vstack(np.array(matrix))))
    #matrix = np.where(matrix>0.0000000000001, matrix,0)
    print('done')
    print(matrix)
    return scipy.linalg.solve_triangular(matrix, np.identity(n_max+1))

print(Radial_function_matrix(0,30))

def Radial_function_matrix(m, n_max):
    scaling = []
    matrix = None
    matrix = []
    empty = np.zeros(n_max+1)
    for i in range(m):
        empty *=0
        empty[n_max-i] = 1
        ##print('hi',empty)
        matrix.append(empty.copy())
    ##print(matrix)
    for n in range(m,n_max+1,2):
        faktor = []
        for i in range(int((n_max-n))):
            faktor.append(0)
        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k !=int((n-m)/2):
                faktor.append(0)

        for i in range(m):
            faktor.append(0)
        matrix.append((faktor.copy()))

        if n != n_max:
            empty *=0
            empty[n_max-n-1] = 1
            matrix.append(empty.copy())
    ##print(matrix)

    ##print(scaling)
    matrix = (np.rot90(numpy.vstack(np.array(matrix))))
    ##print(matrix)
    return scipy.linalg.solve_triangular(matrix, np.identity(n_max+1))

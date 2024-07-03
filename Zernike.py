import numpy
import numpy as np
from numpy import convolve
import scipy
import math

#Fix sorting, Matrix and vactors are sortet in oposite ways
n_max = 25
##########################################################################################################################
# sin terms 1/pi
'''
def Radial_function(n,m, n_max):
    faktor = []
    scaling = []
    for i in range(n_max):
        scaling.append(1/((n_max-i-1)+2))
    #exp = []
    for i in range(n_max-n):
        faktor.append(0)

    for k in range(int((n-m)/2+1)):
        faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
        if k != int((n-m)/2):
            faktor.append(0)
        #exp.append(n-2*k)

    for i in range(m):
        faktor.append(0)

    scale = convolve(faktor,faktor)[-n_max:]
    ##print(scale)
    ##print(scaling)
    scale = np.einsum('i,i', scaling,scale)
    return np.array(faktor/scale)
def Radial_function_no_zero(n,m, n_max,m_min):
    faktor = []
    scaling = []
    for i in range(0,n_max,2):
        scaling.append(1/((n_max-i-1)+2))
    #exp = []

    for i in range(m_min,n_max-n,2):
        faktor.append(0)

    for k in range(int((n-m)/2+1)):
        faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )


    scale = convolve(faktor,faktor)[int((-n_max+m_min)/2-1):]
    ##print(scale)
    ##print(scaling)
    scale = np.einsum('i,i', scaling,scale)
    return np.flip(np.array(faktor/scale))
def Radial_function_matrix(m, n_max):
    scaling = []
    matrix = []
    for i in range(m,n_max+1,2):
        scaling.append(1/((n_max-i)+2))
    for n in range(m,n_max+1,2):
        faktor = []
        for i in range(int((n_max/2-n/2))):
            faktor.append(0)
        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
        scale = convolve(faktor,faktor)[int((-n_max+m)/2-1):]
        scale = np.einsum('i,i', scaling,scale)
        matrix.append((faktor/scale))
        ##print(scaling)
    return np.rot90(numpy.vstack(np.array(matrix)))
def Multiply(x,y,n_max):
    return convolve(x,y)[-n_max:]
n_max = 7
print(Radial_function(0,0,n_max))

#print(b)
c = Multiply(a,b,3)
#print(c)
c = np.array([12.,-8.])
Mat = scipy.linalg.solve_triangular(Radial_function_matrix(2,4), np.identity(2))
#print(Mat)
###print(scipy.linalg.solve_triangular(Radial_function_matrix(0,8), np.identity(5)))
#print(np.einsum('ij,j',Mat,c))
#out = convolve(Radial_function(5,1,n_max),Radial_function(1,1,n_max))[-n_max:]

##print(out)


#numpy.polynomial.polynomial.Polynomial



'''





















##########################################################################################################
#No scaling
###################################################################





# sin terms 1/pi
def Radial_function(n,m, n_max):
    faktor = []

    for i in range(n_max-n):
        faktor.append(0)

    for k in range(int((n-m)/2+1)):
        faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
        if k != int((n-m)/2):
            faktor.append(0)
        #exp.append(n-2*k)

    for i in range(m):
        faktor.append(0)
#x = math.factorial(8)

    return np.flip(np.array(faktor))
def Radial_function_no_zero(n,m, n_max,m_min):
    faktor = []

    for i in range(m_min,n_max-n,2):
        faktor.append(0)

    for k in range(int((n-m)/2+1)):
        faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )



    return np.flip(np.array(faktor))
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
def Multiply(x,y,n_max):
    # fix, work in no zero space
    x = np.flip(x)
    y = np.flip(y)
    return np.flip(convolve(x,y)[-n_max-1:])
#n_max = 7
##print(Radial_function(0,0,n_max))
##print(Radial_function_matrix(0,8))
#print(np.zeros(0))
def forward(m1,m2,n1,n2,n_max):
    In1 = Radial_function(n1,m1,n_max)
    In2 = Radial_function(n2,m2,n_max)
    Mult = Multiply(In1,In2,n_max)
    m_out1 = np.abs(m1-m2)
    m_out2 = np.abs(m1+m2)
    m_out2 = np.min((m_out2,n_max+1))
    Mat1 = Radial_function_matrix(m_out1,n_max)
    ##print(Mat1)
    if m_out2 > n_max:
        Mat2 = np.zeros((n_max+1,n_max+1))
    else:
        Mat2 = Radial_function_matrix(m_out2,n_max)
    lower=0
    higher = 0
    inbetween = 0
    #if not m_out1 == m_out2:
    for i in range(n_max+1,n_max+1-m_out1,-1):
        lower +=math.ceil(i/2)
    for i in range(1,n_max+1-m_out2):
        higher +=math.ceil(i/2)
    for i in range(n_max+1-m_out1-1,n_max+1-m_out2,-1):
        inbetween +=math.ceil(i/2)
    '''
    print('start')
    print(m_out1)
    print(m_out2)
    print(higher)
    print(lower)
    print(inbetween)
    '''
    #print(len(np.einsum('ij,j->i',Mat1,Mult)[m_out1:]))
    #print(len(np.einsum('ij,j->i',Mat2,Mult)[m_out2:]))
    out1 = np.einsum('ij,j->i',Mat1,Mult)[m_out1:]
    out2 = np.einsum('ij,j->i',Mat2,Mult)[m_out2:]
    #print(len(out1))
    #print(len(out2))
    out1 = out1[::2]
    out2 = out2[::2]
    #print(len(out1))
    #print(len(out2))
    out_dim_0 = np.zeros(lower,dtype=float)

    if not m_out1 == m_out2:
        out_dim_0 = np.append(out_dim_0,np.zeros(len(out1)))
    out_dim_0 = np.append(out_dim_0,np.zeros(inbetween))
    out_dim_0 = np.append(out_dim_0,out2)
    out_dim_0 = np.append(out_dim_0,np.zeros(higher))
    #print(len(out_dim_0))
    #dackel
    out_dim_1 = np.zeros(lower,dtype=float)
    if m1>m2:
        out_dim_1 = np.append(out_dim_1,out1)
    else:
        out_dim_1 = np.append(out_dim_1,np.zeros(len(out1)))
    out_dim_1 = np.append(out_dim_1,np.zeros(inbetween))

    if not m_out1 == m_out2:
        out_dim_1 = np.append(out_dim_1,np.zeros(len(out2)))
    out_dim_1 = np.append(out_dim_1,np.zeros(higher))

    out_dim_2 = np.zeros(lower,dtype=float)
    if m1==m2:
        out_dim_2 = np.append(out_dim_2,out1)
    else:
        out_dim_2 = np.append(out_dim_2,np.zeros(len(out1)))
    out_dim_2 = np.append(out_dim_2,np.zeros(inbetween))
    if not m_out1 == m_out2:
        out_dim_2 = np.append(out_dim_2,np.zeros(len(out2)))
    out_dim_2 = np.append(out_dim_2,np.zeros(higher))

    out_dim_3 = np.zeros(lower,dtype=float)
    if m1<m2:
        out_dim_3 = np.append(out_dim_3,out1)
    else:
        out_dim_3 = np.append(out_dim_3,np.zeros(len(out1)))
    out_dim_3 = np.append(out_dim_3,np.zeros(inbetween))
    if not m_out1 == m_out2:
        out_dim_3 = np.append(out_dim_3,np.zeros(len(out2)))
    out_dim_3 = np.append(out_dim_3,np.zeros(higher))
    #print(len(out))
    #dackel
    ##print(np.einsum('ij,j->i',Mat2,Mult))
    ##print(np.einsum('ij,j->i',Mat1,Mult))
    #,(np.einsum('ij,j->i',Mat1,Mult)),(np.zeros(higher)),(np.einsum('ij,j->i',Mat2,Mult))])
    #print(len(out))
    '''
    print(len(out_dim_0))
    print(len(out_dim_1))
    print(len(out_dim_2))
    print(len(out_dim_3))
    '''
    out = np.transpose(np.array([out_dim_0,out_dim_1,out_dim_2,out_dim_3]))
    return out
    #return (m_out1,np.einsum('ij,j',Mat1,Mult)), (m_out2,np.einsum('ij,j',Mat2,Mult))

#print(math.ceil(30/2))
#print(-(-30//2))
#print(Radial_function(7,1,14))
#print(Radial_function(7,1,14)[0:])
#print(Radial_function(7,1,14)[2:])
#print(Radial_function(7,7,14))
#print(Radial_function(1,1,14))
print(31*16/2)
#print(-(-30//2))
#print(-(-31//2))
#print(math.ceil(30/2))
#print(math.ceil(31/2))
#print(forward(2,1,3,1,31))
print(math.ceil(0/2))
forward(0,0,0,0,31)
forward(1,1,1,1,31)
forward(2,1,3,1,31)


n_max=30
def Zernicke_matrix_generator(n_max):
    n_max_calc = n_max+1
    lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
    grid = np.zeros((lengh,lengh,lengh,4))
    for m1 in range(0, n_max+1):
        print('done')
        m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
        #print(lengh)
        #print(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
        #print(m1_lengh)
        for m2 in range(0, n_max+1):
            m2_lengh = lengh - int(((n_max_calc-m2+1)*(n_max_calc-m2)/2)/2+math.ceil((n_max_calc-m2)/4))
            count1=0
            for n1 in range(m1,n_max+1,2):
                count2=0
                for n2 in range(m2,n_max+1,2):
                    #print('done')
                    x = forward(m1,m2,n1,n2,n_max)
                    #print(len(x))
                    grid[m1_lengh+count1,m2_lengh+count2,:,:] = x
                    count2 +=1
                count1 +=1
    return grid

def Fulltransform(n_max,in1,in2, Zernike_transform = None):
    Matrix_plus = [[[1/2,0],[0,-1/2]],[[0,1/2],[1/2,0]]]
    Matrix_minus_pos =[[[1/2,0],[0,1/2]],[[0,1/2],[-1/2,0]]]
    Matrix_minus_neg =[[[1/2,0],[0,1/2]],[[0,-1/2],[1/2,0]]]
    Matrix_minus_neut =[[[1/2,0],[0,1/2]],[[0,0],[0,0]]]
    transform = np.array([Matrix_plus,Matrix_minus_pos,Matrix_minus_neut,Matrix_minus_neg])
    if (Zernike_transform is None):
        matrix = Zernicke_matrix_generator(n_max)
    else:
        matrix = Zernike_transform
    out = np.einsum('im,ijkl,jn->klmn', in1,matrix,in2)
    #print(out[16:19])
    #print(out[31:33])
    out = np.einsum('lamn,klmn->ka', transform,out)
    return out
print(Radial_function(30,0,31))
def Radial_function(n,m, n_max):
    faktor = []
    scaling = []
    for i in range(n_max+n_max+1):
        scaling.append(1/((2*n_max-i)**2+2))

    for i in range(n_max-n):
        faktor.append(0)

    for k in range(int((n-m)/2+1)):
        faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
        if k != int((n-m)/2):
            faktor.append(0)
        #exp.append(n-2*k)

    for i in range(m):
        faktor.append(0)
    scale = convolve(faktor,faktor)
    scale = np.einsum('i,i', scaling,scale)

    faktor = np.array(faktor/scale)
    faktor = np.array(faktor)
    return np.flip(faktor)


#print(forward(0,0,16,16,17))
print(Radial_function(30,0,31))
donkey
one = np.zeros((256,2))
two = np.zeros((256,2))
matrix= Zernicke_matrix_generator(30)
one[16,0]=1
two[31,0]=1
print(Fulltransform(30,one,two,matrix)[0:50])
one *=0
two *=0
one[16,1]=1
two[31,1]=1
print(Fulltransform(30,one,two,matrix)[0:50])
one *=0
two *=0
one[16,0]=1
two[31,1]=1
print(Fulltransform(30,one,two,matrix)[0:50])
one *=0
two *=0
one[16,1]=1
two[31,0]=1
print(Fulltransform(30,one,two,matrix)[0:50])
dackel








matrix = Zernicke_matrix_generator(30)
print(np.shape(matrix))
one = np.zeros(256)
two = np.zeros(256)
one[1]=1
two[16]=1
print('start')
print(np.einsum('ijkl,i,j->kl',matrix,one,two)[0:20])
one = np.zeros(256)
two = np.zeros(256)
one[1]=1
two[2]=1
print('start')
print(np.einsum('ijkl,i,j->kl',matrix,one,two)[0:20])
#print(np.shape(grid))

one = np.zeros(256)
two = np.zeros(256)
one[16]=1
two[17]=1
print('start')
print(np.einsum('ijkl,i,j->kl',matrix,one,two)[0:40])

one = np.zeros(256)
two = np.zeros(256)
one[16]=1
two[2]=1
print('start')
print(np.einsum('ijkl,i,j->kl',matrix,one,two)[0:20])
print(np.shape(matrix))
def signtransform(m1,m2,vec1,vec2):
    # 2D Vector encoding the sign of the Polinome. (1,0) corresponds to sign = +, aka cos(mx), while (0,1) corresponds to sin(mx) or sign = -.
    # The transformation of of polimnome order gets slightly altered due to the sign of m1-m2
    # In addition to the sign of the polinome coefficient m, the sign of the vector values might also be altered by the transformation ( sin(nx)*sin(mx)=cos((n-m)x)-cos((n+m)x)     )
    s = np.sign(np.sign(m1-m2)+1) # 1 if m1 > m2 or m1 = m2. 0 if m1<m2

    # The case m1 = m2 gets ignored, as the term in question is sin((m1-m2)x), which is zero for m1=m2
    # build in this way, so the matrix may be put into an init method, and then multiplied by [s,1-s]

    Mat_n_plus_m =[[[1/2,0],[0,-1/2]],[[0,1/2],[1/2,0]]]
    Mat_n_minus_m_pos =[[[1/2,0],[0,1/2]],[[0,1/2],[-1/2,0]]]
    Mat_n_minus_m_neg =[[[1/2,0],[0,1/2]],[[0,-1/2],[1/2,0]]]
    outplus = np.einsum('i,ijk,j->k',vec1,Mat_n_plus_m,vec2)
    outminus = s* np.einsum('i,ijk,j->k',vec1,Mat_n_minus_m_pos,vec2) + (1-s) * np.einsum('i,ijk,j->k',vec1,Mat_n_minus_m_pos,vec2)
    return (outplus,outminus)
def signtransform(m1,m2,vec1,vec2):
    # 2D Vector encoding the sign of the Polinome. (1,0) corresponds to sign = +, aka cos(mx), while (0,1) corresponds to sin(mx) or sign = -.
    # The transformation of of polimnome order gets slightly altered due to the sign of m1-m2
    # In addition to the sign of the polinome coefficient m, the sign of the vector values might also be altered by the transformation ( sin(nx)*sin(mx)=cos((n-m)x)-cos((n+m)x)     )
    s = np.sign(np.sign(m1-m2)+1) # 1 if m1 > m2 or m1 = m2. 0 if m1<m2

    # The case m1 = m2 gets ignored, as the term in question is sin((m1-m2)x), which is zero for m1=m2
    # build in this way, so the matrix may be put into an init method, and then multiplied by [s,1-s]

    Mat_n_plus_m =[[[1/2,0],[0,-1/2]],[[0,1/2],[1/2,0]]]
    Mat_n_minus_m =[[[1/2,0],[0,1/2]],[[0,s*1/2],[-1/2*s,0]]]
    outplus = np.einsum('i,ijk,j->k',vec1,Mat_n_plus_m,vec2)
    outminus =  np.einsum('i,ijk,j->k',vec1,Mat_n_minus_m,vec2)
    return (outplus,outminus)

#print(signtransform(2,1,[1,0],[1,0]))

# Implement n**3 Matrix
# Copy it 2*2*8 times
#perform product independantly for each combination of 2 and 2
# upper triangle goes to 0-3, lower one to 4-7
# reduce back down via sign transform function
# Product is of dimension n x n -> n. Therefore, we have n**2 independant weights. If we eant the model to be invariant towards permutation of input1 and input2, we have (n**2)/2 +n/2 independant features, yet have to properly implement this to autograd. Luckily, this is for the moment not necessary.
# (n x 2) * (n x 2) -> (n x 8)
# (n x 8) * (8 x 2) -> (n x 2)

#Matrix_lower =

#### copy both inputs to have (n x 4) inputs, with one being ordered 1,1,2,2 and the other one being ordered 1,2,1,2
# Do (4 x n)* n**3 x 3 x 2 * (n x 4) product -> n x 4 x 3 x 2
# 2 being n+m or n-m and 3 being sgn(n-m). This is reducible to dim 4 as m+n term is independant of sgn(n-m).
# Multiply with a 4 x 3 x 2 x 2 matrix to collapse it down to feature space
# Optionally multiply it with  4 x 3 x 2 x 4 matrix instead to keep it in computation space


# For multiplicities of the same input Have the n**3 matrix be n**3 x m x m to allow free propagation between Nodes
# Paramater space size = num_layers* n**2  * m **2
# Base storage usage ~n**3
'''

n_max = 5

d = Radial_function(3,1,n_max)
e = Radial_function(1,1,n_max)
#print('In 1',d)
#print('in 2 ',e)
c = Multiply(e,d,n_max)
#print('Multiplied',c)
#c = np.array([-2.,3.])
Mat = Radial_function_matrix(2,n_max)
##print(Radial_function_matrix(2,5))
##print(Radial_function_matrix(2,4))
#print('Matrix',Mat)
###print(scipy.linalg.solve_triangular(Radial_function_matrix(0,8), np.identity(5)))
#print('Output',np.einsum('ij,j',Mat,(c)))
#out = convolve(Radial_function(5,1,n_max),Radial_function(1,1,n_max))[-n_max:]
'''
##print(out)
import math

import numpy
import numpy as np
import scipy
import torch
import torch.nn as nn
from numpy import convolve


class  Non_linearity(nn.Module):
    '''
    A Non-linearity defined for Zernike objects. The Non-linearity needs to act on the absolute value of sqrt((Z_m^n)**2 + (Z_-m^n)**2), in order to not break equivariance.
    This function replicates the functionallity of ReLU, with a learnable cutoff. It takes the abs value, applies a 1D linear layer to it, and applies a differentiable step-function to it.
    This is multiplied to the input. The step function is used so out is proportional to in, not to in**2, which leads to NaNs appearing.
    '''
    def __init__(self, normalize= False):
        super().__init__()
        self.lin = nn.Linear(1,1)
        self.normalize = normalize
    def forward(self,x):
        square = torch.sum(x**2, dim=-1,keepdim=True)
        a = self.lin(square)
        a = (torch.tanh(a)+1)/2

        if self.normalize:
            norm = torch.sum(torch.sqrt(square),dim=-2,keepdim=True)
            return a*x/norm
        else:
            return a * x
class Lintrans3(nn.Module):
    def __init__(self, inp,out,Non_lin=False):
        super().__init__()
        self.lin = nn.Linear(inp,out)
        self.Non_lin_bool = Non_lin
        if Non_lin:
            self.Non_lin = Non_linearity()
    def forward(self,x):
        x = torch.transpose((self.lin(torch.transpose(x,-1,-3))),-1,-3)
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x
class Multilin(nn.Module):
    '''
    A class that creates a linear layer with independant weights along it's channels.
    In this framework, this serves to create a linear layer convolving the channels with each other, but doing this in a way that is independent for different m,n.
    Yet, it acts identical on n,m and n,-m; therefore keeping rotation equivariance
    '''
    def __init__(self, inp,out,size,Non_lin=False):
        super().__init__()
        #self.device = 'cuda:2'
        #self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,inp,out, device = self.device)))#/np.sqrt(inp))
        self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,inp,out)))#/np.sqrt(inp))
        self.Non_lin_bool = Non_lin
        if Non_lin:
            self.Non_lin = Non_linearity()
    def forward(self,x):
        x = torch.einsum('ijk,...jil->...kil',self.weight_lin,x)
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x



class Zernike_layer(nn.Module):
    def __init__(self, n_max = 30,n_max_2=None, n_out=30, multichanneled = False,in_channels = 1 ,intermediate_channels=1, out_channels =1 ,last_layer = False, fast_test_dimensionality = False,normalize=False):
        super().__init__()
        #self.device = device
        '''
        The core Zernike layer. The final forward method is essentially yust simple matrix multiplication, most of the things happen in preprocessing.

        The arguments n_max and n_max_2 are the orders of the Zernike Vectors input into the Network. If n_max_2 is not set, it is set to n_max_2 = n_max.
        n_out is the desired Order of the output. All inputs get embedded into the space of the highest of the three n, and the product performed in that space.
        Finally, if n_out is smaller than this n, only the terms corresponding to n_out get output.

        The argument "fast_test_dimensionality" skips that processing to test wether everything else works. This is due to the processing taking rather long, which is annoying for bugfixing.


        '''

        self.increase_in1 = False
        if n_max_2 is not None:
            if n_max_2 < n_max:
                print('For inputs of different Orders, please use in2 as the input of higher order')
                Donkey
        if fast_test_dimensionality:
            print('Warning, this layer is completely unfunctional, yet will load faster, so you can test wether all dimensions of your Tensors add up')
        if n_max < n_out:
            #batch_size=1
            Zernike_normalization = Zernike_Norms(n_out)
            self.Zernike_normalization = Zernike_normalization()
            self.increase_in1 = True
            self.in_mask_1 = torch.nn.parameter.Parameter(self.create_mask_increase(n_max,n_out),requires_grad=False)
            self.zero_mask = torch.nn.parameter.Parameter(self.create_zero_mask(n_out),requires_grad=False)
            out_size= self.calc_size(n_out)
            if fast_test_dimensionality:
                self.Zernike_matrix = torch.nn.parameter.Parameter(torch.zeros(out_size,out_size,out_size,4),requires_grad=False)
            else:
                self.Zernike_matrix = torch.nn.parameter.Parameter(torch.tensor(self.Zernicke_matrix_generator(n_out),dtype=torch.float),requires_grad=False)
            size = self.calc_size(n_out)

        self.increase_in2 = False
        if n_max_2 is None:
            if self.increase_in1 == True:
                self.increase_in2 = True
                self.in_mask_2 = self.in_mask_1
        elif n_max_2 < n_out:
            if self.increase_in1 == False:
                Zernike_normalization = Zernike_Norms(n_out)
                self.Zernike_normalization = Zernike_normalization()
                out_size= self.calc_size(n_out)
                if fast_test_dimensionality:
                    self.Zernike_matrix = torch.nn.parameter.Parameter(torch.zeros(out_size,out_size,out_size,4),requires_grad=False)
                else:
                    self.Zernike_matrix = torch.nn.parameter.Parameter(torch.tensor(self.Zernicke_matrix_generator(n_out),dtype=torch.float),requires_grad=False)
                size = self.calc_size(n_out)
            self.increase_in2 = True
            self.in_mask_2 = torch.nn.parameter.Parameter(self.create_mask_increase(n_max_2,n_out),requires_grad=False)
            self.zero_mask = torch.nn.parameter.Parameter(self.create_zero_mask(n_out),requires_grad=False)
        if self.increase_in1 == False and self.increase_in2 == False:
            size = self.calc_size(n_max)
            Zernike_normalization = Zernike_Norms(n_max)
            self.Zernike_normalization = Zernike_normalization()
            self.zero_mask = torch.nn.parameter.Parameter(self.create_zero_mask(n_max),requires_grad=False)
            if fast_test_dimensionality:
                self.Zernike_matrix = torch.nn.parameter.Parameter(torch.zeros(size,size,size,4),requires_grad=False)
            else:
                self.Zernike_matrix = torch.nn.parameter.Parameter(torch.tensor(self.Zernicke_matrix_generator(n_max),dtype=torch.float),requires_grad=False)





        Matrix_plus = [[[1/2,0],[0,-1/2]],[[0,1/2],[1/2,0]]]
        Matrix_minus_pos =[[[1/2,0],[0,1/2]],[[0,-1/2],[1/2,0]]]
        Matrix_minus_neut =[[[1/2,0],[0,1/2]],[[0,0],[0,0]]]
        Matrix_minus_neg =[[[1/2,0],[0,1/2]],[[0,1/2],[-1/2,0]]]
        self.transform = torch.nn.parameter.Parameter(torch.tensor(np.array([Matrix_plus,Matrix_minus_pos,Matrix_minus_neut,Matrix_minus_neg]),dtype=torch.float),requires_grad=False)
        if normalize:
            self.Nonlin = Non_linearity(normalize=True)
        else:
            self.Nonlin = Non_linearity(normalize=False)
        self.weight = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,size)))#/size)
        self.reduce = False
        if n_max > n_out:
            self.reduce = True
            self.out_mask = torch.nn.parameter.Parameter(self.create_mask_decrease(n_max,n_out),requires_grad=False)
        self.last_layer = last_layer
        self.multichanneled = False
        if multichanneled != False:
            self.multichanneled = True
        if multichanneled == 'same':
            # Possibly make this independant for each m,n. This would lead to a more pronounced role of multiple channels, yet drastically increase complexitly.
            self.In_Lin1 = Lintrans3(in_channels,intermediate_channels)
            self.In_Lin2 = Lintrans3(in_channels,intermediate_channels)
            self.Out_Lin  = Lintrans3(intermediate_channels,out_channels)
        if multichanneled == 'independant':
            # Possibly make this independant for each m,n. This would lead to a more pronounced role of multiple channels, yet drastically increase complexitly.
            self.In_Lin1 = Multilin(in_channels,intermediate_channels,size,Non_lin=True)
            self.In_Lin2 = Multilin(in_channels,intermediate_channels,size,Non_lin=True)
            self.Out_Lin  = Multilin(intermediate_channels,out_channels,size)
    def calc_size(self,n_max):
        '''
        Calculating the amount of terms in the Zernike decomposition depending on n. This calculates the amount of radial polinomes, the final decomposition will have size= (calc_size(n),2)
        '''
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh
    def create_mask_decrease(self,n_max,n_out):
        '''
        Creating a mask that is able to convert input of order n_max to output of order n_out, with n_max>n_out

        '''
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        mask = torch.ones(lengh)
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                if m1>n_out or n1>n_out:
                    mask[m1_lengh+count] -= 1
                count+=1
        mask = mask.bool()
        return mask
    def create_mask_increase(self,n_max,n_out):
        '''
        Creating a mask that is able to convert input of order n_max to output of order n_out, with n_max<n_out

        '''
        n_out_calc = n_out+1
        n_max_calc = n_max+1
        lengh = int(((n_out_calc+1)*n_out_calc/2)/2+math.ceil(n_out_calc/4))
        lengh_in = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        mask = torch.zeros(lengh,lengh_in)
        for m1 in range(0, n_out+1):
            m1_lengh = lengh - int(((n_out_calc-m1+1)*(n_out_calc-m1)/2)/2+math.ceil((n_out_calc-m1)/4))
            count=0
            for n1 in range(m1,n_out+1,2):
                m_in_lengh = lengh_in - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
                if not (m1>n_max or n1>n_max):
                    mask[m1_lengh+count, m_in_lengh+count] += 1
                count+=1
        #mask = mask.bool()
        return mask
    def Radial_function(self,n,m, n_max):
        '''
        Creating the representation of the radial functions in terms of polynomials
        '''
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

        norm = self.Zernike_normalization[n][m]
        faktor = np.array(faktor)
        faktor = faktor/norm
        return np.flip(faktor)




    def Radial_function_matrix(self,m, n_max):
        '''
        Creating a Matrix that transforms any valid polinomial to it's represantation in terms of Radial polinomials of given Order m.
        Valid polynomials a_i r**i are gives by a_i = 0 for all i<m and a_i = 0 for all i-m not even.

        This is done by creating a matrix relating all polinomials in question to their representation in terms of powers of r up to i=n_max.
        This matrix is then numerically inverted to have it point from the space of powers of r to the space of Radial polinomials.
        All powers of r which are zero by the rules given above are filled with a one on the diagonal. This is done so the matrix is still invertable, yet has no impact on the final output.

        '''
        matrix = None
        matrix = []
        empty = np.zeros(n_max+1)
        for i in range(m):
            empty *=0
            empty[n_max-i] = 1
            matrix.append(empty.copy())
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
            norm = self.Zernike_normalization[n][m]
            faktor = np.array(faktor)/norm

            matrix.append((faktor.copy()))

            if n != n_max:
                empty *=0
                empty[n_max-n-1] = 1
                matrix.append(empty.copy())
        matrix = (np.rot90(numpy.vstack(np.array(matrix))))

        return scipy.linalg.solve_triangular(matrix, np.identity(n_max+1))
    def Multiply(self,x,y,n_max):
        '''
        Multiplying two polinmoials in terms of powers of r may be done by np.convolve. Yet, our polynomials are ordered in a way that is inverted in regards to how convolve expects them.
        '''
        x = np.flip(x)
        y = np.flip(y)
        return np.flip(convolve(x,y)[-n_max-1:])
    def Calculate_matrix_coefficients(self,m1,m2,n1,n2,n_max):
        '''
        Creating one line of the final Matrix. It calculates the output of (n1,m1) and (n2,m2) being multiplied to each other. For this,
        1) The polynomials are encoded to their respective representation in powers of r
        2) Both get multiplied to each other
        3) The respective proper matrices to convert the terms in powers of r to the Radial polynomials of the proper m are created; and multiplied to the output of step 2)
        4) They get placed in the proper place in the global matrix. For the m1-m2 terms, we seperate the terms with m1 > m2, m1 = m2 and m1 < m2, as they lead to different final outputs.


        '''
        In1 = self.Radial_function(n1,m1,n_max)
        In2 = self.Radial_function(n2,m2,n_max)
        Mult = self.Multiply(In1,In2,n_max)
        m_out1 = np.abs(m1-m2)
        m_out2 = np.abs(m1+m2)
        m_out2 = np.min((m_out2,n_max+1))
        Mat1 = self.Radial_function_matrix(m_out1,n_max)
        ##print(Mat1)
        if m_out2 > n_max:
            Mat2 = np.zeros((n_max+1,n_max+1))
        else:
            Mat2 = self.Radial_function_matrix(m_out2,n_max)
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
        out1 = np.einsum('ij,j->i',Mat1,Mult)[m_out1:]
        out2 = np.einsum('ij,j->i',Mat2,Mult)[m_out2:]
        out1 = out1[::2]
        out2 = out2[::2]
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
        out = np.transpose(np.array([out_dim_0,out_dim_1,out_dim_2,out_dim_3]))
        return out

    def Zernicke_matrix_generator(self,n_max):
        '''
        Iterating over all pairs of inputs to generate the full conversion matrix

        '''
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        grid = np.zeros((lengh,lengh,lengh,4))
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            for m2 in range(0, n_max+1):
                m2_lengh = lengh - int(((n_max_calc-m2+1)*(n_max_calc-m2)/2)/2+math.ceil((n_max_calc-m2)/4))
                count1=0
                for n1 in range(m1,n_max+1,2):
                    count2=0
                    for n2 in range(m2,n_max+1,2):
                        #print('done')
                        x = self.Calculate_matrix_coefficients(m1,m2,n1,n2,n_max)
                        #print(len(x))
                        grid[m1_lengh+count1,m2_lengh+count2,:,:] = x
                        count2 +=1
                    count1 +=1
        return grid
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)

    def create_zero_mask(self,n_max):
        '''
        The Zernike Product will per definition output 0 for all Terms with m=0 and a sin function. In fact, these terms don't even exist for the decomposition, as they are proportonal to sin(mx)=sin(0*x).
        While the output of the product will set the to zero, the linear layers may not do so, if there is a bias involved. At the moment, there is none, yet it seems wise to make completely sure nothing of the kind is happening to preserve equivariance.
        '''
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        mask = torch.ones(lengh,2)
        for i in range(0,(n_max//2)+1):
            mask[i,1] = 0
        return mask



    def forward(self,in1,in2):
        if self.increase_in1:
            '''
            Increasing size of input 1 in case it needs to
            '''
            in1 = torch.einsum('ij,...jk->...ik',self.in_mask_1,in1)
        if self.increase_in2:
            '''
            Increasing size of input 2 in case it needs to
            '''
            in2 = torch.einsum('ij,...jk->...ik',self.in_mask_2,in2)
        if self.multichanneled:
            '''
            If there are multiple channels, we first apply a linear layer along the channel dimension. The nonlinearity gets called within the linear layer
            '''
            in1 = self.In_Lin1(in1)
            in2 = self.In_Lin2(in2)
        '''
        We explicitely set all terms of m =-0 to zero
        '''
        in1 = torch.einsum('ij,...ij->...ij',self.zero_mask,in1)
        in2 = torch.einsum('ij,...ij->...ij',self.zero_mask,in2)
        '''
        We multiply our inputs to the Zernike matrix. A weight Matrix is added that is able to learn what interactions the model is supposed to favor
        '''
        out = torch.einsum('...im,ijkl,ij,...jn->...klmn', in1,self.Zernike_matrix,self.weight,in2)
        # Do not put anything inbetween, as it might break equivariance
        '''
        We collapse the four different channels that correspond to the different cases considered above to their representation in +- m.
        '''
        out = torch.einsum('lamn,...klmn->...ka', self.transform,out)
        #print(out)
        #print('hi2')
        if not self.last_layer:
            '''
            Afterwards, we call a Non-linearity
            '''
            out = self.Nonlin(out)
        if self.multichanneled and not self.last_layer:
            '''
            Finally, one more linear layer along channel dimension
            '''
            out = self.Out_Lin(out)
        if self.reduce:
            '''
            If the output size is smaller then one of the input sizes, we need to consider only the output terms of interest
            '''
            out = out[:,:,self.out_mask,:]
        return out


'''

           __n__n__
    .------`-\00/-'
   /  ##  ## (oo)
  / \## __   ./
     |//YY \|/
     |||   |||

     Renata


A cow (her name is Renata) to improve Code beauty for this terribly coded class


'''
class Zernike_Norms(nn.Module):
    '''
    As the normalization of the polinomials depends on the image size, the norm is calculated numerically here.
    Normalizing all filters to the same value is useful, as it allows us to recrate images by simply adding the filters*coefficients up.
    Not having them normalized leads to an overrepresentation of filters of higher norm.

    '''
    def __init__(self, n_max = 30 ):
        super().__init__()
        self.norm_output = self.calc_norms(n_max)



    def Radial_function(self,n,m, n_max):
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
        faktor = np.array(faktor)
        return np.flip(faktor)

    def Zernicke_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros((lengh,n_max+1))
        #Basis = []
        Basis = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        #print(np.shape(Basis))
        for m1 in range(0, n_max+1):
            #m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                #print(n1,m1)
                Basis[n1][m1] = self.Radial_function(n1,m1,n_max)
                count+=1
        return Basis
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)


    def calc_norms(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        #print(Zernike_functions)
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                if Zernike_functions[i][j] is None:
                    functions[i][j] = numpy.polynomial.polynomial.Polynomial([0])
                    #print('None')
                else:
                    functions[i][j] = (numpy.polynomial.polynomial.Polynomial(Zernike_functions[i][j]))
        out = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                out[i][j] = torch.tensor(np.array(functions[i][j](np.sqrt((x ** 2 + z ** 2)))*np.cos(j*np.arctan2(x , (z )))),dtype=torch.float)#,functions[i][j](np.sqrt((x ** 2 + z ** 2)))*np.sin(j*np.arctan2(x ,(z  )))])*self.mask(x,z),dtype=torch.float)

        norm = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                norm[i][j] = (torch.sum(torch.abs(out[i][j]),dim= (-1,-2),keepdim = False)).item()

        return norm
    def forward(self):
        return self.norm_output

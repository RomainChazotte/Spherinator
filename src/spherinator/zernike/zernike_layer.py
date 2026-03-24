import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import NonLinearity, MultiLinear
from .zernike_norm import ZernikeNorms
import os
import numpy as np
import scipy
from skimage.measure import block_reduce
import warnings

class ZernikeLayer(nn.Module):
    '''
    Core Zernike-based layer for structured interactions between input features.

    This layer computes the product of two Zernike-embedded tensors (e.g., representing radial/angular
    structure) and applies optional channel mixing, non-linearities, and masking to ensure rotational equivariance.

    Supports input of different Zernike orders (n_max, n_max_2), projection to a target output order (n_out),
    and optional flip-invariance, gating, and multichannel support.

    Args:
            - n_max (int): maximum Zernike order of input 1. Default: 30
            - n_max_2 (int): optional; Zernike order of input 2 (defaults to n_max)
            - n_out (int): desired output Zernike order. Default: 30
            - multichanneled (bool or str): use linear mixing across channels ('same', 'independant', or False)
            - in_channels (int): number of input channels
            - intermediate_channels (int): internal channel dimension
            - out_channels (int): number of output channels
            - image_size (int): size of the input image (square assumed)
            - fast_test_dimensionality (bool): if True, skips matrix loading to speed up debugging
            - normalize (bool): placeholder for future normalization logic
            - device (str): device on which the model will be run; e.g. 'cuda', 'cpu', or 'mps' 
            (for macOS with Metal programming framework). Default: 'cuda'
    '''
    def __init__(self, n_max=30, n_max_2=None, n_out=30,  multichanneled=False, in_channels=1, intermediate_channels=1, out_channels=1, image_size=14, fast_test_dimensionality=False, normalize=False,invert=False, vast_weights=False,Skip_connect=False, device='cuda'):
        super().__init__()
        if invert:
            __ = n_max
            n_max = n_out
            n_out = __

            __ = in_channels
            in_channels = out_channels
            out_channels = __
            intermediate_channels = __


        '''
        The core Zernike layer. The final forward method is essentially yust simple matrix multiplication, most of the things happen in preprocessing.

        The arguments n_max and n_max_2 are the orders of the Zernike Vectors input into the Network. If n_max_2 is not set, it is set to n_max_2 = n_max.
        n_out is the desired Order of the output. All inputs get embedded into the space of the highest of the three n, and the product performed in that space.
        Finally, if n_out is smaller than this n, only the terms corresponding to n_out get output.

        The argument "fast_test_dimensionality" skips that processing to test wether everything else works. This is due to the processing taking rather long, which is annoying for bugfixing.
        '''
        
        self.device = device
        self.flag_multichanneled = False  
        self.increase_in1 = False  # Whether input 1 needs to be upsampled
        self.increase_in2 = False  # Whether input 2 needs to be upsampled
        self.reduce = False        # Whether the output will be down-projected
        eps = torch.finfo(torch.float32).eps 
        

        if fast_test_dimensionality:
            warnings.warn('Warning, this layer is completely unfunctional, yet will load faster, so you can test wether all dimensions of your Tensors add up')
        # ---

        # --- Upsample input 1 if needed ---
        if n_max < n_out:
            self. Zernike_normalization = ZernikeNorms(n_out, image_size=image_size)()

            self.increase_in1 = True
            self.in_mask_1 = self._create_mask_increase(n_max, n_out)
            self.zero_mask = self._create_zero_mask(n_out)
            
            size = self._calc_size(n_out)
            if fast_test_dimensionality:
                self.Zernike_matrix = torch.zeros(size,size,size, 4, device = self.device)
            else:
            	path = '.preprocessed/Zernike_layer_matrix_{}_{}'.format(image_size,n_out)
            	self.Zernike_matrix = self._load_or_create_zernike_matrix(n_out, image_size, path)


        # --- Upsample input 2 if needed ---
        if n_max_2 is not None:
            if n_max_2 < n_max:
                print('For inputs of different Orders, please use in2 as the input of higher order')
        
        if n_max_2 is None:
            if self.increase_in1:
                self.increase_in2 = True
                self.in_mask_2 = self.in_mask_1
        elif n_max_2 < n_out:
            if not self.increase_in1:
                self.Zernike_normalization = ZernikeNorms(n_out, image_size=image_size)()
                size = self._calc_size(n_out)
                if fast_test_dimensionality:
                    self.Zernike_matrix = torch.zeros(size, size, size, 4, device=self.device)
                else:
                	path = '.preprocessed/Zernike_layer_matrix_{}_{}'.format(image_size,n_out)
                	self.Zernike_matrix = self._load_or_create_zernike_matrix(n_out, image_size, path)
            self.increase_in2 = True
            self.in_mask_2 = self._create_mask_increase(n_max_2, n_out)
            self.zero_mask = self._create_zero_mask(n_out)
        # --- No upsampling needed ---
        if not self.increase_in1 and not self.increase_in2:
            size = self._calc_size(n_max)
            self.Zernike_normalization = ZernikeNorms(n_max, image_size=image_size)()
            self.zero_mask = self._create_zero_mask(n_max)

            if fast_test_dimensionality:
                self.Zernike_matrix = torch.zeros(size,size,size,4, device = self.device)
            else:
                path = '.preprocessed/Zernike_layer_matrix_{}_{}'.format(image_size,n_max)
                self.Zernike_matrix = self._load_or_create_zernike_matrix(n_max, image_size,path)
        # --- Collapse matrix for combining output channels ---
        self.transform = self._init_transform_tensor()

        # --- Non-linearity ---
        self.Nonlin = NonLinearity(self._calc_size(n_out))

        # --- Learnable interaction weights ---
        stdv= 1./math.sqrt(size)
        # self.weight = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size, size)))
        # self.weight = torch.nn.parameter.Parameter(torch.nn.init.uniform_(torch.empty(size,size),a=-stdv,b= stdv))

        # --- Downsample output if n_out < max(n_max, n_max_2) ---
        if n_max > n_out:
            self.reduce = True
            self.out_mask = self._create_mask_decrease(n_max, n_out)
        
        # --- Multichannel configuration ---
        if multichanneled != False:
            self.flag_multichanneled = True


        if multichanneled == 'independant':
            self.In_Lin1 = MultiLinear(in_channels, intermediate_channels, size, non_lin=True)
            self.In_Lin2 = MultiLinear(in_channels, intermediate_channels, size, non_lin=True)
            size = self._calc_size(n_out)
            self.Out_Lin  = MultiLinear(intermediate_channels, out_channels, size)



        # --- Final normalization of Zernike matrix ---
        n_maximal = max(n_max, n_max_2) if n_max_2 is not None else n_max
        self.Zernike_matrix = self.Zernike_matrix / (
        							torch.abs( torch.sum ( self.Zernike_matrix, dim=(-1,-2), keepdim=True ) ) + eps )
        self.Zernike_matrix = torch.where( torch.abs ( self.Zernike_matrix ) < 1e-5, 0, self.Zernike_matrix)    
        if n_out < n_maximal:
            mask = self._create_mask_decrease(n_maximal,n_out)
            self.Zernike_matrix = self.Zernike_matrix[:,:,mask,:]

        self.normalized_zernike_matrix = torch.where( torch.abs ( self.Zernike_matrix ) < 1e-5, 0, 1.).to('cpu')   
        size = self._calc_size(max(n_max,n_out))
        stdv= 1./math.sqrt(size)
        if not vast_weights:
            self.weight = torch.nn.parameter.Parameter(torch.nn.init.uniform_(torch.empty(size,size),a=-stdv,b= stdv))
        elif vast_weights:
            self.weight = torch.nn.init.uniform_(torch.empty(size,size),a=-stdv,b= stdv)#.to('cuda')
            self.weight =  torch.nn.parameter.Parameter(torch.einsum('ij,ijkl->ijkl',self.weight,self.normalized_zernike_matrix))#.to('cuda')
            self.normalized_zernike_matrix = self.normalized_zernike_matrix.to('cuda')
        else: 
            raise Exception("vast_weights needs to be True or False")

        if vast_weights:
            warn = 'Warning, the amount of parameters that pytorch lightning prints is vastly larger then the actual amount. A rough estimate for the actual amount is slightly more then the printed amount times {}'.format(torch.sum(self.normalized_zernike_matrix)/torch.numel(self.normalized_zernike_matrix))
            warnings.warn(warn)
        self.vast_weights =vast_weights
        if Skip_connect:
            # This will do a linear layer as skip connect, one could also make it not learnable at all, specifically if in_channels=out_channels. Will only fdo skip connect for input 1.
            self.Skip_connect_flag = True
            out_size = self._calc_size(n_out)
            self.SkipLinear_1 = MultiLinear(in_channels, out_channels, out_size, non_lin=True)
        else:
            self.Skip_connect_flag = False

    def forward(self,in1,in2):
    	# If input 1 was defined at lower order, increase its dimensionality to match the Zernike output space
        if self.increase_in1:
            in1 = torch.einsum('ij,...jk->...ik', self.in_mask_1,in1)

        # Same for input 2 if necessary
        if self.increase_in2:
            in2 = torch.einsum('ij,...jk->...ik', self.in_mask_2,in2)

        if self.Skip_connect_flag:
            if self.reduce:
                out1 = self.SkipLinear_1(in1[:,:,self.out_mask])
            else:
                out1 = self.SkipLinear_1(in1)
        # If using multichannel architecture, apply input projections (e.g. linear layers) to each input
        if self.flag_multichanneled:
            in1 = self.In_Lin1(in1)
            in2 = self.In_Lin2(in2)

        # Set all imaginary components of m = 0 modes to zero to enforce rotation-equivariant constraints
        in1 = torch.einsum('ij,...ij->...ij', self.zero_mask,in1)
        in2 = torch.einsum('ij,...ij->...ij', self.zero_mask,in2)

        if self.vast_weights:
            in1 = torch.einsum('...im,...jn->...ijmn', in1, in2)    # Compute bilinear product using learnable weight matrix
            in1 = torch.einsum('ijkl,ijkl,...ijmn->...klmn',self.Zernike_matrix,self.weight, in1)   # Apply the Zernike product tensor to compute output coefficients in Zernike space
        else:
            in1 = torch.einsum('...im,ij,...jn->...ijmn', in1,self.weight, in2)    # Compute bilinear product using learnable weight matrix
            in1 = torch.einsum('ijkl,...ijmn->...klmn',self.Zernike_matrix, in1)   # Apply the Zernike product tensor to compute output coefficients in Zernike space

        # Do not put anything inbetween, as it might break equivariance

        # Collapse 4 intermediate channels (from different m-combinations) into final form (m, ±1)
        out = torch.einsum('lamn,...klmn->...ka', self.transform,in1)
        #Apply a Non-linearity
        out = self.Nonlin(out)


        # Final output projection (used if multichannel architecture is enabled)
        if self.flag_multichanneled:
            out = self.Out_Lin(out)

        if self.Skip_connect_flag:
            out = out+out1
        return out

    def _load_or_create_zernike_matrix(self, n, image_size, path):
        if os.path.isfile(path):
            return torch.load(path).to(self.device)
        mat = torch.tensor(self.zernicke_matrix_generator(n), dtype=torch.float)
        torch.save(mat, path)
        return mat.to(self.device)

    def zernicke_matrix_generator(self,n_max):
        ''' Generates the full Zernike product interaction matrix '''

        n_max_calc = n_max + 1
        lengh = int( ( ( n_max_calc + 1 ) * n_max_calc / 2 ) / 2 + math.ceil ( n_max_calc / 4 ) )
        grid = np.zeros( ( lengh, lengh, lengh, 4 ) )   # Output tensor

        # Iterate over all valid (m1, n1) and (m2, n2) input pairs
        for m1 in range(n_max + 1):
        	# Index offset for (n1, m1)
            m1_lengh = lengh - int( ( ( n_max_calc - m1 + 1 ) * ( n_max_calc - m1 ) / 2 ) / 2 + math.ceil( ( n_max_calc - m1 ) / 4 ) )
            for m2 in range(n_max+1):
            	# Index offset for (n2, m2)
                m2_lengh = lengh - int(((n_max_calc-m2+1)*(n_max_calc-m2)/2)/2+math.ceil((n_max_calc-m2)/4))
                
                count1 = 0
                for n1 in range(m1, n_max + 1, 2):
                    count2=0
                    for n2 in range(m2,n_max+1,2):
                    	# Compute Zernike product
                        coeffs = self._calculate_matrix_coefficients(m1, m2, n1, n2, n_max)

                        # Fill output tensor at proper location
                        grid[m1_lengh+count1, m2_lengh+count2, :, :] = coeffs
                        count2 +=1
                    count1 +=1

        return grid

    def _calculate_matrix_coefficients(self, m1, m2, n1, n2, n_max):
        '''
        Computes the Zernike product coefficients for (n1, m1) × (n2, m2).

        The result is a vector of coefficients in the Zernike basis representing the product,
    	separated into four different cases depending on the relationship between m1 and m2:
        - Δm = ±(m1 ± m2), split into four channels:
            0 → combined output (cos-cos)
            1 → m1 > m2 (cos-sin)
            2 → m1 == m2 (sin-sin, even)
            3 → m1 < m2 (sin-cos)
		Steps:
		1. Generate radial Zernike functions for (n1, m1) and (n2, m2)
	    2. Multiply their radial profiles as polynomials in r
	    3. Project the product back into valid Zernike modes using transformation matrices
	    4. Distribute the result into the appropriate output bins
		
		Args:
         - m1, m2 (int): angular orders of the two input functions
         - n1, n2 (int): radial orders of the two input functions
         - n_max (int): maximum allowed order for output
        '''

        # Step 1: Get radial polynomials in r**i form
        In1 = self._get_radial_function(n1,m1,n_max)
        In2 = self._get_radial_function(n2,m2,n_max)

        # Step 2: Multiply them (convolution in polynomial space)
        Mult = self._multiply(In1,In2,n_max)

        # Step 3: Compute resulting angular orders
        m_out1 = np.abs(m1 - m2)
        m_out2 = np.abs(m1 + m2)
        m_out2 = np.min( ( m_out2 , n_max + 1 ) )

        # Get conversion matrices from monomial basis to Zernike radial basis
        Mat1 = self._radial_function_matrix(m_out1,n_max)
        Mat2 = self._radial_function_matrix(m_out2, n_max) if m_out2 <= n_max else np.zeros((n_max + 1, n_max + 1))

        # Step 4: Compute output mode offsets (how many basis elements before each m)
        lower = sum(math.ceil(i / 2) for i in range(n_max + 1, n_max + 1 - m_out1, -1))
        higher = sum(math.ceil(i / 2) for i in range(1, n_max + 1 - m_out2))
        inbetween = sum(math.ceil(i / 2) for i in range(n_max + 1 - m_out1 - 1, n_max + 1 - m_out2, -1))
       	
       	# Project result back to Zernike space
        out1 = np.einsum('ij,j->i', Mat1, Mult)[m_out1:][::2]
        out2 = np.einsum('ij,j->i', Mat2, Mult)[m_out2:][::2]

        # Step 5: Place results into appropriate bins

        # Case 0: always include output from m_out2 (combined cosine-cosine)
        out_dim_0 = np.zeros(lower,dtype=float)

        if not m_out1 == m_out2:
            out_dim_0 = np.append(out_dim_0,np.zeros(len(out1)))

        out_dim_0 = np.append(out_dim_0,np.zeros(inbetween))
        out_dim_0 = np.append(out_dim_0,out2)
        out_dim_0 = np.append(out_dim_0,np.zeros(higher))

        # Case 1: m1 > m2 (cos-sin)
        out_dim_1 = np.zeros(lower,dtype=float)
        if m1>m2:
            out_dim_1 = np.append(out_dim_1,out1)
        else:
            out_dim_1 = np.append(out_dim_1,np.zeros(len(out1)))
        out_dim_1 = np.append(out_dim_1,np.zeros(inbetween))

        # Case 2: m1 == m2 (sin-sin)
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

        # Case 3: m1 < m2 (sin-cos)
        out_dim_3 = np.zeros(lower,dtype=float)
        if m1<m2:
            out_dim_3 = np.append(out_dim_3,out1)
        else:
            out_dim_3 = np.append(out_dim_3,np.zeros(len(out1)))
        out_dim_3 = np.append(out_dim_3,np.zeros(inbetween))
        if not m_out1 == m_out2:
            out_dim_3 = np.append(out_dim_3,np.zeros(len(out2)))
        out_dim_3 = np.append(out_dim_3,np.zeros(higher))

        # Final result: shape (num_output_modes, 4)
        out = np.transpose(np.array([out_dim_0,out_dim_1,out_dim_2,out_dim_3]))
        return out

    def _get_radial_function(self, n, m, n_max):            
        ''' Compute the radial function of Zernike polynomials '''

        factor = []

        # Ensure there are zeros for indices smaller than (n - m)
        for i in range(n_max - n):
            factor.append(0)

        # Loop over k to calculate the radial polynomial terms
        for k in range(int((n - m) / 2 + 1)):
            num_ = (-1) ** k * math.factorial(n - k)
            denom = (math.factorial(k) *
                     math.factorial((n + m) // 2 - k) *
                     math.factorial((n - m) // 2 - k))
            
            term = num_ / denom
            factor.append(term)

            # If not at the midpoint, append a zero to maintain the symmetry
            if k != int((n - m) / 2):
                factor.append(0)
                
        # Fill in zeros for higher m terms
        for i in range(m):
            factor.append(0)

        factor = np.array(factor)

        return np.flip(factor) 

    def _multiply(self, x, y, n_max):
        '''
        Multiplies two radial polynomials defined in terms of powers of r.
        The Zernike radial functions are represented as coefficient arrays for polynomials in r^i,
    	but stored in reverse order compared to what "convolve" expects (highest degree first).
    	This function flips both input polynomials, performs convolution, then flips the result back.
    	It also truncates the result to keep only terms up to degree n_max.

    	Args:
         - x (np.ndarray): Coefficients of the first polynomial (highest to lowest degree)
         - y (np.ndarray): Coefficients of the second polynomial (same order as x)
         - n_max (int): Maximum radial degree to keep in the output
        '''
        # Flip to match expected input for np.convolve (lowest degree first)
        x = np.flip(x)
        y = np.flip(y)

        # Convolve and truncate to degree n_max (keep highest terms only)
        result = np.convolve(x,y)[-n_max-1:]

        # Flip back to original coefficient order (highest to lowest)
        return np.flip(result)

    def _radial_function_matrix(self,m, n_max):
        '''
        Constructs a matrix that transforms any valid polinomial to it's represantation in 
        terms of radial polinomials of given order m. Valid polynomials a_i r**i are gives by a_i = 0 
        for all i<m and a_i = 0 for all i-m not even.

        This is done by creating a matrix relating all polinomials in question to their representation 
        in terms of powers of r up to i=n_max. This matrix is then numerically inverted to have it point 
        from the space of powers of r to the space of radial polinomials. All powers of r which are zero 
        by the rules given above are filled with a one on the diagonal. This is done so the matrix is 
        still invertable, yet has no impact on the final output.

        Steps:
	    1. Build a square matrix where each row is a Zernike radial polynomial (or placeholder for invalid powers).
	    2. The polynomials are expressed in terms of r^i up to degree n_max.
	    3. Pad invalid monomials with identity rows to preserve invertibility.
	    4. Solve the resulting triangular system to get the transformation matrix.

	    Args: 
	     - m (int): fixed angular order for the radial functions.
         - n_max (int): maximum radial order to support (also determines polynomial degree).

        '''
        matrix = []     
        empty = np.zeros(n_max + 1)

        # Step 1: Add identity rows for invalid lower-order monomials (r^i where i < m)
        for i in range(m):
            empty *=0
            empty[n_max-i] = 1         # Fill from the right (highest degree first)
            matrix.append(empty.copy())

        # Step 2: Add normalized Zernike radial polynomials for valid (n, m)
        for n in range(m, n_max + 1, 2):
        	# Build Zernike radial polynomial
            faktor = []
            for i in range(int(( n_max - n ))):
                faktor.append(0)
            for k in range(int(( n - m ) / 2 + 1)):
                faktor.append((-1)**k * math.factorial(n - k) /(
                							math.factorial(k) * 
                							math.factorial(int((n + m) / 2 - k))* 
                							math.factorial(int((n - m) / 2 - k)))  )
                if k != int((n - m) / 2):
                    faktor.append(0)

            for i in range(m):
                faktor.append(0)

            norm = self.Zernike_normalization[n][m]   # Normalize with precomputed value
            faktor = np.array(faktor)/norm
            matrix.append((faktor.copy()))

            # Step 3: Fill in additional identity rows to preserve matrix shape
            if n != n_max:
                empty *= 0
                empty[n_max - n - 1] = 1
                matrix.append(empty.copy())

        # Step 4: Convert matrix to lower-triangular form and invert        
        matrix = (np.rot90(np.vstack(np.array(matrix))))  # Flip to align polynomial degrees properly

        return scipy.linalg.solve_triangular(matrix, np.identity(n_max + 1))


    def _calc_size(self,n_max):
        '''
        Calculating the amount of terms in the Zernike decomposition depending on n. This calculates the amount of radial polinomes, the final decomposition will have size= (calc_size(n),2)
        '''
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def _init_transform_tensor(self):
        Matrix_plus = [[[1/2, 0], [0, -1/2]], [[0, 1/2], [1/2, 0]]]
        Matrix_minus_pos = [[[1/2, 0], [0, 1/2]], [[0, -1/2], [1/2, 0]]]
        Matrix_minus_neut = [[[1/2, 0], [0, 1/2]], [[0, 0], [0, 0]]]
        Matrix_minus_neg = [[[1/2, 0], [0, 1/2]], [[0, 1/2], [-1/2, 0]]]
        matrices = [Matrix_plus, Matrix_minus_pos, Matrix_minus_neut, Matrix_minus_neg]

        return torch.tensor(np.array(matrices), dtype=torch.float, device=self.device)
    

    def _create_mask_decrease(self, n_max, n_out):
        '''
        Creates a mask to convert input of order n_max to output of order n_out, with n_max>n_out
        This is used when input (n_max) needs to be embedded into a lower-order basis (n_out). 

        '''
        n_max_calc = n_max + 1
        lengh = int( ( ( n_max_calc + 1 ) * n_max_calc / 2 ) / 2 + math.ceil( n_max_calc / 4 ) )
        mask = torch.ones(lengh, device=self.device)

        for m1 in range(n_max + 1):
            m1_lengh = lengh - int( ( ( n_max_calc - m1 + 1 ) * ( n_max_calc - m1 ) / 2 ) / 2 + math.ceil( ( n_max_calc - m1 ) / 4 ) )
            count = 0
            for n1 in range(m1, n_max+1, 2):
                if m1>n_out or n1>n_out:
                    mask[m1_lengh + count] -= 1
                count+=1

        mask = mask.bool()    
        return mask

    def _create_mask_increase(self, n_max, n_out):
        '''
        Creates a mask to convert input of order n_max to output of order n_out. 
        This is used when input (n_max) needs to be embedded into a higher-order basis (n_out). 
        '''
        n_out_calc = n_out + 1
        n_max_calc = n_max + 1
        lengh_out = int( ( ( n_out_calc + 1 ) * n_out_calc / 2 ) / 2 + math.ceil( n_out_calc / 4 ) )     
        lengh_in = int( ( ( n_max_calc + 1 ) * n_max_calc / 2 ) / 2 + math.ceil( n_max_calc / 4 ) )
        
        mask = torch.zeros(lengh_out, lengh_in, device=self.device)
        
        for m in range(n_out+1):
            m1_lengh = lengh_out - int(((n_out_calc - m + 1) * (n_out_calc - m) / 2) / 2 + math.ceil((n_out_calc - m) / 4))
            
            count=0
            for n1 in range(m, n_out+1, 2):
                m_in_lengh = lengh_in - int(((n_max_calc - m + 1) * (n_max_calc - m) / 2) / 2 + math.ceil((n_max_calc - m) / 4))
                if not (m>n_max or n1>n_max):
                    mask[m1_lengh + count, m_in_lengh + count] += 1
                count+=1

        return mask


    def _create_zero_mask(self,n_max):
        '''
        Creates a mask to explicitly zero out sine components (m=0, sin(mθ)) in Zernike representations.

	    In Zernike polynomials, the sine part of m=0 modes is undefined or identically zero:
	        sin(0 * θ) = 0 ∀ θ
	    These components do not carry any useful information and can cause issues
	    if processed through layers with bias or numerical noise.

	    This mask ensures that:
	    - The sine component (index 1) of modes with m=0 is explicitly set to zero.
	    - All other (n, m) modes retain both cosine (index 0) and sine (index 1) components.
        '''

        n_max_calc = n_max + 1

        lengh = int( ( ( n_max_calc + 1 ) * n_max_calc / 2 ) / 2 + math.ceil( n_max_calc / 4 ) )
        mask = torch.ones(lengh, 2, device=self.device)

        # Manually zero out all sine components for m=0 modes
    	# These are always the first (n//2)+1 entries in the flattened Zernike functions
        for i in range( 0, ( n_max // 2) + 1):
            mask[i,1] = 0

        return mask



   

    

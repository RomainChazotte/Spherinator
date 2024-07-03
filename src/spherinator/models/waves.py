
from scipy.constants import physical_constants
import matplotlib.pyplot as plt
import scipy.special as sp
import seaborn as sns
import numpy as np
import argparse
import torch
import h5py
from zernike import RZern
import numpy
import numpy as np
from numpy import convolve
import scipy
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
# got this code from
# https://github.com/ssebastianmag/hydrogen-wavefunctions/tree/main



class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 ):
        super().__init__()
        self.Zernike_matrix = torch.tensor(np.array(self.create_filter(n_max)),dtype=torch.float)
        size = self.calc_size(n_max)

    def calc_size(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def M_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros(lengh)
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count] = m1
                count+=1
        return Basis


    def Radial_function(self,n,m, n_max):
        faktor = []

        scaling = []
        for i in range(n_max+n_max+1):
            scaling.append(1/((2*n_max-i)+2))

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
        #faktor = np.array(faktor)
        return np.flip(faktor)
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)


    def create_filter(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = []
        for i in range(len(Zernike_functions)):

            functions.append(numpy.polynomial.polynomial.Polynomial(Zernike_functions[i]))

        eps = np.finfo(float).eps
        out = []
        M = self.M_embedding_generator(n_max)
        for i in range(len(Zernike_functions)):
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z + eps))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  + eps)))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        out = out*out_mask
    def Zernicke_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros((lengh,n_max+1))
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count,:] = self.Radial_function(n1,m1,n_max)
                count+=1
        return Basis
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)


    def create_filter(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = []
        for i in range(len(Zernike_functions)):

            functions.append(numpy.polynomial.polynomial.Polynomial(Zernike_functions[i]))

        eps = np.finfo(float).eps
        out = []
        M = self.M_embedding_generator(n_max)
        for i in range(len(Zernike_functions)):
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z + eps))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  + eps)))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        out = out*out_mask
        '''
        import matplotlib.pyplot as plt
        for a in range(0,len(out),4):
            #print('hi')
            plt.figure()
            plt.imshow(out[a,0])
            plt.savefig('Filtercos{}.png'.format(a))
            plt.close()
            plt.figure()
            plt.imshow(out[a,1])
            plt.savefig('Filtersin{}.png'.format(a))
            plt.close()
        '''
        return out

    def forward(self):
        norm = (torch.sum(torch.abs(self.Zernike_matrix),dim= (-1,-2),keepdim = True))
        #eps = 0.0000005
        print(norm)
        #self.Zernike_matrix = self.Zernike_matrix/(norm+eps)
        #This should be implemented in init, do this later
        print(self.Zernike_matrix.size())
        '''
        print(input.size())
        #out = torch.einsum('ijkl,...kl->...ij',self.Zernike_matrix,input)
        print(out.size())
        '''
        return self.Zernike_matrix

x = Zernike_embedding(30)
a = x()

donkey
#for i in range(16):
cart = RZern(16)
L, K = 128, 128
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)

c = np.zeros(cart.nk)
print(len(c))
plt.figure(1)
Zer = []
for i in range(1, 136):
    plt.subplot(12, 12, i)
    c *= 0.0
    #c[3] = 1.0
    c[i-1] = 1.0
    Phi = cart.eval_grid(c, matrix=True)
    #Phi =
    np.nan_to_num(Phi,copy=False)
    Zer.append(Phi)
    #print(Phi)
    plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
    plt.axis('off')
Zer = torch.tensor(np.array(Zer))
plt.savefig('zerpic.png')
plt.close()

x = Zernike_embedding(16)
out = np.array(x())
plt.figure(1)
for i in range(1,136):
    plt.subplot(12, 12, i)
    plt.imshow(out[(i-1)//2,(i-1)%2], origin='lower', extent=(-1, 1, -1, 1))
    plt.axis('off')

plt.savefig('zerpiccustom.png')
plt.close()


donkey
class IMGwaves():
    def __init__(self, n_max,l_max ):
        self.n = n_max
        self.l_max = l_max
    def forward(self):
        #print(self.compute_wavefunction(1, 0, 0, 0.1))
        out = torch.tensor(np.array([self.compute_wavefunction(n, l, m, 0.04)  for l in range(self.l_max+1) for n in range(self.n) for m in range(-l,l+1)]))
        return out


    def radial_function(self,n, l, r, a0):
        """ Compute the normalized radial part of the wavefunction using
        Laguerre polynomials and an exponential decay factor.

        Args:
            n (int): principal quantum number
            l (int): azimuthal quantum number
            r (numpy.ndarray): radial coordinate
            a0 (float): scaled Bohr radius
        Returns:
            numpy.ndarray: wavefunction radial component
        """
        scale = 14
        x = (r-n*a0*scale)**2/(8*(a0*scale)**2/(self.n-n+1))
        custom_rad = np.exp(-x)/(a0*scale*np.sqrt(8*np.pi/(self.n-n+1)))


        return custom_rad


    def angular_function(self,m, l, theta, phi):
        """ Compute the normalized angular part of the wavefunction using
        Legendre polynomials and a phase-shifting exponential factor.

        Args:
            m (int): magnetic quantum number
            l (int): azimuthal quantum number
            theta (numpy.ndarray): polar angle
            phi (int): azimuthal angle
        Returns:
            numpy.ndarray: wavefunction angular component
        """

        legendre = sp.lpmv(m, l, np.cos(theta))

        constant_factor = ((-1) ** m) * np.sqrt(
            ((2 * l + 1) * sp.factorial(l - np.abs(m))) /
            (4 * np.pi * sp.factorial(l + np.abs(m)))
        )
        return constant_factor * legendre * np.real(np.exp(1.j * m * phi))


    def compute_wavefunction(self,n, l, m, a0_scale_factor):
        """ Compute the normalized wavefunction as a product
        of its radial and angular components.

        Args:
            n (int): principal quantum number
            l (int): azimuthal quantum number
            m (int): magnetic quantum number
            a0_scale_factor (float): Bohr radius scale factor
        Returns:
            numpy.ndarray: wavefunction
        """





        '''
        cart = RZern(6)
        L, K = 200, 250
        ddx = np.linspace(-1.0, 1.0, K)
        ddy = np.linspace(-1.0, 1.0, L)
        xv, yv = np.meshgrid(ddx, ddy)
        cart.make_cart_grid(xv, yv)

        c0 = np.random.normal(size=cart.nk)
        Phi = cart.eval_grid(c0, matrix=True)
        c1 = cart.fit_cart_grid(Phi)[0]
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.plot(range(1, cart.nk + 1), c0, marker='.')
        plt.plot(range(1, cart.nk + 1), c1, marker='.')
        '''



        cart = RZern(10)
        L, K = 200, 250
        ddx = np.linspace(-1.0, 1.0, K)
        ddy = np.linspace(-1.0, 1.0, L)
        xv, yv = np.meshgrid(ddx, ddy)
        cart.make_cart_grid(xv, yv)

        c = np.zeros(cart.nk)
        plt.figure(1)
        for i in range(1, 100):
            plt.subplot(10, 10, i)
            c *= 0.0
            c[i] = 1.0
            Phi = cart.eval_grid(c, matrix=True)
            plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
            plt.axis('off')

        plt.savefig('zerpic.png')
        plt.close()
        x = Zernike_embedding(20)
        out = x()
        plt.figure(1)
        for i in range(100):
            plt.subplot(10, 10, i)
            plt.imshow(out[i//2,i%2], origin='lower', extent=(-1, 1, -1, 1))
            plt.axis('off')

        plt.savefig('zerpiccustom.png')
        plt.close()



        # Scale Bohr radius for effective visualization
        a0 = a0_scale_factor * physical_constants['Bohr radius'][0] * 1e+12

        # z-x plane grid to represent electron spatial distribution
        grid_extend = 480
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        # Use epsilon to avoid division by zero during angle calculations
        eps = np.finfo(float).eps

        # Ψnlm(r,θ,φ) = Rnl(r).Ylm(θ,φ)
        psi = self.radial_function(
            n, l, np.sqrt((x ** 2 + z ** 2)), a0
        ) * self.angular_function(
            m, l, np.arctan(x / (z + eps)), 0
        )
        return psi

if __name__ == '__main__':

    '''
    for i in range(11):
        waves = IMGwaves(i+3,3)
        out = torch.tensor(waves.forward())
        import matplotlib.pyplot as plt
        plt.figure()
        print(len(out))
        plt.imshow(out[-3].T)
        plt.savefig('pic{}.png'.format(i+3))
        plt.close()
    '''
    waves = IMGwaves(10,10)
    out = torch.tensor(waves.forward())
    import matplotlib.pyplot as plt
    for i in range(24):
        plt.figure()
        print(len(out))
        plt.imshow(out[-i].T)
        plt.savefig('pic{}.png'.format(i))
        plt.close()

class IMGwaves():
    def __init__(self, n_max,l_max ):
        self.n = n_max
        self.l_max = l_max
    def forward(self):
        #print(self.compute_wavefunction(1, 0, 0, 0.1))
        out = torch.tensor(np.array([self.compute_wavefunction(n, l, m, 0.04)  for l in range(self.l_max+1) for n in range(self.n) for m in range(-l,l+1)]))
        return out


    def radial_function(self,n, l, r, a0):
        """ Compute the normalized radial part of the wavefunction using
        Laguerre polynomials and an exponential decay factor.

        Args:
            n (int): principal quantum number
            l (int): azimuthal quantum number
            r (numpy.ndarray): radial coordinate
            a0 (float): scaled Bohr radius
        Returns:
            numpy.ndarray: wavefunction radial component
        """
        scale = 14.
        x = ((r-n*a0*scale)**2/(8*(a0*scale)**2/(self.n-n+1)))
        x = np.where(x<600,x,600)
        custom_rad = np.exp(-x)
        custum_rad =custom_rad/(a0*scale*np.sqrt(8*np.pi/(self.n-n+1)))


        return custom_rad


    def angular_function(self,m, l, theta, phi):
        """ Compute the normalized angular part of the wavefunction using
        Legendre polynomials and a phase-shifting exponential factor.

        Args:
            m (int): magnetic quantum number
            l (int): azimuthal quantum number
            theta (numpy.ndarray): polar angle
            phi (int): azimuthal angle
        Returns:
            numpy.ndarray: wavefunction angular component
        """

        legendre = sp.lpmv(m, l, np.cos(theta))

        constant_factor = ((-1) ** m) * np.sqrt(
            ((2 * l + 1) * sp.factorial(l - np.abs(m))) /
            (4 * np.pi * sp.factorial(l + np.abs(m)))
        )
        return constant_factor * legendre * np.real(np.exp(1.j * m * phi))


    def compute_wavefunction(self,n, l, m, a0_scale_factor):
        """ Compute the normalized wavefunction as a product
        of its radial and angular components.

        Args:
            n (int): principal quantum number
            l (int): azimuthal quantum number
            m (int): magnetic quantum number
            a0_scale_factor (float): Bohr radius scale factor
        Returns:
            numpy.ndarray: wavefunction
        """

        # Scale Bohr radius for effective visualization
        a0 = a0_scale_factor * physical_constants['Bohr radius'][0] * 1e+12

        # z-x plane grid to represent electron spatial distribution
        grid_extend = 480
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        # Use epsilon to avoid division by zero during angle calculations
        eps = np.finfo(float).eps

        # Ψnlm(r,θ,φ) = Rnl(r).Ylm(θ,φ)
        psi = self.radial_function(
            n, l, np.sqrt((x ** 2 + z ** 2)), a0
        ) * self.angular_function(
            m, l, np.arctan(x / (z + eps)), 0
        )
        return psi


class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 ):
        super().__init__()
        self.Zernike_matrix = torch.tensor(np.array(self.create_filter(n_max)),dtype=torch.float)
        size = self.calc_size(n_max)

    def calc_size(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def M_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros(lengh)
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count] = m1
                count+=1
        return Basis


    def Radial_function(self,n,m, n_max):
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
        #faktor = np.array(faktor)
        return np.flip(faktor)

    def Zernicke_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros((lengh,n_max+1))
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count,:] = self.Radial_function(n1,m1,n_max)
                count+=1
        return Basis
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,y,0)


    def create_filter(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = []
        for i in range(len(Zernike_functions)):

            functions.append(numpy.polynomial.polynomial.Polynomial(Zernike_functions[i]))

        eps = np.finfo(float).eps
        out = []
        M = self.M_embedding_generator(n_max)
        for i in range(len(Zernike_functions)):
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan(x / (np.abs(z) + eps))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan(x / (np.abs(z)  + eps)))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        out = out*out_mask
        '''
        import matplotlib.pyplot as plt
        for a in range(0,len(out),4):
            #print('hi')
            plt.figure()
            plt.imshow(out[a,0])
            plt.savefig('Filtercos{}.png'.format(a))
            plt.close()
            plt.figure()
            plt.imshow(out[a,1])
            plt.savefig('Filtersin{}.png'.format(a))
            plt.close()
        '''
        return out

    def forward(self):
        norm = (torch.sum(torch.abs(self.Zernike_matrix),dim= (-1,-2),keepdim = True))
        eps = 0.0000005
        self.Zernike_matrix = self.Zernike_matrix/(norm+eps)
        #This should be implemented in init, do this later
        '''
        print(self.Zernike_matrix.size())
        print(input.size())
        #out = torch.einsum('ijkl,...kl->...ij',self.Zernike_matrix,input)
        print(out.size())
        '''
        return self.Zernike_matrix


if __name__ == '__main__':

    '''
    for i in range(11):
        waves = IMGwaves(i+3,3)
        out = torch.tensor(waves.forward())
        import matplotlib.pyplot as plt
        plt.figure()
        print(len(out))
        plt.imshow(out[-3].T)
        plt.savefig('pic{}.png'.format(i+3))
        plt.close()
    '''
    waves = IMGwaves(10,7)
    out = torch.tensor(waves.forward())
    import matplotlib.pyplot as plt
    for i in range(24):
        plt.figure()
        print(len(out))
        plt.imshow(out[-i].T)
        plt.savefig('pic{}.png'.format(i))
        plt.close()
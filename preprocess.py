""" Provides access to the galaxy zoo images.
"""

import os

import numpy
import skimage.io as io
import torch
import torch.nn as nn

#from spherinator.models import zernike_classifier

#from ..models.zernike_classifier import Zernike_embedding
#from .spherinator_dataset import SpherinatorDataset

class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 , device = 'cuda:2', numerical_expand = 4 ):
        super().__init__()
        self.num = numerical_expand
        if os.path.isfile('Zernike_decode_encode_size424_{}'.format(n_max)) :
            self.Zernike_matrix = torch.load('Zernike_decode_encode_size424_{}'.format(n_max))
        else:
            self.Zernike_matrix = self.create_filter(n_max)
            torch.save(self.Zernike_matrix,'Zernike_decode_encode_size424_{}'.format(n_max))
        #size = self.calc_size(n_max)


        #self.Zernike_matrix = self.create_filter(n_max)
        self.Zernike_matrix = self.Zernike_matrix.to(device)#*16
        #self.norm_matrix = np.array(self.norm_matrix)
        #self.Zernike_matrix= torch.nn.parameter.Parameter(self.Zernike_matrix,requires_grad=False)
        #self.device = 'cuda:2'
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
        return np.where(y<1,1,0)


    def create_filter(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, int(424*self.num))
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
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z ))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  )))])
        #print(out[0])
        # Add restriction to r<1
        #out_mask = self.mask(x,z)
        #out = torch.tensor(np.array(out*out_mask),dtype=torch.float)#, device =  'cuda:2')

        # norm = []
        # for i in range(len(Zernike_functions)):
        #     norm.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z )))+eps,functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  )))+eps])

        # norm = torch.tensor(np.array(norm*out_mask),dtype=torch.float)

        # norm = torch.sqrt((torch.sum((norm)**2,dim= (-1,-2),keepdim = True)))*self.num


        # out = out/norm
        out = np.array(out)
        out =torch.tensor( block_reduce(out,(1,1, self.num, self.num),func=np.sum),dtype=torch.float)


        z = x = np.linspace(-grid_extend, grid_extend, int(424))
        z, x = np.meshgrid(z, x)
        out_mask = torch.tensor( self.mask(x,z))
        out= torch.einsum('ijkl,kl->ijkl',out,out_mask)

        norm = torch.sqrt((torch.sum((out)**2,dim= (-1,-2),keepdim = True)))+eps
        out = out/norm
        return out#*self.num

    def embed(self,input):
        out = torch.einsum('ijkl,...kl->...ij',self.Zernike_matrix,input)
        return out

    def decode(self,input):
        out = torch.einsum('ijkl,...ij->...kl',self.Zernike_matrix,input)
        return out

class GalaxyZooDataset():
    """Provides access to galaxy zoo images."""

    def __init__(
        self,
        data_directory: str= "/hits/fast/ain/Data/Morphology_GalaxyZoo/images_training_rev1",
        extension: str = "jpg",
        label_file: str = '/hits/fast/ain/Data/Morphology_GalaxyZoo/training_solutions_rev1.csv',
        transform=None,
    ):
        """Initializes an galaxy zoo data set.

        Args:
            data_directory (str): The directory that contains the images for this dataset.
            extension (str, optional): The file extension to use when searching for file.
                Defaults to "jpeg".
            label_file (str): The name of the file that contains the labels used for training of
                testing. By default None is specified. In this case no labels will be returned for
                the individual items!
            transform (torchvision.transforms.Compose, optional): A single or a set of
                transformations to modify the images. Defaults to None.
        """
        print('init')
        self.data_directory = data_directory
        self.transform = transform
        self.files = []
        for file in os.listdir(data_directory):
            if file.endswith(extension):
                self.files.append(os.path.join(data_directory, file))

        #print(len(self.files))
        # if label_file is str():
        #     self.labels = torch.Tensor(numpy.zeros(self.len))
        # else:
        self.labels = torch.Tensor(
            numpy.loadtxt(label_file, delimiter=",", skiprows=1)[:, 1:]
        )
        #print(self.labels.size())
        self.current_index = []
        self.embed = Zernike_embedding(32,device='cpu')
        #print(self.files)
        for i in range(len(self.files)):
            self.process(i)
            #print('process')


    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.files)

    def process(self, index: int):
        """Retrieves the item/items with the given indices from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            data: Data of the item/items with the given indices.
        """
        self.current_index = index
        data = io.imread(self.files[index])
        data = torch.Tensor(data)
        # Swap axis 0 and 2 to bring the color channel to the front
        data = torch.swapaxes(data, 0, 2)
        # Normalize the RGB values to values between 0 and 1
        data /= 255.0
        data = self.embed.embed(data)
        torch.save(data,'/local_data/AIN/chazotrn/Morphology_GalaxyZoo/Zernike_images/{}.pt'.format(index))

        if self.transform:
            data = self.transform(data)



    def get_metadata(self, index: int):
        """Retrieves the metadata of the item/items with the given indices from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            metadata: Metadata of the item/items with the given indices.
        """
        metadata = {
            "filename": self.files[index],
            "labels": self.labels[index],
        }
        return metadata
x = GalaxyZooDataset()
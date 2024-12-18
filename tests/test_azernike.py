import torch

from spherinator.models import (ZernikeClassifier, ZernikeDecoder,
                                ZernikeEncoder)

#model = ZernikeAutoencoder()
# def test_encoding_equivariance():
#     """Check if weights are reproducible"""
#     model = ZernikeAutoencoder()#.to('cuda:2')
#     input = model.example_input_array#.to('cuda:2')

#     assert torch.isclose(
#         torch.rot90(model.Decoding_Function(model.Embedding_Function(input)), k=1, dims=[-2, -1]), model.Decoding_Function(model.Embedding_Function(torch.rot90(input, k=1, dims=[-2, -1]))), atol=1e-5
#     ).all()

# def test_model_equivariance():
#     """Check if weights are reproducible"""
#     model = ZernikeAutoencoder()#.to('cuda:2')
#     input = model.example_input_array#.to('cuda:2')
#     input_rot=model.Embedding_Function(torch.rot90(input, k=1, dims=[-2, -1]))
#     input = model.Embedding_Function(input)
#     output = torch.rot90(model.Decoding_Function(model.encoder.Product0(input,input)), k=1, dims=[-2, -1])
#     output_rot = model.Decoding_Function(model.encoder.Product0(input_rot,input_rot))
#     output_sum = torch.sum(output)
#     output_sum_rot = torch.sum(output_rot)
#     print(output)
#     assert torch.isclose(
#         output/output_sum, output_rot/output_sum_rot, atol=1e-3
#     ).all()
#     assert torch.isclose(
#         output_sum, output_sum_rot, atol=1e-3
#     ).all()
#     assert torch.isclose(
#         output, output_rot, atol=1e-3
#     ).all()

# def test_embedding_equivariance():
#     """Check if weights are reproducible"""
#     model = ZernikeClassifier().to('cuda:2')
#     input = model.example_input_array.to('cuda:2')
#     print('rot')
#     output_rot,_=model.Multiscale_filters.embed(torch.rot90(input, k=1, dims=[-2, -1]))
#     print('normal')
#     output,_ = model.Multiscale_filters.embed(input)
#     print(output.size())
#     output = torch.rot90(output, k=1, dims=[0, 1])
#     # output = model.encode(input)
#     # output_rot = (model.encode(input_rot))
#     output_sum = torch.sum(output)
#     output_sum_rot = torch.sum(output_rot)
#     # print((output[:,:,:,3,0]-output_rot[:,:,:,3,0])/output_rot[:,:,:,3,0])
#     # print(output_rot[:,:,:,3])
#     # print(output[:,:,:,3])
#     # print(output/output_sum)
#     # print(output_rot/output_sum_rot)
#     assert torch.isclose(
#         output/output_sum, output_rot/output_sum_rot, atol=1e-3
#     ).all()
#     assert torch.isclose(
#         output_sum, output_sum_rot, atol=1e-3
#     ).all()
#     assert torch.isclose(
#         output, output_rot, atol=1e-3
#     ).all()


def test_full_model_equivariance():
    """Check if weights are reproducible"""
    model = ZernikeClassifier().to('cuda:2')
    input = model.example_input_array.to('cuda:2')
    print('rot')
    #print(torch.rot90(input, k=1, dims=[-2, -1]))
    output_rot,_=model(torch.rot90(input, k=1, dims=[-2, -1]))[0]
    print('normal')
    #print(input)
    output,_ = model(input)[0]
    print(output.size())
    #output = torch.rot90(output, k=1, dims=[1, 2])
    # output = model.encode(input)
    # output_rot = (model.encode(input_rot))
    output_sum = torch.sum(output)

    output_sum_rot = torch.sum(output_rot)
    print(output)
    print(output_rot)
    # print(output/output_sum)
    # print(output_rot/output_sum_rot)
    # assert torch.isclose(
    #     output/output_sum, output_rot/output_sum_rot, atol=1e-3
    # ).all()
    assert torch.isclose(
        output_sum, output_sum_rot, atol=1e-3
    ).all()
    assert torch.isclose(
        output, output_rot, atol=1e-3
    ).all()
#

# def test_small_embedding_equivariance():
#     """Check if weights are reproducible"""
#     model = ZernikeClassifier().to('cuda:2')
#     input = model.example_input_array.to('cuda:2')
#     print('rot')
#     #print(torch.rot90(input, k=1, dims=[-2, -1]))
#     print(input.size())
#     #input = torch.nn.functional.pad(input, (3,3,3,3), "constant",0)
#     print(input.size())
#     print('input')
#     print(input)
#     print(model.Multiscale_filters.Zernike_matrix[0,0,0,0])
#     print(torch.einsum('ij,lmij->lm',model.Multiscale_filters.Zernike_matrix[0,0,0,0],input))
#     print(torch.einsum('mnijkl,...akl->...mnaij',model.Multiscale_filters.Zernike_matrix,input)[0,0,0,0])
#     print(model.Multiscale_filters.embed(input)[0,0,0,0])
#     #donkey
#     output_rot= model.Multiscale_filters.embed(torch.rot90(input, k=1, dims=[-2, -1]))[0]
#     #output = model.Multiscale_filters.Zernike_matrix

#     #output_rot= torch.rot90(output, dims=[-2, -1])
#     #print(output_rot.size())
#     #output_rot,_=model(torch.rot90(input, k=1, dims=[-2, -1]))[0]
#     #print('normal')
#     #print(input)
#     output= model.Multiscale_filters.embed(input)[0]
#     output = torch.rot90(output, k=1, dims=[0, 1])
#     # output = model.encode(input)
#     # output_rot = (model.encode(input_rot))
#     # for i in range(output.size(3)):
#     #     print(i)
#     #     print(output[:,:,0,i,0])
#     #     print(output_rot[:,:,0,i,0])
#     #     print(torch.max(torch.abs(output[:,:,0,i,0]-output_rot[:,:,0,i,0])))
#     #     print(output[:,:,0,i,1])
#     #     print(output_rot[:,:,0,i,1])
#     #     print(torch.max(torch.abs(output[:,:,0,i,1]-output_rot[:,:,0,i,1])))
#     # for k in range(output.size(2)):
#     #     for i in range(output.size(0)):
#     #         for j in range(output.size(0)):
#     #             print(k,i,j)
#     #             print(output[i,j,k,1])
#     #             #print(output_rot[3,3,i,0])
#     #             #print(torch.max(torch.abs(output[:,:,0,i,0]-output_rot[:,:,0,i,0])))
#     #             #print(output[i,i,3,1])
#     #             #print(output_rot[3,3,i,1])
#     #             #print(torch.max(torch.abs(output[:,:,0,i,1]-output_rot[:,:,0,i,1])))
#     print(output.size())
#     #output[13,13,:,:,:] = 0
#     #output_rot[13,13,:,:,:] = 0
#     output_sum = torch.sum(output)
#     output_sum_rot = torch.sum(output_rot)
#     print('set to 0')

#     print(output[13,13,:,:,:])
#     for i in range(output.size(3)):
#         print(i)
#         print(output[:,:,0,i,0])
#         #print(output[i,i,0,3,1])
#         #print(output_rot[3,3,i,0])
#         #print(torch.max(torch.abs(output[:,:,0,i,0]-output_rot[:,:,0,i,0])))
#         #print(output[i,i,3,1])

#     for i in range(output.size(3)):
#         print(i)
#         print(output[:,:,0,i,0])
#         print(output_rot[:,:,0,i,0])
#         #print(output[i,i,0,3,1])
#         #print(output_rot[3,3,i,0])
#         print(torch.max(torch.abs(output[:,:,0,i,0]-output_rot[:,:,0,i,0])))
#         print(torch.max(torch.abs(output[:,:,0,i,1]-output_rot[:,:,0,i,1])))
#         print(output[:,:,0,i,1])
#         print(output_rot[:,:,0,i,1])
#         #print(output[i,i,3,1])
#     #print(output/output_sum)
#     #print(output_rot/output_sum_rot)
#     #print(output_rot.size())
#     # assert torch.isclose(
#     #     output/output_sum, output_rot/output_sum_rot, atol=1e-3
#     # ).all()
#     assert torch.isclose(
#         output_sum, output_sum_rot, atol=1e-3
#     ).all()
#     assert torch.isclose(
#         output, output_rot, atol=1e-3
#     ).all()


test_full_model_equivariance()
#test_small_embedding_equivariance()
#test_embedding_equivariance()
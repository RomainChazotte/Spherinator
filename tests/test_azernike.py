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



def test_full_model_equivariance():
    """Check if weights are reproducible"""
    model = ZernikeClassifier().to('cuda:1')
    input = model.example_input_array.to('cuda:1')
    print('rot')
    output_rot,_=model(torch.rot90(input, k=1, dims=[-2, -1]))
    print('normal')
    output,_ = model(input)
    # output = model.encode(input)
    # output_rot = (model.encode(input_rot))
    output_sum = torch.sum(output)
    output_sum_rot = torch.sum(output_rot)
    print(output)
    print(output_rot)
    print(output/output_sum)
    print(output_rot/output_sum_rot)
    assert torch.isclose(
        output/output_sum, output_rot/output_sum_rot, atol=1e-3
    ).all()
    assert torch.isclose(
        output_sum, output_sum_rot, atol=1e-3
    ).all()
    assert torch.isclose(
        output, output_rot, atol=1e-3
    ).all()

test_full_model_equivariance()
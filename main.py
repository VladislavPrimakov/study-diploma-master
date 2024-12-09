from networks import EmbeddingNet, SiameseNetSigmoid
import torch

torch.onnx.export(SiameseNetSigmoid(1), torch.randn(2, 1, 28, 28) , "model.onnx", input_names=["MNIST"], output_names=["Embedding"])

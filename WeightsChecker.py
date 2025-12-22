import torch
model1 = torch.load("D:\MODELS\V17.pth_epoch_1.pth")
model2 = torch.load("D:\MODELS\V17.pth_epoch_4.pth")

same = all(torch.equal(model1[k], model2[k]) for k in model1)
print("Same weights:", same)

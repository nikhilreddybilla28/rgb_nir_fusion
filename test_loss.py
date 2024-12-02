import mmseg.models.losses.cross_entropy_loss as CLE
import torch
import torch.nn.functional as F

# target = torch.normal()
# preds = torch.rand(size=(1,3,1,512))

# target = torch.tensor([[[1,2],[3,1]]]) # NxHxW
# preds = torch.tensor([[[[0.1,0.6,0.2,0.1],[0.2,0.1,0.2,0.5]],
#                        [[0.05,0.2,0.45,0.3],[0.6,0.1,0.2,0.1]]]]) # NxHxWxC

target = torch.tensor([[[1,255]]]).cuda() # NxHxW
preds = torch.tensor([[[[0.1,0.6,0.2,0.1],[0.25,0.25,0.25,0.25]]]]).cuda() # NxHxWxC

preds = preds.permute(0,3,1,2)


# print(target.shape, preds.shape)
# #4 classes


loss = CLE.cross_entropy(preds, target)
print(loss)
# # loss.backward()



# l = F.one_hot(target, 4).permute(0,3,1,2)
# print(-1.0*torch.mean(torch.sum(l * torch.log(preds), 1)))
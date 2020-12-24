import copy
import torch
from torch import nn
import torch.nn.functional as F
import config.configure as config

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention(nn.Module):
    r"""
    Self-Attention module, corresponding the global graph.
    Given lots of polyline vectors, each length is 'C', we want to get the predicted feature vector.
    """

    def __init__(self, C, hard=False):
        r"""
        self.linear is 3 linear transformers for Q, K, V.
        :param C: the length of input feature vector.
        """
        super(Attention, self).__init__()
        self.C = C
        # self.linear = clones(nn.Linear(C, C), 3)
        self.klinear = nn.Linear(C, C)
        self.qlinear = nn.Linear(C, C)
        self.vlinear = nn.Linear(C, C)
        self.dropout = nn.Dropout(p=0.1)
        self.hard = hard

    def forward(self, P, M, mask):
        r"""

        :param P: a list of polyline vectors, form a tensor.
                P.shape = [batch size, n, C]
        :param id: index of predicted vector.
                id.shape = [batch size]
        :return: output.
        """

        # batchSize, n, C = P.shape

        # Q = torch.zeros(0, C).to(device)

        # Qt = self.linear[0](P)  # [batch size, n, C]
        # for i in range(P.shape[0]):
        #     # x = id[i].item()
        #     q = Qt[i, 0].unsqueeze(0)  # [1, C]
        #     Q = torch.cat((Q, q), dim=0)
        # Q.unsqueeze_(1)  # Q's shape is # [batch size, 1, C]
        
        

        L, N, V, C = P.shape
        mL, mN, mV, mC = M.shape
        # mN = N
        # mV = mNV // mN
        # M = M.permute(0, 2, 3, 1)
        # M = M.view(mN, mV, mL, mC).contiguous()
        # M = M.view(mN, mV*mT, mC)

        P = P.reshape((-1, V, C))
        M = M.reshape((-1, mV, mC)) 
        Query = self.klinear(P) # (LN, V, C)
        Key = self.qlinear(M)  # (mLN, mV, C)
        Value = self.vlinear(M) # (mLN, mV, C)
        ans = torch.matmul(Query, Key.permute(0, 2, 1))  # [LN, V, mV]

        mask = mask.unsqueeze(1) # [N, V, mV]
        mask = mask.repeat((L,V,1)) # [LN, V, mV]
        zero_tensor = torch.ones_like(ans) * -1e1
        ans = torch.where(mask, ans, zero_tensor)
        ans = F.softmax(ans, dim=-1)

        att = ans # [LN, V, mV]
        att = att.reshape((L, N, V, -1)) # [L, N, V, mV]
        att = torch.mean(torch.mean(att, dim=[0]),dim=[-2]) # (N, V)

        if self.hard:
            argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).repeat((L,1,1)) # [LN, 1, 1]
            ans = torch.zeros_like(ans).scatter(2, argmax_att, 1)


        ans = torch.matmul(ans, Value) #[LN, V, mV]*[LN, mV, C] --> [LN, V, C]
        ans = ans.reshape((L,N*V,-1))
        # ans = ans.squeeze(1).reshape((L,N,C))
        # att = att.squeeze(1)

        return ans, att


        # Q = self.klinear(P) # (N, T, C)
        # K = self.qlinear(M)  # (N, mVT, C)
        # Value = self.vlinear(M) # (N, mVT, C)
        # ans = torch.matmul(Q, K.permute(0, 2, 1))  # (N, T, mVT)

        # ans = ans.view(mN, -1, mV, mT).contiguous()

        # # dk = torch.diag(1/(torch.sum(mask.float(), dim=1)))
        # # ans = torch.einsum('nlvt,nm->mlvt', (ans, dk))

        # mask = mask.unsqueeze(1)
        # mask = mask.unsqueeze(-1)
        # mask = mask.repeat((1,T,1,mT))
        # zero_tensor = torch.ones_like(ans) * -1e15
        # ans = torch.where(mask, ans, zero_tensor)
        # ans = F.softmax(ans, dim=2)

        # att = ans # (N, T, mV, mT)
        # att = torch.mean(att, dim=[1,3]) # (N, mV)

        # ans = ans.view(mN, -1, mV*mT) # (N, T, mVT)

        # if self.hard:
        #     argmax_att = torch.argmax(att, dim=-1).unsqueeze(-1).repeat((T,1,1)) # [N, 1, 1]
        #     ans = torch.zeros_like(ans).scatter(2, argmax_att, 1)


        # ans = torch.matmul(ans, Value)  # (N, T, C)

        # ans = ans.permute(1, 0, 2) # (T, N, C)

        # # att = torch.max(torch.max(att, dim=2)[0], dim=-1)[0]
        # # ans.squeeze_(1)

        # return ans, att


    
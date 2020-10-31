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

    def __init__(self, C):
        r"""
        self.linear is 3 linear transformers for Q, K, V.
        :param C: the length of input feature vector.
        """
        super(Attention, self).__init__()
        self.linear = clones(nn.Linear(C, C), 3)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, P, M, mask=None):
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
        
        
        P = P.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        NV, T, C = P.shape
        N = int(NV / config.max_num_object)
        V = config.max_num_object
        P = P.view(N, V, T, C).contiguous()
        P = P.view(N, V*T, C)
        mNV, mT, mC = M.shape
        mN = int(mNV / config.max_num_map)
        mV = config.max_num_map
        # M = M.permute(0, 2, 3, 1)
        M = M.view(mN, mV, mT, mC).contiguous()
        M = M.view(mN, mT*mV, mC)

        Q = self.linear[0](P) #!!!
        K = self.linear[1](M)  # [batch size, n, C]
        Value = self.linear[2](M)
        ans = torch.matmul(Q, K.permute(0, 2, 1))  # [batch size, 1, n]
        ans = ans.view(N, V*T, mV, mT).contiguous()
        if not mask is None:
            bool_mask = mask.bool()
            a = torch.zeros_like(ans).cuda().float()
            for i in range(ans.shape[0]):
                a[i:i+1,:,bool_mask[i,0,:,0],:] = F.softmax(ans[i:i+1,:,bool_mask[i,0,:,0],:], dim=2)
                argmax_att = torch.argmax(torch.mean(a[i], dim=-1), dim=-1)
            ans = a
        else:
            ans = F.softmax(ans, dim=2)
        ans = ans.view(N, V*T, -1)
        att = ans

        ans = torch.matmul(ans, Value)  # [batch size, 1, C]

        ans = ans.view(N, V, T, -1).contiguous() 
        ans = ans.view(N*V, T, -1)  
        ans = ans.permute(1, 0, 2) 

        att = att.view(N, V, T, mV, mT)
        att = torch.mean(att, dim=[2,4])
        # att = torch.max(torch.max(att, dim=2)[0], dim=-1)[0]
        # ans.squeeze_(1)

        return ans, att


    # def forward(self, P, M, mask=None):
    #     r"""

    #     :param P: a list of polyline vectors, form a tensor.
    #             P.shape = [batch size, n, C]
    #     :param id: index of predicted vector.
    #             id.shape = [batch size]
    #     :return: output.
    #     """

    #     # batchSize, n, C = P.shape

    #     # Q = torch.zeros(0, C).to(device)

    #     # Qt = self.linear[0](P)  # [batch size, n, C]
    #     # for i in range(P.shape[0]):
    #     #     # x = id[i].item()
    #     #     q = Qt[i, 0].unsqueeze(0)  # [1, C]
    #     #     Q = torch.cat((Q, q), dim=0)
    #     # Q.unsqueeze_(1)  # Q's shape is # [batch size, 1, C]
        
        
    #     P = P.permute(0, 3, 2, 1).contiguous()
    #     M = M.permute(0, 3, 2, 1).contiguous()
    #     N, V, T, C = P.shape
    #     P = P.view(N, V*T, C)

    #     mN, mV, mT, mC = M.shape
    #     M = M.view(mN, mV*mT, mC)

    #     Q = self.linear[0](P) #!!!
    #     K = self.linear[1](M)  # [batch size, n, C]
    #     Value = self.linear[2](M)
    #     ans = torch.matmul(Q, K.permute(0, 2, 1))  # [batch size, 1, n]
    #     ans = ans.view(N, V*T, mV, mT).contiguous()
    #     if not mask is None:
    #         bool_mask = mask.bool()
    #         a = torch.zeros_like(ans).cuda().float()
    #         for i in range(ans.shape[0]):
    #             a[i:i+1,:,bool_mask[i,0,:,0],:] = F.softmax(ans[i:i+1,:,bool_mask[i,0,:,0],:], dim=2)
    #             argmax_att = torch.argmax(torch.mean(a[i], dim=-1), dim=-1)
    #         # ans = a
    #     else:
    #         ans = F.softmax(ans, dim=2)
    #     ans = ans.view(N, V*T, -1)
    #     att = ans

    #     ans = torch.matmul(ans, Value)  # [batch size, 1, C]

    #     ans = ans.view(N, V, T, -1)
    #     ans = ans.permute(0, 3, 2, 1) 
    #     # ans.squeeze_(1)

    #     return ans, att

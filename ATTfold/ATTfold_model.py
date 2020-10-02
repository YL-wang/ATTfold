import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ATTfold.common.utils import soft_sign
from scipy.sparse import diags

class ATTfold(nn.Module):

    def __init__(self, d, L, steps):
        super(ATTfold, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d = d
        self.L = L

        # input:one-hot sequence
        self.conv1d1 = nn.Conv1d(in_channels=4, out_channels=d,
                                 kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)

        # position_embedding process
        self.position_embedding_1d = nn.Parameter(
            torch.randn(1, d, 512)
        )
        self.PE_net = nn.Sequential(
            nn.Linear(111, 5 * d),
            nn.ReLU(),
            nn.Linear(5 * d, 5 * d),
            nn.ReLU(),
            nn.Linear(5 * d, d))
        # transformer-encoder
        self.encoder_layer = nn.TransformerEncoderLayer(2 * d, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)

        # CNN-decoder
        self.conv_test_1 = nn.Conv2d(in_channels=6 * d, out_channels=d, kernel_size=1)
        self.bn_conv_1 = nn.BatchNorm2d(d)
        self.conv_test_2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1)
        self.bn_conv_2 = nn.BatchNorm2d(d)
        self.conv_test_3 = nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1)


        # The parameters of constraint_3
        self.steps = steps
        self.s = nn.Parameter(torch.Tensor([math.log(10)]))
        self.w = nn.Parameter(torch.randn(1))
        self.rho_m = nn.Parameter(torch.randn(512, 512))

        self.alpha = nn.Parameter(torch.Tensor([0.005]))
        self.beta = nn.Parameter(torch.Tensor([0.05]))
        self.lr_decay_alpha = nn.Parameter(torch.Tensor([0.99]))
        self.lr_decay_beta = nn.Parameter(torch.Tensor([0.99]))


    def forward(self, pe, seq):

        P_t_list = list()
        x = seq
        position_embeds = self.PE_net(pe.view(-1, 111)).view(-1, self.L, self.d)  # N*L*111 -> b*L*d
        position_embeds = position_embeds.permute(0, 2, 1)  # b*d*L
        seq = seq.permute(0, 2, 1)  # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))  # d*L just for increase the capacity

        seq = torch.cat([seq, position_embeds], 1)  # 2d*L
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))# max_len,batch,2d
        seq = seq.permute(1, 2, 0)# batch, 2d, max_len

        seq_mat = self.matrix_rep(seq)  # （batch,4d,L,L)

        p_mat = self.matrix_rep(position_embeds)  # (batch,2d,L,L)

        encoding = torch.cat([seq_mat, p_mat], 1)  # (batch,6d,l,l)

        score_mat = F.relu(self.bn_conv_1(self.conv_test_1(encoding)))#(b,d,l,l)
        score_mat = F.relu(self.bn_conv_2(self.conv_test_2(score_mat)))#(b,d,l,l)
        score_mat = self.conv_test_3(score_mat)#(b,1,l,l)

        score_mat = score_mat.view(-1, self.L, self.L) #(b,l,l)
        score_mat = (score_mat + torch.transpose(score_mat, -1, -2)) / 2 #(b,l,l)
        score_mat = score_mat.view(-1, self.L, self.L)

        m = self.constraint_matrix_batch(x)  # N*L*L constrain

        S = torch.sigmoid(score_mat - self.s) * score_mat

        # initialization
        P_hat_tmp = (torch.sigmoid(S)) * torch.sigmoid(S - self.s).detach()
        P_tmp = self.contact_a(P_hat_tmp, m)
        M_tmp = self.w * F.relu(torch.sum(P_tmp, dim=-1) - 1).detach()

        P_t_list.append(P_tmp)
        # gradient descent
        for t in range(self.steps):
            M_updated, P_updated, P_hat_updated = self.update_rule(
                S, m, M_tmp, P_tmp, P_hat_tmp, t)

            P_hat_tmp = P_hat_updated
            P_tmp = P_updated
            M_tmp = M_updated

            P_t_list.append(P_tmp)
        return score_mat, P_t_list[-1]

    def update_rule(self, S, m, M, P, P_hat, t):

        grad_P = - S / 2 + (M * soft_sign(torch.sum(P,
                                                       dim=-1) - 1)).unsqueeze_(-1).expand(S.shape)
        grad = P_hat * m * (grad_P + torch.transpose(grad_P, -1, -2))

        P_hat_updated = P_hat - self.alpha * torch.pow(self.lr_decay_alpha,
                                                       t) * grad

        P_hat_updated = F.relu(
            torch.abs(P_hat_updated) - self.rho_m * self.alpha * torch.pow(self.lr_decay_alpha, t))

        P_hat_updated = torch.clamp(P_hat_updated, -1, 1)
        P_updated = self.contact_a(P_hat_updated, m)

        grad_M = F.relu(torch.sum(P_updated, dim=-1) - 1)
        M_updated = M + self.beta * torch.pow(self.lr_decay_beta,
                                                    t) * grad_M

        return M_updated, P_updated, P_hat_updated

    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1)  # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L, 1, 1)
        x2 = x2.repeat(1, 1, L, 1)
        mat = torch.cat([x, x2], -1)  # L*L*2d

        mat_tril = torch.tril(mat.permute(0, -1, 1, 2))  # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag #（batch,4d,L,L)
        return mat


#constraint_1 and constraint_2
    def constraint_matrix_batch(self, x):
        base_a = x[:, :, 0]
        base_u = x[:, :, 1]
        base_c = x[:, :, 2]
        base_g = x[:, :, 3]
        batch = base_a.shape[0]
        length = base_a.shape[1]
        au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
        au_ua = au + torch.transpose(au, -1, -2)
        cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
        cg_gc = cg + torch.transpose(cg, -1, -2)
        ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
        ug_gu = ug + torch.transpose(ug, -1, -2)
        m = au_ua + cg_gc + ug_gu

        mask = diags([1] * 7, [-3, -2, -1, 0, 1, 2, 3],
                     shape=(m.shape[-2], m.shape[-1])).toarray()
        m = m.masked_fill(torch.Tensor(mask).bool().to(self.device), 0)
        return m

    def contact_a(self, a_hat, m):
        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m
        return a





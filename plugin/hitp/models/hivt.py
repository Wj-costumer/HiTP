# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import LaplaceNLLLoss
from ..losses import SoftTargetCrossEntropyLoss
from ..metrics import ADE
from ..metrics import FDE
from ..metrics import MR
from .global_interactor import GlobalInteractor
from .local_encoder import LocalEncoder
from .decoder import MLPDecoder
from ..utils import TemporalData
from .predictor_decoder import Decoder, DecoderResCat

class HiVT(nn.Module):
    def __init__(self,
                 future_steps: int, # 预测未来帧数 default=12
                 num_modes: int, # 预测多模态轨迹数
                 rotate: bool, # 是否旋转到中心agent所在坐标系 
                 node_dim: int, # default = 128 和embedding维度一样 query_feat
                 edge_dim: int, # default = 128 query_pos
                 embed_dim: int, # default = 128 
                 num_heads: int, # default = 8
                 dropout: float,
                 num_temporal_layers: int, # default = 3
                 num_global_layers: int,
                 local_radius: float, # default = 5.0
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 decoder=dict,
                 global_flag=False,
                 local_flag=True,
                 map_flag=False
                 ) -> None:
        super(HiVT, self).__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.global_flag = global_flag
        self.local_flag = local_flag
        self.map_flag = map_flag
        
        self.local_encoder = LocalEncoder(node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
#         self.decoder = MLPDecoder(local_channels=embed_dim,
#                                   global_channels=embed_dim,
#                                   future_steps=future_steps,
#                                   num_modes=num_modes,
#                                   uncertain=True)
        self.decoder = Decoder(**decoder)
    
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        self.minADE = None#ADE()
        self.minFDE = None#FDE()
        self.minMR = None#MR()

    def forward(self, data: TemporalData, actions_list: list, actions_is_valid_list: list, **kwargs):
        if data.x.shape[0] == 1:
            return self.decoder(batch_size=1, 
                            inputs=None, 
                            inputs_lengths=None, 
                            hidden_states=data.x.unsqueeze(0)[:,:,:128], # 1 * 1 * 128
                            device=data.x.device,
                            labels=data.y, 
                            labels_is_valid=data.padding_mask, 
                            actions=actions_list,
                            actions_is_valid=actions_is_valid_list,
                            agents=data.x,
                            agents_indices=None, 
                            **kwargs)
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        if self.local_flag:
            local_embed = self.local_encoder(data=data)
        else:
            local_embed = data.x[:, :128]
            
        if self.global_flag:
            global_embed = self.global_interactor(data=data, local_embed=local_embed) # mode_num * agent_num * embed_dim
        else:
            global_embed = local_embed.unsqueeze(0) # agent_num * embed_dim -> 1 * agent_num * embed_dim
        return self.decoder(batch_size=1, 
                            inputs=None, 
                            inputs_lengths=None, 
                            hidden_states=global_embed, 
                            device=global_embed.device,
                            labels=data.y, 
                            labels_is_valid=data.padding_mask, 
                            actions=actions_list,
                            actions_is_valid=actions_is_valid_list,
                            agents=local_embed,
                            agents_indices=None, 
                            **kwargs)

    def training_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser

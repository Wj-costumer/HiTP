import random
from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from .predictor_lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP
from .. import utils as utils


class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states, **kwargs):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class TrajectoryTransformerDecoder(nn.Module):
    def __init__(self, feature_dim, output_dim, num_heads=8, num_decoder_layers=3, num_trajectories=6, num_timesteps=12, dim_feedforward=512):
        super(TrajectoryTransformerDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_trajectories = num_trajectories
        self.num_timesteps = num_timesteps
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        self.positional_encoding = PositionalEncoding(feature_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_projection = nn.Linear(feature_dim, output_dim) # 输出维度为num_trajectories*num_timesteps*2 + 6

    def forward(self, src):
        # src: (bs, n, 128)
        bs, n, _ = src.size()
        src = src.permute(1, 0, 2)  # 调整为(n, bs, 128)以符合PyTorch的Transformer期望的输入维度
        src = self.positional_encoding(src)

        memory = torch.zeros((n, bs, self.feature_dim)).to(src.device)
        output = self.transformer_decoder(src, memory)
        output = self.output_projection(output)

        return output.permute(1, 0, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register as buffer to avoid being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Decoder(nn.Module):
    def __init__(self,
                 variety_loss=False,
                 variety_loss_prob=False,
                 nms_threshold=3.0,
                 goals_2D=False,
                 hidden_size=128,
                 future_frame_num=12,
                 mode_num=6,
                 cls_num=6,
                 do_eval=False,
                 train_pred_probs_only=False,
                 K_is_1=False,
                 K_is_1_constant=False,
                 exclude_static=False,
                 reduce_prob_of=None,
                 rebalance_prob=False,
                 use_trans=False
                 ):
        super(Decoder, self).__init__()

        self.variety_loss = variety_loss
        self.variety_loss_prob = variety_loss_prob

        self.nms_threshold = nms_threshold
        self.goals_2D = goals_2D
        self.future_frame_num = future_frame_num
        self.mode_num = mode_num
        self.cls_num = cls_num
        self.do_eval = do_eval
        self.train_pred_probs_only = train_pred_probs_only
        self.train_pred_probs_only_values = 270.0 / np.array([1600, 340, 270, 800, 1680, 12000], dtype=float)
        if use_trans:
            self.decoder = TrajectoryTransformerDecoder
        else:
            self.decoder = DecoderResCat
        
        if self.variety_loss: 
            
            self.action_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=self.future_frame_num * self.cls_num)

            if variety_loss_prob:
                if self.train_pred_probs_only:
                    self.train_pred_probs_only_decocer = self.decoder(hidden_size, hidden_size, out_features=6)
                out_features = 6 * self.future_frame_num * 2 + 6
            else:
                out_features = 6 * self.future_frame_num * 2
            if use_trans:
                self.variety_loss_decoder = self.decoder(feature_dim=hidden_size, output_dim=out_features)
            else:
                self.variety_loss_decoder = self.decoder(hidden_size, hidden_size, out_features=out_features)
        else:
            assert False

        self.K_is_1 = K_is_1
        self.K_is_1_constant = K_is_1_constant
        self.exclude_static = exclude_static
        self.reduce_prob_of = reduce_prob_of
        self.rebalance_prob = rebalance_prob

    def forward_variety_loss(self, hidden_states: Tensor, # torch.Size([1, 6, 11, 128])
                             batch_size, inputs: Tensor,
                             inputs_lengths: List[int], 
                             labels_is_valid: List[np.ndarray],
                             actions_is_valid:List[np.ndarray], 
                             loss: Tensor,
                             loss_a: Tensor,
                             DE: np.ndarray, 
                             device, 
                             labels: List[np.ndarray], 
                             actions: List[np.ndarray], 
                             agents: Tensor,
                             agents_indices=None):
        """
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        assert batch_size == 1
        agent_num = len(agents_indices) if agents_indices is not None else agents[0].shape[0]
        assert agent_num == labels[0].shape[0]
        if True:
            if agents_indices is not None:
                agents_indices = torch.tensor(agents_indices, dtype=torch.long, device=device)
                hidden_states = hidden_states[:, agents_indices, :]
            else:
                hidden_states = hidden_states[:, :agent_num, :]
            
            for idx in range(batch_size):
                action = actions[idx]
                rows_to_modify = np.any(action == -1, axis=1)
                action[action > 1] -= 2
                action[action < 2] = 0
                actions_is_valid[idx][rows_to_modify] = 0 # label<1标记无效
            
            # print('debug: ', actions, actions_is_valid)
            actions_one_hot = F.one_hot(torch.tensor(actions).to(torch.int64), num_classes=self.cls_num) # BS * N * future_step -> BS * N * future_step * cls_num
            try:
                outputs = self.variety_loss_decoder(hidden_states[:, -1, :, :]) # [bs, agent_num, 150]
                outputs_a = self.action_loss_decoder(hidden_states[:, -1, :, :]) # [bs, agent_num, 96]
            except:
                breakpoint()
            if True:
                if self.train_pred_probs_only:
                    pred_probs = F.log_softmax(self.train_pred_probs_only_decocer(hidden_states), dim=-1)
                else:
                    pred_probs = F.log_softmax(outputs[:, :, -6:], dim=-1)
                outputs = outputs[:, :, :-6].view([batch_size, agent_num, 6, self.future_frame_num, 2]) # torch.Size([1, 16, 6, 12, 2])
            else:
                outputs = outputs.view([batch_size, agent_num, 6, self.future_frame_num, 2])
            outputs_a = outputs_a.view([batch_size, agent_num, self.future_frame_num, self.cls_num]) # torch.Size([1, 16, 12, ])
        
        CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none') # initiatize before refer to
        pred_actions = []
        for i in range(batch_size):
            if True:
                if self.rebalance_prob:
                    if not hasattr(self, 'rebalance_prob_values'):
                        self.rebalance_prob_values = np.zeros(6, dtype=int)

                valid_agent_num = 0
                for agent_idx in range(agent_num):
                    should_train = True

                    gt_points = np.array(labels[i][agent_idx]).reshape([self.future_frame_num, 2])
                    last_valid_index = -1
                
                    for j in range(self.future_frame_num)[::-1]:
                        if labels_is_valid[i][agent_idx, j]:
                            last_valid_index = j
                            break
                        
                    argmin = np.argmin(utils.get_dis_point_2_points(gt_points[last_valid_index], utils.to_numpy(outputs[i, agent_idx, :, last_valid_index, :])))

                    # argmin = utils.argmin_traj(labels[i][agent_idx], labels_is_valid[i][agent_idx], utils.to_numpy(outputs[i, agent_idx]))
                    loss_ = F.smooth_l1_loss(outputs[i, agent_idx, argmin],
                                             torch.tensor(labels[i][agent_idx], device=device, dtype=torch.float), reduction='none') # (f_steps, 2)
                    outputs_a_softmax = F.softmax(outputs_a[i, agent_idx, :, :]) # fut_steps * num_cls
                    pred_actions_ = torch.argmax(outputs_a_softmax, dim=-1).cpu().numpy() # convert to pred action labels
                    pred_actions.append(pred_actions_)
                    loss_action = CrossEntropyLoss(outputs_a_softmax, actions_one_hot[i][agent_idx].to(device, dtype=torch.float)) # (f_steps, 1) 
                    if self.rebalance_prob:
                        self.rebalance_prob_values[argmin] += 1

                    if self.train_pred_probs_only:
                        if np.random.rand() > self.train_pred_probs_only_values[argmin]:
                            should_train = False


                    if self.reduce_prob_of is not None:
                        if np.random.randint(0, 50) == 0:
                            torch.set_printoptions(precision=3, sci_mode=False)
                            print('pred_probs[i, agent_idx]', pred_probs[i, agent_idx].exp())
                            print(outputs[i, agent_idx, :, -1])
                        pred_probs[i, agent_idx] = pred_probs[i, agent_idx].exp()
                        pred_probs[i, agent_idx, 5] *= 0.1
                        pred_probs[i, agent_idx, 1] *= 3.0
                        pred_probs[i, agent_idx, 2] *= 3.0
                        pred_probs[i, agent_idx] = pred_probs[i, agent_idx].log()
                        # past_boxes_list = mapping[0]['past_boxes_list']

                    loss_ = loss_ * torch.tensor(labels_is_valid[i][agent_idx], device=device, dtype=torch.float).view(self.future_frame_num, 1)
                    loss_action = loss_action * torch.tensor(actions_is_valid[i][agent_idx], device=device, dtype=torch.float)
                    if actions_is_valid[i][agent_idx].sum() > utils.eps and should_train:
                        loss_a[i] += loss_action.sum() / actions_is_valid[i][agent_idx].sum()
                    if labels_is_valid[i][agent_idx].sum() > utils.eps and should_train:
                        loss[i] += loss_.sum() / labels_is_valid[i][agent_idx].sum()
                        loss[i] += F.nll_loss(pred_probs[i, agent_idx].unsqueeze(0), torch.tensor([argmin], device=device))

                        valid_agent_num += 1

                if valid_agent_num > 0:
                    loss[i] = loss[i] / valid_agent_num
                    loss_a[i] = loss_a[i] / valid_agent_num
                    
                if self.rebalance_prob:
                    print(self.rebalance_prob_values)
        results = dict(
            pred_outputs=utils.to_numpy(outputs),
            pred_probs=utils.to_numpy(pred_probs),
            pred_actions=np.array(pred_actions),
            gt_actions= actions[0],
            actions_is_valid=actions_is_valid[0]
        )
   
        return loss.mean(), loss_a.mean(), results, None

    def forward(self,
                # mapping: List[Dict],
                batch_size,
                inputs: Tensor,
                inputs_lengths: List[int],
                hidden_states: Tensor,
                device,
                labels,
                labels_is_valid,
                actions,
                actions_is_valid,
                agents,
                agents_indices=None,
                **kwargs,
                ):
        """
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        """
        loss = torch.zeros(batch_size, device=device)
        loss_a = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_frame_num])
        if self.variety_loss:
            if self.rebalance_prob:
                with torch.no_grad():
                    return self.forward_variety_loss(hidden_states.unsqueeze(0), batch_size, inputs, inputs_lengths, labels_is_valid, actions_is_valid, loss, loss_a, DE, device, labels, actions, 
                                                    agents=[agents], agents_indices=agents_indices)
            else:
                return self.forward_variety_loss(hidden_states.unsqueeze(0), batch_size, inputs, inputs_lengths, labels_is_valid, actions_is_valid, loss, loss_a, DE, device, labels, actions, 
                                                 agents=[agents], agents_indices=agents_indices)
        else:
            assert False






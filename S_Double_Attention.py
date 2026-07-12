## -*- coding: utf-8 -*-
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.nn.functional import scaled_dot_product_attention
#
#
#class StrokeCrossRegionAttention(nn.Module):
#    """SSRA: True multi-head attention (independent Q/K/V projection per head)"""
#
#    def __init__(self, embed_dim, num_heads=8):
#        super(StrokeCrossRegionAttention, self).__init__()
#        self.embed_dim = embed_dim
#        self.num_heads = num_heads
#        self.head_dim = embed_dim // num_heads
#        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#
#        # Independent Q/K/V projection layers for each head
#        self.cross_q_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.non_cross_k_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.original_v_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#    def forward(self, non_cross_feat, cross_feat, original_feat):
#        B, C, H, W = non_cross_feat.shape
#
#        # Generate Q/K/V for all heads
#        Q_list = [q_head(cross_feat) for q_head in self.cross_q_heads]
#        Q = torch.cat(Q_list, dim=1)
#
#        K_list = [k_head(non_cross_feat) for k_head in self.non_cross_k_heads]
#        K = torch.cat(K_list, dim=1)
#
#        V_list = [v_head(original_feat) for v_head in self.original_v_heads]
#        V = torch.cat(V_list, dim=1)
#
#        # Reshape for multi-head attention
#        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W)
#        K = K.reshape(B, self.num_heads, self.head_dim, H, W)
#        V = V.reshape(B, self.num_heads, self.head_dim, H, W)
#
#        Q = Q.flatten(3).transpose(2, 3)
#        K = K.flatten(3).transpose(2, 3)
#        V = V.flatten(3).transpose(2, 3)
#
#        # Scaled dot product attention
#        attn_output = scaled_dot_product_attention(
#            query=Q,
#            key=K,
#            value=V,
#            attn_mask=None,
#            dropout_p=0.2
#        )
#
#        # Reshape back
#        attn_output = attn_output.transpose(2, 3).contiguous()
#        attn_output = attn_output.reshape(B, C, H, W)
#
#        return attn_output
#
#
#class StrokeSegmentComponentAttention(nn.Module):
#    """SSCA: True multi-head attention (independent Q/K/V projection per head)"""
#
#    def __init__(self, embed_dim, num_heads=8):
#        super(StrokeSegmentComponentAttention, self).__init__()
#        self.embed_dim = embed_dim
#        self.num_heads = num_heads
#        self.head_dim = embed_dim // num_heads
#        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#
#        # Independent Q/K/V projection layers for each head
#        self.non_cross_q_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.non_cross_k_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.ssra_v_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#    def forward(self, non_cross_feat, original_feat, ssra_feat):
#        B, C, H, W = non_cross_feat.shape
#
#        # Generate Q/K/V for all heads
#        Q_list = [q_head(original_feat) for q_head in self.non_cross_q_heads]
#        Q = torch.cat(Q_list, dim=1)
#
#        K_list = [k_head(original_feat) for k_head in self.non_cross_k_heads]
#        K = torch.cat(K_list, dim=1)
#
#        V_list = [v_head(ssra_feat) for v_head in self.ssra_v_heads]
#        V = torch.cat(V_list, dim=1)
#
#        # Reshape for multi-head attention
#        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W)
#        K = K.reshape(B, self.num_heads, self.head_dim, H, W)
#        V = V.reshape(B, self.num_heads, self.head_dim, H, W)
#
#        Q = Q.flatten(3).transpose(2, 3)
#        K = K.flatten(3).transpose(2, 3)
#        V = V.flatten(3).transpose(2, 3)
#
#        # Scaled dot product attention
#        attn_output = scaled_dot_product_attention(
#            query=Q,
#            key=K,
#            value=V,
#            attn_mask=None,
#            dropout_p=0.2
#        )
#
#        # Reshape back
#        attn_output = attn_output.transpose(2, 3).contiguous()
#        attn_output = attn_output.reshape(B, C, H, W)
#
#        return attn_output
#
#
#class SDoubleAttention(nn.Module):
#    """Double Attention Module (aligned with Transformer multi-head logic)"""
#
#    def __init__(self, input_dim, embed_dim, num_heads=8):
#        super(SDoubleAttention, self).__init__()
#        self.channel_adjust = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
#        self.ssra = StrokeCrossRegionAttention(embed_dim, num_heads)
#        self.ssca = StrokeSegmentComponentAttention(embed_dim, num_heads)
#
#    def forward(self, cross_feat, non_cross_feat, original_feat):
#        # Unify channel dimension to embed_dim
#        cross_feat = self.channel_adjust(cross_feat)
#        non_cross_feat = self.channel_adjust(non_cross_feat)
#        original_feat = self.channel_adjust(original_feat)
#
#        # Double attention process
#        ssra_out = self.ssra(non_cross_feat, cross_feat, original_feat)
#        ssca_out = self.ssca(non_cross_feat, original_feat, ssra_out)
#
#        return ssca_out# -*- coding: utf-8 -*-
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.nn.functional import scaled_dot_product_attention
#
#
#class StrokeCrossRegionAttention(nn.Module):
#    """SSRA: True multi-head attention (independent Q/K/V projection per head)"""
#
#    def __init__(self, embed_dim, num_heads=8):
#        super(StrokeCrossRegionAttention, self).__init__()
#        self.embed_dim = embed_dim
#        self.num_heads = num_heads
#        self.head_dim = embed_dim // num_heads
#        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#
#        # Independent Q/K/V projection layers for each head
#        self.cross_q_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.non_cross_k_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.original_v_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#    def forward(self, non_cross_feat, cross_feat, original_feat):
#        B, C, H, W = non_cross_feat.shape
#
#        # Generate Q/K/V for all heads
#        Q_list = [q_head(cross_feat) for q_head in self.cross_q_heads]
#        Q = torch.cat(Q_list, dim=1)
#
#        K_list = [k_head(non_cross_feat) for k_head in self.non_cross_k_heads]
#        K = torch.cat(K_list, dim=1)
#
#        V_list = [v_head(original_feat) for v_head in self.original_v_heads]
#        V = torch.cat(V_list, dim=1)
#
#        # Reshape for multi-head attention
#        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W)
#        K = K.reshape(B, self.num_heads, self.head_dim, H, W)
#        V = V.reshape(B, self.num_heads, self.head_dim, H, W)
#
#        Q = Q.flatten(3).transpose(2, 3)
#        K = K.flatten(3).transpose(2, 3)
#        V = V.flatten(3).transpose(2, 3)
#
#        # Scaled dot product attention
#        attn_output = scaled_dot_product_attention(
#            query=Q,
#            key=K,
#            value=V,
#            attn_mask=None,
#            dropout_p=0.2
#        )
#
#        # Reshape back
#        attn_output = attn_output.transpose(2, 3).contiguous()
#        attn_output = attn_output.reshape(B, C, H, W)
#
#        return attn_output
#
#
#class StrokeSegmentComponentAttention(nn.Module):
#    """SSCA: True multi-head attention (independent Q/K/V projection per head)"""
#
#    def __init__(self, embed_dim, num_heads=8):
#        super(StrokeSegmentComponentAttention, self).__init__()
#        self.embed_dim = embed_dim
#        self.num_heads = num_heads
#        self.head_dim = embed_dim // num_heads
#        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#
#        # Independent Q/K/V projection layers for each head
#        self.non_cross_q_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.non_cross_k_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#        self.ssra_v_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.BatchNorm2d(embed_dim),
#                nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, kernel_size=1),
#                nn.Dropout2d(0.2)
#            ) for _ in range(num_heads)
#        ])
#
#    def forward(self, non_cross_feat, original_feat, ssra_feat):
#        B, C, H, W = non_cross_feat.shape
#
#        # Generate Q/K/V for all heads
#        Q_list = [q_head(original_feat) for q_head in self.non_cross_q_heads]
#        Q = torch.cat(Q_list, dim=1)
#
#        K_list = [k_head(original_feat) for k_head in self.non_cross_k_heads]
#        K = torch.cat(K_list, dim=1)
#
#        V_list = [v_head(ssra_feat) for v_head in self.ssra_v_heads]
#        V = torch.cat(V_list, dim=1)
#
#        # Reshape for multi-head attention
#        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W)
#        K = K.reshape(B, self.num_heads, self.head_dim, H, W)
#        V = V.reshape(B, self.num_heads, self.head_dim, H, W)
#
#        Q = Q.flatten(3).transpose(2, 3)
#        K = K.flatten(3).transpose(2, 3)
#        V = V.flatten(3).transpose(2, 3)
#
#        # Scaled dot product attention
#        attn_output = scaled_dot_product_attention(
#            query=Q,
#            key=K,
#            value=V,
#            attn_mask=None,
#            dropout_p=0.2
#        )
#
#        # Reshape back
#        attn_output = attn_output.transpose(2, 3).contiguous()
#        attn_output = attn_output.reshape(B, C, H, W)
#
#        return attn_output
#
#
#class SDoubleAttention(nn.Module):
#    """Double Attention Module (aligned with Transformer multi-head logic)"""
#
#    def __init__(self, input_dim, embed_dim, num_heads=8):
#        super(SDoubleAttention, self).__init__()
#        self.channel_adjust = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
#        self.ssra = StrokeCrossRegionAttention(embed_dim, num_heads)
#        self.ssca = StrokeSegmentComponentAttention(embed_dim, num_heads)
#
#    def forward(self, cross_feat, non_cross_feat, original_feat):
#        # Unify channel dimension to embed_dim
#        cross_feat = self.channel_adjust(cross_feat)
#        non_cross_feat = self.channel_adjust(non_cross_feat)
#        original_feat = self.channel_adjust(original_feat)
#
#        # Double attention process
#        ssra_out = self.ssra(non_cross_feat, cross_feat, original_feat)
#        ssca_out = self.ssca(non_cross_feat, original_feat, ssra_out)
#
#        return ssca_out


















## -*- coding: utf-8 -*-
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#
#class StrokeCrossRegionAttention(nn.Module):
#    def __init__(self, embed_dim, num_heads=8, spatial_size=(8,8)):
#        super().__init__()
#        self.embed_dim = embed_dim
#        self.num_heads = num_heads
#        self.head_dim = embed_dim // num_heads
#        assert self.head_dim * num_heads == embed_dim
#
#        self.cross_q_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, 1),
#                nn.Dropout2d(0.05)
#            ) for _ in range(num_heads)
#        ])
#        self.non_cross_k_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, 1),
#                nn.Dropout2d(0.05)
#            ) for _ in range(num_heads)
#        ])
#        self.original_v_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
#                nn.Conv2d(embed_dim, self.head_dim, 1),
#                nn.Dropout2d(0.05)
#            ) for _ in range(num_heads)
#        ])
#
#        self.max_dist = max(spatial_size)
#        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2*self.max_dist-1, 2*self.max_dist-1))
#        nn.init.normal_(self.rel_pos_bias, std=0.02)
#
#    def _get_relative_position_bias(self, H, W, device):
#        grid_h = torch.arange(H, device=device)
#        grid_w = torch.arange(W, device=device)
#        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
#        pos = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)
#        rel_diff = pos.unsqueeze(1) - pos.unsqueeze(0)
#        rel_h = rel_diff[...,0] + self.max_dist -1
#        rel_w = rel_diff[...,1] + self.max_dist -1
#        rel_h = rel_h.clamp(0, 2*self.max_dist-2)
#        rel_w = rel_w.clamp(0, 2*self.max_dist-2)
#        bias = self.rel_pos_bias[:, rel_h.long(), rel_w.long()]
#        return bias.permute(1,2,0)
#
#    def forward(self, non_cross_feat, cross_feat, original_feat):
#        B, C, H, W = non_cross_feat.shape
#        Q = torch.cat([q(cross_feat) for q in self.cross_q_heads], dim=1)
#        K = torch.cat([k(non_cross_feat) for k in self.non_cross_k_heads], dim=1)
#        V = torch.cat([v(original_feat) for v in self.original_v_heads], dim=1)
#
#        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2,3)
#        K = K.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2,3)
#        V = V.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2,3)
#
#        attn_score = torch.matmul(Q, K.transpose(-2,-1)) * (self.head_dim ** -0.5)
#        attn_score += self._get_relative_position_bias(H,W,Q.device).permute(2,0,1).unsqueeze(0)
#        attn_weight = F.softmax(attn_score, dim=-1)
#        attn_weight = F.dropout(attn_weight, 0.05, training=self.training)
#        out = torch.matmul(attn_weight, V)
#
#        return out.transpose(2,3).reshape(B,C,H,W)
#
#
#class StrokeSegmentComponentAttention(nn.Module):
#    def __init__(self, embed_dim, num_heads=8, spatial_size=(8,8)):
#        super().__init__()
#        self.embed_dim = embed_dim
#        self.num_heads = num_heads
#        self.head_dim = embed_dim // num_heads
#
#        self.non_cross_q_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim,3,padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim,3,padding=1),
#                nn.Conv2d(embed_dim, self.head_dim,1),
#                nn.Dropout2d(0.05)
#            ) for _ in range(num_heads)
#        ])
#        self.non_cross_k_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim,3,padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim,3,padding=1),
#                nn.Conv2d(embed_dim, self.head_dim,1),
#                nn.Dropout2d(0.05)
#            ) for _ in range(num_heads)
#        ])
#        self.ssra_v_heads = nn.ModuleList([
#            nn.Sequential(
#                nn.Conv2d(embed_dim, embed_dim,3,padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
#                nn.Conv2d(embed_dim, embed_dim,3,padding=1),
#                nn.Conv2d(embed_dim, self.head_dim,1),
#                nn.Dropout2d(0.05)
#            ) for _ in range(num_heads)
#        ])
#
#        self.max_dist = max(spatial_size)
#        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2*self.max_dist-1, 2*self.max_dist-1))
#        nn.init.normal_(self.rel_pos_bias, std=0.02)
#
#    def _get_relative_position_bias(self, H, W, device):
#        grid_h = torch.arange(H, device=device)
#        grid_w = torch.arange(W, device=device)
#        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
#        pos = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)
#        rel_diff = pos.unsqueeze(1) - pos.unsqueeze(0)
#        rel_h = rel_diff[...,0] + self.max_dist -1
#        rel_w = rel_diff[...,1] + self.max_dist -1
#        rel_h = rel_h.clamp(0, 2*self.max_dist-2)
#        rel_w = rel_w.clamp(0, 2*self.max_dist-2)
#        bias = self.rel_pos_bias[:, rel_h.long(), rel_w.long()]
#        return bias.permute(1,2,0)
#
#    def forward(self, non_cross_feat, original_feat, ssra_feat):
#        B, C, H, W = non_cross_feat.shape
#        Q = torch.cat([q(original_feat) for q in self.non_cross_q_heads], dim=1)
#        K = torch.cat([k(original_feat) for k in self.non_cross_k_heads], dim=1)
#        V = torch.cat([v(ssra_feat) for v in self.ssra_v_heads], dim=1)
#
#        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2,3)
#        K = K.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2,3)
#        V = V.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2,3)
#
#        attn_score = torch.matmul(Q, K.transpose(-2,-1)) * (self.head_dim ** -0.5)
#        attn_score += self._get_relative_position_bias(H,W,Q.device).permute(2,0,1).unsqueeze(0)
#        attn_weight = F.softmax(attn_score, dim=-1)
#        attn_weight = F.dropout(attn_weight, 0.05, training=self.training)
#        out = torch.matmul(attn_weight, V)
#
#        return out.transpose(2,3).reshape(B,C,H,W)
#
#
#class SDoubleAttention(nn.Module):
#    def __init__(self, input_dim, embed_dim, num_heads=8, spatial_size=(8,8)):
#        super().__init__()
#        self.channel_adjust = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
#        self.ssra = StrokeCrossRegionAttention(embed_dim, num_heads, spatial_size)
#        self.ssca = StrokeSegmentComponentAttention(embed_dim, num_heads, spatial_size)
#
#    def forward(self, cross_feat, non_cross_feat, original_feat):
#        cross_feat = self.channel_adjust(cross_feat)
#        non_cross_feat = self.channel_adjust(non_cross_feat)
#        original_feat = self.channel_adjust(original_feat)
#        ssra_out = self.ssra(non_cross_feat, cross_feat, original_feat)
#        ssca_out = self.ssca(non_cross_feat, original_feat, ssra_out)
#        return ssca_out















# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAwareRelativeBias(nn.Module):
    def __init__(self, num_heads, spatial_size=(8,8), content_channels=512):
        super().__init__()
        self.num_heads = num_heads
        max_dist = max(spatial_size)
        self.max_dist = max_dist
        self.base_bias = nn.Parameter(torch.zeros(num_heads, 2*max_dist-1, 2*max_dist-1))
        nn.init.normal_(self.base_bias, std=0.02)
        self.modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(content_channels, content_channels // 8),
            nn.ReLU(),
            nn.Linear(content_channels // 8, 1),
            nn.Sigmoid()
        )

    def _get_relative_position_bias(self, H, W, device, scale):
  
        grid_h = torch.arange(H, device=device)
        grid_w = torch.arange(W, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        pos = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)
        rel_diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        rel_h = rel_diff[..., 0] + self.max_dist - 1
        rel_w = rel_diff[..., 1] + self.max_dist - 1
        rel_h = rel_h.clamp(0, 2*self.max_dist-2)
        rel_w = rel_w.clamp(0, 2*self.max_dist-2)
      
        base = self.base_bias[:, rel_h.long(), rel_w.long()]
        base = base.unsqueeze(0) 
        return base * scale.view(-1, 1, 1, 1)

    def forward(self, H, W, content_feat, device):
        
        scale = self.modulator(content_feat)  # (B,1)
        bias = self._get_relative_position_bias(H, W, device, scale)
        return bias


class StrokeCrossRegionAttention(nn.Module):
  
    def __init__(self, embed_dim, num_heads=8, spatial_size=(8,8)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.cross_q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.non_cross_k_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.original_v_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])

       
        self.prototype = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)

        self.content_bias = ContentAwareRelativeBias(num_heads, spatial_size, embed_dim)

    def forward(self, non_cross_feat, cross_feat, original_feat):
        B, C, H, W = non_cross_feat.shape

        
        Q = torch.cat([q(cross_feat) for q in self.cross_q_heads], dim=1)
        K = torch.cat([k(non_cross_feat) for k in self.non_cross_k_heads], dim=1)
        V = torch.cat([v(original_feat) for v in self.original_v_heads], dim=1)

        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)
        K = K.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)
        V = V.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)

        proto_bias = torch.einsum('bhld,hd->bhl', Q, self.prototype).unsqueeze(-1)

       
        dynamic_bias = self.content_bias(H, W, cross_feat, Q.device)

        attn_score = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_score = attn_score + dynamic_bias + proto_bias

        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = F.dropout(attn_weight, 0.05, training=self.training)
        out = torch.matmul(attn_weight, V)

        out = out.transpose(2, 3).reshape(B, C, H, W)
        return out


class StrokeSegmentComponentAttention(nn.Module):
  
    def __init__(self, embed_dim, num_heads=8, spatial_size=(8,8)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.non_cross_q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.non_cross_k_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.ssra_v_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])

        self.prototype = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)

        self.content_bias = ContentAwareRelativeBias(num_heads, spatial_size, embed_dim)

    def forward(self, non_cross_feat, original_feat, ssra_feat):
        B, C, H, W = non_cross_feat.shape

        Q = torch.cat([q(original_feat) for q in self.non_cross_q_heads], dim=1)
        K = torch.cat([k(original_feat) for k in self.non_cross_k_heads], dim=1)
        V = torch.cat([v(ssra_feat) for v in self.ssra_v_heads], dim=1)

        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)
        K = K.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)
        V = V.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)

        
        proto_bias = torch.einsum('bhld,hd->bhl', Q, self.prototype).unsqueeze(-1)

        dynamic_bias = self.content_bias(H, W, original_feat, Q.device)

        attn_score = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_score = attn_score + dynamic_bias + proto_bias

        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = F.dropout(attn_weight, 0.05, training=self.training)
        out = torch.matmul(attn_weight, V)

        out = out.transpose(2, 3).reshape(B, C, H, W)
        return out

class SDoubleAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=8, spatial_size=(8,8)):
        super().__init__()
        self.channel_adjust = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
        self.ssra = StrokeCrossRegionAttention(embed_dim, num_heads, spatial_size)
        self.ssca = StrokeSegmentComponentAttention(embed_dim, num_heads, spatial_size)

    def forward(self, cross_feat, non_cross_feat, original_feat):
        cross_feat = self.channel_adjust(cross_feat)
        non_cross_feat = self.channel_adjust(non_cross_feat)
        original_feat = self.channel_adjust(original_feat)
        ssra_out = self.ssra(non_cross_feat, cross_feat, original_feat)
        ssca_out = self.ssca(non_cross_feat, original_feat, ssra_out)
        return ssca_out











# Enhanced PDE Diffusion Neural Network for Face Expression Recognition
# Modified with Time-Dependent Vector Alpha/Beta Coefficients

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- Enhanced PDE Diffusion Layer with Time-Dependent Vector Parameters ---
class DiffusionLayer(nn.Module):
    def __init__(self, size=48, dt=0.01, dx=1.0, dy=1.0, num_steps=10):
        super().__init__()
        self.size = size
        self.dt = dt
        self.dx = dx  # Spatial step in x-direction
        self.dy = dy  # Spatial step in y-direction
        self.num_steps = num_steps

        # Base diffusion coefficients as 1D vectors (learnable)
        # Alpha vector for x-direction diffusion
        self.alpha_base_x = nn.Parameter(torch.ones(size) * 2.0)
        self.alpha_base_y = nn.Parameter(torch.ones(size) * 2.0)
        
        # Beta vector for y-direction diffusion  
        self.beta_base_x = nn.Parameter(torch.ones(size) * 2.0)
        self.beta_base_y = nn.Parameter(torch.ones(size) * 2.0)

        # Time-dependent modulation parameters as vectors (learnable)
        self.alpha_time_coeff_x = nn.Parameter(torch.zeros(size))
        self.alpha_time_coeff_y = nn.Parameter(torch.zeros(size))
        self.beta_time_coeff_x = nn.Parameter(torch.zeros(size))
        self.beta_time_coeff_y = nn.Parameter(torch.zeros(size))

        # Higher-order time dependence (quadratic terms)
        self.alpha_time_quad_x = nn.Parameter(torch.zeros(size))
        self.alpha_time_quad_y = nn.Parameter(torch.zeros(size))
        self.beta_time_quad_x = nn.Parameter(torch.zeros(size))
        self.beta_time_quad_y = nn.Parameter(torch.zeros(size))

        # Stability parameters
        self.stability_eps = 1e-6

        print(f"Initialized DiffusionLayer with vector coefficients:")
        print(f"  - Size: {size}x{size}, dx={dx}, dy={dy}")
        print(f"  - Vector parameters: 4 base + 4 linear + 4 quadratic = 12 vectors")
        print(f"  - Total learnable parameters: {12 * size}")

    def get_alpha_beta_vectors_at_time(self, t):
        """Get alpha and beta coefficient vectors at time t"""
        # Time-dependent vectors with quadratic terms for richer dynamics
        alpha_x_t = (self.alpha_base_x + 
                     self.alpha_time_coeff_x * t + 
                     self.alpha_time_quad_x * t**2)
        alpha_y_t = (self.alpha_base_y + 
                     self.alpha_time_coeff_y * t + 
                     self.alpha_time_quad_y * t**2)
        
        beta_x_t = (self.beta_base_x + 
                    self.beta_time_coeff_x * t + 
                    self.beta_time_quad_x * t**2)
        beta_y_t = (self.beta_base_y + 
                    self.beta_time_coeff_y * t + 
                    self.beta_time_quad_y * t**2)

        # Ensure positive coefficients for stability
        alpha_x_t = torch.clamp(alpha_x_t, min=self.stability_eps)
        alpha_y_t = torch.clamp(alpha_y_t, min=self.stability_eps)
        beta_x_t = torch.clamp(beta_x_t, min=self.stability_eps)
        beta_y_t = torch.clamp(beta_y_t, min=self.stability_eps)

        return alpha_x_t, alpha_y_t, beta_x_t, beta_y_t

    def expand_vector_to_matrix(self, vec_x, vec_y, direction='x'):
        """
        Convert vectors to coefficient matrices for diffusion
        vec_x: coefficients varying along x-direction (size,)
        vec_y: coefficients varying along y-direction (size,)
        direction: 'x' or 'y' to determine how to combine the vectors
        """
        if direction == 'x':
            # For x-direction diffusion: broadcast vec_x along y-axis
            # Each row has the same x-varying coefficients
            matrix = vec_x.unsqueeze(0).expand(self.size, -1)  # (size, size)
        else:  # direction == 'y'
            # For y-direction diffusion: broadcast vec_y along x-axis  
            # Each column has the same y-varying coefficients
            matrix = vec_y.unsqueeze(1).expand(-1, self.size)  # (size, size)
            
        return matrix

    def forward(self, u):
        B, _, H, W = u.shape
        u = u.squeeze(1)

        # Apply multiple diffusion steps with time-dependent vector coefficients
        current_time = 0.0
        for step in range(self.num_steps):
            # Get vector coefficients at current time
            alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = self.get_alpha_beta_vectors_at_time(current_time)
            
            # Strang splitting: half step x, full step y, half step x
            # Convert vectors to matrices for x-direction diffusion
            alpha_matrix_x = self.expand_vector_to_matrix(alpha_x_t, alpha_y_t, direction='x')
            u = self.diffuse_x_vectorized(u, alpha_matrix_x, self.dt / 2, self.dx)
            current_time += self.dt / 2

            # Update coefficients and do y-direction diffusion
            alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = self.get_alpha_beta_vectors_at_time(current_time)
            beta_matrix_y = self.expand_vector_to_matrix(beta_x_t, beta_y_t, direction='y')
            u = self.diffuse_y_vectorized(u, beta_matrix_y, self.dt, self.dy)
            current_time += self.dt / 2

            # Final half step in x-direction
            alpha_x_t, alpha_y_t, beta_x_t, beta_y_t = self.get_alpha_beta_vectors_at_time(current_time)
            alpha_matrix_x = self.expand_vector_to_matrix(alpha_x_t, alpha_y_t, direction='x')
            u = self.diffuse_x_vectorized(u, alpha_matrix_x, self.dt / 2, self.dx)

        return u.unsqueeze(1)

    def diffuse_x_vectorized(self, u, alpha_matrix, dt, dx):
        """
        Vectorized diffusion in x-direction using proper dx spacing
        Solves: ∂u/∂t = ∇·(α∇u) in x-direction
        """
        B, H, W = u.shape
        device = u.device

        # Reshape for batch processing: (B, H, W) -> (B*H, W)
        u_flat = u.contiguous().view(B * H, W)

        # Expand alpha_matrix for all batches: (H, W) -> (B*H, W)
        alpha_expanded = alpha_matrix.unsqueeze(0).expand(B, -1, -1).contiguous().view(B * H, W)

        # Apply smoothing to coefficients for stability
        alpha_smooth = self.smooth_coefficients(alpha_expanded, dim=1)
        coeff = alpha_smooth * dt / (dx ** 2)  # Using dx for x-direction

        # Build tridiagonal system coefficients
        a = -coeff  # sub-diagonal
        c = -coeff  # super-diagonal
        b = 1 + 2 * coeff  # main diagonal

        # Apply boundary conditions (Neumann - no flux at boundaries)
        b_modified = b.clone()
        b_modified[:, 0] = 1 + coeff[:, 0]
        b_modified[:, -1] = 1 + coeff[:, -1]

        # Solve all tridiagonal systems in p

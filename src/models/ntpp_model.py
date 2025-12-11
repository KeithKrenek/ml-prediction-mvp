"""
Neural Temporal Point Process (NTPP) Model

Implements a recurrent neural network-based temporal point process
for predicting timing of discrete events (Trump posts).

Unlike Prophet which models periodicity, NTPP models the conditional
intensity function λ(t | history) - the instantaneous rate of events
given past history. This is ideal for high-frequency, bursty posting
behavior (20+ posts/day).

Architecture:
- LSTM encoder for event history
- Neural intensity function λ(t)
- Monotonic network ensures λ(t) properties
- Trained with negative log-likelihood loss
- Monte Carlo dropout for uncertainty quantification
- Thinning algorithm for proper sampling from intensity

References:
- "Neural Temporal Point Processes" (Mei & Eisner, 2017)
- "Transformer Hawkes Process" (Zuo et al., 2020)
- "Ogata's Thinning Algorithm" for sampling from TPPs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from loguru import logger
import pickle
from pathlib import Path
from dataclasses import dataclass


@dataclass
class NTPPPrediction:
    """
    Structured prediction result with uncertainty quantification.
    """
    predicted_time: datetime
    predicted_hours_ahead: float
    confidence: float
    
    # Distribution statistics
    mean_hours: float
    median_hours: float
    std_hours: float
    
    # Confidence intervals
    ci_lower_hours: float  # 5th percentile
    ci_upper_hours: float  # 95th percentile
    
    # Additional metadata
    n_samples: int
    model_version: str
    trained_at: Optional[datetime]
    
    # Survival probability at different horizons
    prob_within_1h: float
    prob_within_3h: float
    prob_within_6h: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'predicted_time': self.predicted_time,
            'predicted_hours_ahead': self.predicted_hours_ahead,
            'confidence': self.confidence,
            'mean_hours': self.mean_hours,
            'median_hours': self.median_hours,
            'std_hours': self.std_hours,
            'ci_lower_hours': self.ci_lower_hours,
            'ci_upper_hours': self.ci_upper_hours,
            'n_samples': self.n_samples,
            'model_version': self.model_version,
            'trained_at': self.trained_at,
            'prob_within_1h': self.prob_within_1h,
            'prob_within_3h': self.prob_within_3h,
            'prob_within_6h': self.prob_within_6h
        }


class NTPPModel(nn.Module):
    """
    Neural Temporal Point Process model using LSTM.

    Predicts inter-event times (time until next post) based on:
    - History of previous inter-event times
    - Event features (engagement, time-of-day, context, etc.)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize NTPP model.

        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(NTPPModel, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM encoder for event history
        self.lstm = nn.LSTM(
            input_size=feature_dim + 1,  # features + inter-event time
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Intensity function network
        # Maps (hidden_state, time_delta) -> intensity λ(t)
        self.intensity_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive intensity
        )

        # Initial hidden state
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))

        logger.info(f"NTPPModel initialized: feature_dim={feature_dim}, "
                   f"hidden_dim={hidden_dim}, num_layers={num_layers}")

    def forward(
        self,
        event_times: torch.Tensor,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            event_times: Inter-event times [batch_size, seq_len]
            features: Event features [batch_size, seq_len, feature_dim]
            hidden: Optional initial hidden state

        Returns:
            Tuple of (intensity_integral, hidden_state)
        """
        batch_size, seq_len = event_times.shape

        # Initialize hidden state if not provided
        if hidden is None:
            h0 = self.h0.expand(-1, batch_size, -1).contiguous()
            c0 = self.c0.expand(-1, batch_size, -1).contiguous()
            hidden = (h0, c0)

        # Concatenate times and features
        event_times_expanded = event_times.unsqueeze(-1)  # [batch, seq, 1]
        lstm_input = torch.cat([event_times_expanded, features], dim=-1)

        # Encode event history with LSTM
        lstm_out, hidden = self.lstm(lstm_input, hidden)  # [batch, seq, hidden]

        return lstm_out, hidden

    def compute_intensity(
        self,
        hidden_state: torch.Tensor,
        time_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intensity λ(t) at given time delta.

        Args:
            hidden_state: LSTM hidden state [batch, hidden_dim]
            time_delta: Time since last event [batch, 1]

        Returns:
            Intensity values [batch, 1]
        """
        # Concatenate hidden state and time delta
        intensity_input = torch.cat([hidden_state, time_delta], dim=-1)

        # Compute intensity through network
        intensity = self.intensity_net(intensity_input)

        return intensity

    def compute_log_likelihood(
        self,
        event_times: torch.Tensor,
        features: torch.Tensor,
        next_event_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        The log-likelihood for a temporal point process is:
        L = Σ log(λ(t_i)) - ∫ λ(t) dt

        Args:
            event_times: Inter-event times [batch, seq_len]
            features: Event features [batch, seq_len, feature_dim]
            next_event_times: Time to next event [batch, seq_len]

        Returns:
            Negative log-likelihood loss (scalar)
        """
        batch_size, seq_len = event_times.shape

        # Encode history
        lstm_out, _ = self.forward(event_times, features)

        # Compute intensity at event times
        # For each position, predict time to next event
        log_intensities = []
        intensity_integrals = []

        for i in range(seq_len):
            hidden = lstm_out[:, i, :]  # [batch, hidden]
            next_time = next_event_times[:, i:i+1]  # [batch, 1]

            # Intensity at event time
            intensity = self.compute_intensity(hidden, next_time)
            log_intensity = torch.log(intensity + 1e-8)
            log_intensities.append(log_intensity)

            # Integral of intensity from 0 to next_time
            # Approximate with Monte Carlo sampling
            n_samples = 10
            time_samples = torch.linspace(0, 1, n_samples, device=next_time.device)
            time_samples = time_samples.view(1, -1, 1) * next_time.unsqueeze(1)

            hidden_expanded = hidden.unsqueeze(1).expand(-1, n_samples, -1)
            time_samples_flat = time_samples.view(batch_size * n_samples, 1)
            hidden_flat = hidden_expanded.reshape(batch_size * n_samples, -1)

            intensities = self.compute_intensity(hidden_flat, time_samples_flat)
            intensities = intensities.view(batch_size, n_samples)

            # Trapezoidal integration
            integral = torch.trapz(intensities, time_samples.squeeze(2), dim=1, keepdim=True)
            intensity_integrals.append(integral)

        log_intensities = torch.cat(log_intensities, dim=1)  # [batch, seq]
        intensity_integrals = torch.cat(intensity_integrals, dim=1)  # [batch, seq]

        # Log-likelihood: Σ log(λ(t_i)) - ∫ λ(t) dt
        log_likelihood = log_intensities - intensity_integrals
        log_likelihood = log_likelihood.sum(dim=1).mean()  # Average over batch

        # Return negative log-likelihood (for minimization)
        return -log_likelihood

    def compute_survival_function(
        self,
        hidden_state: torch.Tensor,
        time_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute survival function S(t) = exp(-∫₀ᵗ λ(s)ds).
        
        The survival function gives the probability that no event
        has occurred by time t.
        
        Args:
            hidden_state: LSTM hidden state [batch, hidden_dim]
            time_points: Time points to evaluate [n_points]
            
        Returns:
            Survival probabilities [batch, n_points]
        """
        batch_size = hidden_state.shape[0]
        n_points = time_points.shape[0]
        
        # Compute cumulative intensity (integrated hazard)
        time_grid_expanded = time_points.view(1, -1, 1).expand(batch_size, -1, -1)
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, n_points, -1)
        
        time_grid_flat = time_grid_expanded.reshape(-1, 1)
        hidden_flat = hidden_expanded.reshape(-1, self.hidden_dim)
        
        # Get intensity at each time point
        intensities = self.compute_intensity(hidden_flat, time_grid_flat)
        intensities = intensities.view(batch_size, n_points)
        
        # Compute cumulative intensity using trapezoidal rule
        # Λ(t) = ∫₀ᵗ λ(s)ds
        dt = time_points[1] - time_points[0] if n_points > 1 else 0.1
        cumulative_intensity = torch.cumsum(intensities * dt, dim=1)
        
        # Survival function: S(t) = exp(-Λ(t))
        survival = torch.exp(-cumulative_intensity)
        
        return survival
    
    def compute_pdf(
        self,
        hidden_state: torch.Tensor,
        time_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute probability density function f(t) = λ(t) * S(t).
        
        Args:
            hidden_state: LSTM hidden state [batch, hidden_dim]
            time_points: Time points to evaluate [n_points]
            
        Returns:
            PDF values [batch, n_points]
        """
        batch_size = hidden_state.shape[0]
        n_points = time_points.shape[0]
        
        # Get intensity
        time_grid_expanded = time_points.view(1, -1, 1).expand(batch_size, -1, -1)
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, n_points, -1)
        
        time_grid_flat = time_grid_expanded.reshape(-1, 1)
        hidden_flat = hidden_expanded.reshape(-1, self.hidden_dim)
        
        intensities = self.compute_intensity(hidden_flat, time_grid_flat)
        intensities = intensities.view(batch_size, n_points)
        
        # Get survival function
        survival = self.compute_survival_function(hidden_state, time_points)
        
        # PDF = λ(t) * S(t)
        pdf = intensities * survival
        
        return pdf
    
    def sample_thinning(
        self,
        hidden_state: torch.Tensor,
        max_time: float = 48.0,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Sample inter-event times using Ogata's thinning algorithm.
        
        This is the proper way to sample from a temporal point process
        with a neural intensity function.
        
        Args:
            hidden_state: LSTM hidden state [batch, hidden_dim]
            max_time: Maximum time to consider (hours)
            n_samples: Number of samples to generate per batch element
            
        Returns:
            Sampled inter-event times [batch, n_samples]
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device
        
        samples = torch.zeros(batch_size, n_samples, device=device)
        
        # Estimate upper bound on intensity for thinning
        time_grid = torch.linspace(0.1, max_time, 100, device=device)
        time_grid_expanded = time_grid.view(1, -1, 1).expand(batch_size, -1, -1)
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, 100, -1)
        
        time_grid_flat = time_grid_expanded.reshape(-1, 1)
        hidden_flat = hidden_expanded.reshape(-1, self.hidden_dim)
        
        intensities = self.compute_intensity(hidden_flat, time_grid_flat)
        intensities = intensities.view(batch_size, 100)
        
        # Upper bound: max intensity * 1.5 for safety margin
        lambda_upper = intensities.max(dim=1)[0] * 1.5 + 0.1  # [batch]
        
        # Thinning algorithm for each batch element
        for b in range(batch_size):
            accepted = 0
            upper_bound = lambda_upper[b].item()
            
            while accepted < n_samples:
                # Propose from homogeneous Poisson process
                tau = np.random.exponential(1.0 / upper_bound)
                
                if tau > max_time:
                    # Reached max time, use max_time as sample
                    samples[b, accepted] = max_time
                    accepted += 1
                    continue
                
                # Acceptance probability
                t_tensor = torch.tensor([[tau]], device=device, dtype=torch.float32)
                h_tensor = hidden_state[b:b+1, :]
                
                intensity_at_t = self.compute_intensity(h_tensor, t_tensor).item()
                accept_prob = intensity_at_t / upper_bound
                
                if np.random.random() < accept_prob:
                    samples[b, accepted] = tau
                    accepted += 1
        
        return samples

    def predict_next_event_time(
        self,
        event_times: torch.Tensor,
        features: torch.Tensor,
        sample_method: str = 'expected',
        n_samples: int = 100,
        mc_dropout_samples: int = 10
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Predict time to next event with uncertainty quantification.

        Args:
            event_times: History of inter-event times [batch, seq_len]
            features: Event features [batch, seq_len, feature_dim]
            sample_method: 'expected', 'median', or 'sample'
            n_samples: Number of samples for Monte Carlo estimation
            mc_dropout_samples: Number of MC dropout forward passes

        Returns:
            Tuple of (predicted_time [batch, 1], statistics dict)
        """
        batch_size = event_times.shape[0]
        device = event_times.device
        
        # Collect predictions from multiple MC dropout passes
        all_samples = []
        
        for _ in range(mc_dropout_samples):
            # Keep dropout active for uncertainty estimation
            self.train()  # Enable dropout
            with torch.no_grad():
                lstm_out, _ = self.forward(event_times, features)
                hidden = lstm_out[:, -1, :]  # Use last hidden state
                
                # Sample using thinning algorithm
                samples = self.sample_thinning(hidden, max_time=48.0, n_samples=n_samples // mc_dropout_samples)
                all_samples.append(samples)
        
        self.eval()  # Disable dropout for final prediction
        
        # Combine all samples [batch, total_samples]
        all_samples = torch.cat(all_samples, dim=1)
        
        # Compute statistics
        mean_time = all_samples.mean(dim=1, keepdim=True)
        median_time = all_samples.median(dim=1, keepdim=True)[0]
        std_time = all_samples.std(dim=1, keepdim=True)
        
        # Confidence intervals (5th and 95th percentiles)
        sorted_samples, _ = torch.sort(all_samples, dim=1)
        ci_lower_idx = int(0.05 * all_samples.shape[1])
        ci_upper_idx = int(0.95 * all_samples.shape[1])
        ci_lower = sorted_samples[:, ci_lower_idx:ci_lower_idx+1]
        ci_upper = sorted_samples[:, ci_upper_idx:ci_upper_idx+1]
        
        # Compute probability of event within different horizons
        prob_within_1h = (all_samples <= 1.0).float().mean(dim=1)
        prob_within_3h = (all_samples <= 3.0).float().mean(dim=1)
        prob_within_6h = (all_samples <= 6.0).float().mean(dim=1)
        
        # Select prediction based on method
        if sample_method == 'expected':
            predicted_time = mean_time
        elif sample_method == 'median':
            predicted_time = median_time
        elif sample_method == 'sample':
            # Return a random sample
            sample_idx = np.random.randint(0, all_samples.shape[1])
            predicted_time = all_samples[:, sample_idx:sample_idx+1]
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")
        
        # Build statistics dictionary
        stats = {
            'mean_hours': mean_time.squeeze(-1),
            'median_hours': median_time.squeeze(-1),
            'std_hours': std_time.squeeze(-1),
            'ci_lower_hours': ci_lower.squeeze(-1),
            'ci_upper_hours': ci_upper.squeeze(-1),
            'prob_within_1h': prob_within_1h,
            'prob_within_3h': prob_within_3h,
            'prob_within_6h': prob_within_6h,
            'n_samples': all_samples.shape[1],
            'samples': all_samples  # Raw samples for further analysis
        }
        
        return predicted_time, stats


class NTPPPredictor:
    """
    High-level interface for NTPP-based timing prediction.

    Integrates with feature engineering and provides Prophet-like interface.
    """

    def __init__(
        self,
        feature_dim: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize NTPP predictor.

        Args:
            feature_dim: Dimension of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for training
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.model = NTPPModel(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.feature_dim = feature_dim
        self.last_trained = None

        logger.info(f"NTPPPredictor initialized on device: {device}")

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sequence_length: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for training/prediction.

        Args:
            df: DataFrame with features (must be sorted by time)
            feature_cols: List of feature column names
            sequence_length: Length of history sequences

        Returns:
            Tuple of (event_times, features, next_event_times) tensors
        """
        # Extract inter-event times (already computed in features as 'time_since_last_hours')
        if 'time_since_last_hours' in df.columns:
            inter_event_times = df['time_since_last_hours'].values
        else:
            raise ValueError("DataFrame must contain 'time_since_last_hours'")

        # Extract features
        feature_values = df[feature_cols].values
        n_samples = len(df)

        # Create sequences
        event_times_seqs = []
        feature_seqs = []
        next_event_seqs = []

        for i in range(sequence_length, n_samples):
            # History sequence
            event_times_seqs.append(inter_event_times[i-sequence_length:i])
            feature_seqs.append(feature_values[i-sequence_length:i])

            # Next event time (target)
            next_event_seqs.append(inter_event_times[i])

        # Convert to tensors
        event_times = torch.FloatTensor(np.array(event_times_seqs)).to(self.device)
        features = torch.FloatTensor(np.array(feature_seqs)).to(self.device)
        next_events = torch.FloatTensor(np.array(next_event_seqs)).unsqueeze(1).to(self.device)

        # Expand next_events to match sequence length for loss computation
        next_events_expanded = next_events.expand(-1, sequence_length)

        logger.info(f"Prepared {len(event_times_seqs)} sequences of length {sequence_length}")

        return event_times, features, next_events_expanded

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        sequence_length: int = 20,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train NTPP model.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            epochs: Number of training epochs
            batch_size: Batch size
            sequence_length: Sequence length for LSTM
            validation_split: Fraction for validation

        Returns:
            Training history dict
        """
        logger.info(f"Training NTPP model for {epochs} epochs...")

        # Prepare data
        event_times, features, next_events = self.prepare_sequences(
            df, feature_cols, sequence_length
        )

        n_samples = event_times.shape[0]
        n_train = int(n_samples * (1 - validation_split))

        train_times = event_times[:n_train]
        train_features = features[:n_train]
        train_next = next_events[:n_train]

        val_times = event_times[n_train:]
        val_features = features[n_train:]
        val_next = next_events[n_train:]

        history = {'train_loss': [], 'val_loss': []}

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = (n_train + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_train)

                batch_times = train_times[start_idx:end_idx]
                batch_features = train_features[start_idx:end_idx]
                batch_next = train_next[start_idx:end_idx]

                # Forward pass
                self.optimizer.zero_grad()
                loss = self.model.compute_log_likelihood(
                    batch_times, batch_features, batch_next
                )

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / n_batches

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = self.model.compute_log_likelihood(
                    val_times, val_features, val_next
                ).item()

            self.model.train()

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"train_loss={avg_train_loss:.4f}, "
                           f"val_loss={val_loss:.4f}")

        self.last_trained = datetime.now()
        logger.success(f"Training complete! Final val_loss: {history['val_loss'][-1]:.4f}")

        return history

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sequence_length: int = 20,
        n_samples: int = 100,
        mc_dropout_samples: int = 10,
        sample_method: str = 'median',
        return_structured: bool = True
    ) -> Union[Dict, NTPPPrediction]:
        """
        Predict next post time with uncertainty quantification.

        Args:
            df: DataFrame with recent history
            feature_cols: Feature column names
            sequence_length: Sequence length
            n_samples: Number of samples for Monte Carlo estimation
            mc_dropout_samples: Number of MC dropout forward passes
            sample_method: 'expected', 'median', or 'sample'
            return_structured: If True, return NTPPPrediction object

        Returns:
            Prediction dict or NTPPPrediction with time, confidence, and uncertainty
        """
        # Use last sequence_length events
        recent_df = df.tail(sequence_length).copy()

        if len(recent_df) < sequence_length:
            logger.warning(f"Insufficient history: {len(recent_df)} < {sequence_length}")
            # Pad with zeros if needed
            padding_needed = sequence_length - len(recent_df)
            padding_df = pd.DataFrame(
                np.zeros((padding_needed, len(recent_df.columns))),
                columns=recent_df.columns
            )
            recent_df = pd.concat([padding_df, recent_df], ignore_index=True)

        # Prepare sequence
        inter_event_times = recent_df['time_since_last_hours'].values.astype(np.float32)
        feature_values = recent_df[feature_cols].values.astype(np.float32)

        event_times = torch.FloatTensor(inter_event_times).unsqueeze(0).to(self.device)
        features = torch.FloatTensor(feature_values).unsqueeze(0).to(self.device)

        # Predict with uncertainty quantification
        predicted_time_hours, stats = self.model.predict_next_event_time(
            event_times, features,
            sample_method=sample_method,
            n_samples=n_samples,
            mc_dropout_samples=mc_dropout_samples
        )

        predicted_time_hours_val = float(predicted_time_hours.item())
        
        # Calculate predicted datetime
        last_post_time = df['created_at'].iloc[-1]
        if hasattr(last_post_time, 'to_pydatetime'):
            last_post_time = last_post_time.to_pydatetime()
        predicted_datetime = last_post_time + timedelta(hours=predicted_time_hours_val)

        # Compute confidence based on uncertainty
        # Lower std = higher confidence, also factor in probability of near-term event
        std_hours = float(stats['std_hours'].item())
        prob_within_6h = float(stats['prob_within_6h'].item())
        
        # Confidence: combination of low uncertainty and high probability of soon event
        # Normalize std: 0 hours -> 1.0, 12+ hours -> ~0.2
        uncertainty_score = 1.0 / (1.0 + std_hours / 4.0)
        # Combine with probability of event within 6 hours
        confidence = 0.6 * uncertainty_score + 0.4 * prob_within_6h
        confidence = max(0.05, min(0.99, confidence))

        if return_structured:
            return NTPPPrediction(
                predicted_time=predicted_datetime,
                predicted_hours_ahead=predicted_time_hours_val,
                confidence=confidence,
                mean_hours=float(stats['mean_hours'].item()),
                median_hours=float(stats['median_hours'].item()),
                std_hours=std_hours,
                ci_lower_hours=float(stats['ci_lower_hours'].item()),
                ci_upper_hours=float(stats['ci_upper_hours'].item()),
                n_samples=stats['n_samples'],
                model_version='ntpp_v2_uncertainty',
                trained_at=self.last_trained,
                prob_within_1h=float(stats['prob_within_1h'].item()),
                prob_within_3h=float(stats['prob_within_3h'].item()),
                prob_within_6h=prob_within_6h
            )
        else:
            # Return dict for backwards compatibility
            return {
                'predicted_time': predicted_datetime,
                'predicted_hours_ahead': predicted_time_hours_val,
                'confidence': confidence,
                'model_version': 'ntpp_v2_uncertainty',
                'trained_at': self.last_trained,
                'mean_hours': float(stats['mean_hours'].item()),
                'median_hours': float(stats['median_hours'].item()),
                'std_hours': std_hours,
                'ci_lower_hours': float(stats['ci_lower_hours'].item()),
                'ci_upper_hours': float(stats['ci_upper_hours'].item()),
                'prob_within_1h': float(stats['prob_within_1h'].item()),
                'prob_within_3h': float(stats['prob_within_3h'].item()),
                'prob_within_6h': prob_within_6h
            }

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'feature_dim': self.feature_dim,
            'last_trained': self.last_trained
        }, path)

        logger.success(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.feature_dim = checkpoint['feature_dim']
        self.last_trained = checkpoint['last_trained']

        logger.success(f"Model loaded from {path}")


def test_ntpp():
    """Test NTPP model with uncertainty quantification."""
    print("\n" + "="*80)
    print("NEURAL TEMPORAL POINT PROCESS TEST (v2 with Uncertainty)")
    print("="*80 + "\n")

    # Create synthetic data
    np.random.seed(42)
    n_events = 100

    # Simulate posting pattern: mix of regular and bursts
    inter_event_times = []
    for i in range(n_events):
        if i % 20 < 5:  # Burst every 20 posts
            inter_event_times.append(np.random.exponential(0.5))
        else:
            inter_event_times.append(np.random.exponential(3.0))

    timestamps = [datetime.now()]
    for dt in inter_event_times[1:]:
        timestamps.append(timestamps[-1] + timedelta(hours=dt))

    # Create features
    df = pd.DataFrame({
        'created_at': timestamps,
        'time_since_last_hours': inter_event_times,
        'hour': [t.hour for t in timestamps],
        'is_weekend': [t.weekday() >= 5 for t in timestamps],
        'engagement_total': np.random.randint(100, 1000, n_events)
    })

    print(f"Generated {len(df)} synthetic events")
    print(f"Mean inter-event time: {df['time_since_last_hours'].mean():.2f}h")
    print(f"Std inter-event time: {df['time_since_last_hours'].std():.2f}h")
    print()

    # Initialize predictor
    predictor = NTPPPredictor(
        feature_dim=3,  # hour, is_weekend, engagement
        hidden_dim=32,
        num_layers=1
    )

    # Train
    print("Training NTPP model...")
    history = predictor.train(
        df,
        feature_cols=['hour', 'is_weekend', 'engagement_total'],
        epochs=20,
        batch_size=16,
        sequence_length=10
    )

    print(f"\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print()

    # Predict with uncertainty quantification
    print("Making prediction with uncertainty quantification...")
    prediction = predictor.predict(
        df,
        feature_cols=['hour', 'is_weekend', 'engagement_total'],
        sequence_length=10,
        n_samples=100,
        mc_dropout_samples=5,
        return_structured=True
    )

    print(f"\n{'='*60}")
    print("PREDICTION RESULTS (with Uncertainty)")
    print(f"{'='*60}")
    print(f"\n📅 Predicted Time: {prediction.predicted_time}")
    print(f"⏱️  Hours Ahead: {prediction.predicted_hours_ahead:.2f}h")
    print(f"🎯 Confidence: {prediction.confidence:.2%}")
    print()
    print("Distribution Statistics:")
    print(f"  • Mean: {prediction.mean_hours:.2f}h")
    print(f"  • Median: {prediction.median_hours:.2f}h")
    print(f"  • Std Dev: {prediction.std_hours:.2f}h")
    print(f"  • 90% CI: [{prediction.ci_lower_hours:.2f}h, {prediction.ci_upper_hours:.2f}h]")
    print()
    print("Probability of Post Within:")
    print(f"  • 1 hour: {prediction.prob_within_1h:.1%}")
    print(f"  • 3 hours: {prediction.prob_within_3h:.1%}")
    print(f"  • 6 hours: {prediction.prob_within_6h:.1%}")
    print()
    print(f"Samples used: {prediction.n_samples}")
    print(f"Model version: {prediction.model_version}")
    print(f"{'='*60}\n")

    # Test backwards compatible dict output
    print("Testing backwards-compatible dict output...")
    pred_dict = predictor.predict(
        df,
        feature_cols=['hour', 'is_weekend', 'engagement_total'],
        sequence_length=10,
        return_structured=False
    )
    print(f"Dict keys: {list(pred_dict.keys())}")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_ntpp()

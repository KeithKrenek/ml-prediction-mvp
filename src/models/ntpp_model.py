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

References:
- "Neural Temporal Point Processes" (Mei & Eisner, 2017)
- "Transformer Hawkes Process" (Zuo et al., 2020)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
import pickle
from pathlib import Path


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

    def predict_next_event_time(
        self,
        event_times: torch.Tensor,
        features: torch.Tensor,
        sample_method: str = 'expected'
    ) -> torch.Tensor:
        """
        Predict time to next event.

        Args:
            event_times: History of inter-event times [batch, seq_len]
            features: Event features [batch, seq_len, feature_dim]
            sample_method: 'expected' for E[t], 'sample' for random sample

        Returns:
            Predicted inter-event time [batch, 1]
        """
        self.eval()
        with torch.no_grad():
            # Encode history
            lstm_out, _ = self.forward(event_times, features)
            hidden = lstm_out[:, -1, :]  # Use last hidden state

            if sample_method == 'expected':
                # Compute expected inter-event time
                # E[t] = ∫ t * λ(t) * exp(-∫λ(s)ds) dt
                # Approximate with grid search

                max_time = 48.0  # Maximum 48 hours
                n_points = 100
                time_grid = torch.linspace(0.1, max_time, n_points, device=hidden.device)

                batch_size = hidden.shape[0]
                time_grid_expanded = time_grid.view(1, -1, 1).expand(batch_size, -1, -1)

                hidden_expanded = hidden.unsqueeze(1).expand(-1, n_points, -1)

                # Compute intensity at each time point
                time_grid_flat = time_grid_expanded.reshape(-1, 1)
                hidden_flat = hidden_expanded.reshape(-1, self.hidden_dim)

                intensities = self.compute_intensity(hidden_flat, time_grid_flat)
                intensities = intensities.view(batch_size, n_points)

                # Find mode (peak intensity) as prediction
                max_idx = torch.argmax(intensities, dim=1)
                predicted_time = time_grid[max_idx].unsqueeze(1)

                return predicted_time

            elif sample_method == 'sample':
                # Sample from intensity using inverse CDF
                # (More complex, implement if needed)
                raise NotImplementedError("Sampling not yet implemented")

            else:
                raise ValueError(f"Unknown sample method: {sample_method}")


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
        sequence_length: int = 20
    ) -> Dict:
        """
        Predict next post time.

        Args:
            df: DataFrame with recent history
            feature_cols: Feature column names
            sequence_length: Sequence length

        Returns:
            Prediction dict with time and confidence
        """
        self.model.eval()

        # Use last sequence_length events
        recent_df = df.tail(sequence_length)

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
        inter_event_times = recent_df['time_since_last_hours'].values
        feature_values = recent_df[feature_cols].values

        event_times = torch.FloatTensor(inter_event_times).unsqueeze(0).to(self.device)
        features = torch.FloatTensor(feature_values).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predicted_time_hours = self.model.predict_next_event_time(
                event_times, features, sample_method='expected'
            )

        predicted_time_hours = predicted_time_hours.item()

        # Calculate predicted datetime
        last_post_time = df['created_at'].iloc[-1]
        predicted_datetime = last_post_time + timedelta(hours=predicted_time_hours)

        # Estimate confidence (inverse of prediction time - shorter = more confident)
        confidence = 1.0 / (1.0 + predicted_time_hours / 6.0)  # Normalize by 6 hours

        return {
            'predicted_time': predicted_datetime,
            'predicted_hours_ahead': predicted_time_hours,
            'confidence': confidence,
            'model_version': 'ntpp_v1',
            'trained_at': self.last_trained
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
    """Test NTPP model."""
    print("\n" + "="*80)
    print("NEURAL TEMPORAL POINT PROCESS TEST")
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

    # Predict
    print("Making prediction...")
    prediction = predictor.predict(
        df,
        feature_cols=['hour', 'is_weekend', 'engagement_total'],
        sequence_length=10
    )

    print(f"\nPrediction:")
    print(f"  Time: {prediction['predicted_time']}")
    print(f"  Hours ahead: {prediction['predicted_hours_ahead']:.2f}h")
    print(f"  Confidence: {prediction['confidence']:.2f}")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_ntpp()

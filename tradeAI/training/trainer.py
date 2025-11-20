"""
Model trainer for TradeAI
"""

from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from tradeAI.models.base_model import BaseModel
from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Train neural network models for trading prediction."""

    def __init__(
        self,
        model: BaseModel,
        device: Optional[str] = None,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            device: Device to train on (None = auto-detect)
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Will be set during training
        self.optimizer = None
        self.criterion = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        logger.info(f"Trainer initialized: device={device}, model={model}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        task: str = "classification",
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            task: Task type ('classification' or 'regression')

        Returns:
            Training history
        """
        logger.info(
            f"Starting training: {len(X_train)} training samples, "
            f"{len(X_val) if X_val is not None else 0} validation samples"
        )

        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task}")

        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, task)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader, task)
            else:
                val_loss, val_acc = 0.0, 0.0

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
            )

            # Early stopping
            if val_loader is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    if self.checkpoint_dir:
                        self._save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info("Training completed!")

        return self.history

    def _train_epoch(self, dataloader: DataLoader, task: str) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)

            # Calculate loss
            if task == "classification":
                loss = self.criterion(outputs, batch_y)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            else:
                loss = self.criterion(outputs.squeeze(), batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _validate_epoch(self, dataloader: DataLoader, task: str) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)

                # Calculate loss
                if task == "classification":
                    loss = self.criterion(outputs, batch_y)

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
                else:
                    loss = self.criterion(outputs.squeeze(), batch_y)

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        self.model.eval()

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

            # For classification, return class probabilities
            if outputs.shape[1] > 1:
                predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.cpu().numpy()

        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task: str = "classification",
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            task: Task type

        Returns:
            Dictionary of metrics
        """
        test_loader = self._create_dataloader(X_test, y_test, shuffle=False)
        test_loss, test_acc = self._validate_epoch(test_loader, task)

        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }

        logger.info(f"Test metrics: {metrics}")

        return metrics

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if y.dtype == np.int64 else torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

        return dataloader

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        self.model.save(str(path))

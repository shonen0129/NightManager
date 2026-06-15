"""Abstract Base Class for strategy models."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract Base Class for Lead-Lag strategy models."""

    @abstractmethod
    def predict_signals(self, df_exec: pd.DataFrame) -> dict[str, np.ndarray]:
        """Generate raw signals from the execution dataset.

        Args:
            df_exec: Execution DataFrame.

        Returns:
            Dict containing the final signal array (shape N_J,) and other component signals.
        """
        pass

    @abstractmethod
    def build_weights(self, signals: np.ndarray) -> np.ndarray:
        """Construct portfolio weights from signals.

        Args:
            signals: Signal array of shape (n_j,).

        Returns:
            Weight array of shape (n_j,).
        """
        pass

import torch
import torch.nn as nn
from loguru import logger

class Emotion1DCNN(nn.Module):
    def __init__(self, num_classes: int = 12):
        """
        1D-CNN for Audio Emotion Classification.
        Designed to be extremely lightweight (< 50M params) to meet project constraints.
        """
        super().__init__()
        
        # Input shape expected: (Batch_Size, 128 mel_bins, Time_Steps)
        # We treat the 128 mel bins as the input channels.
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # AdaptiveAvgPool1d gracefully squashes any variable time-dimension down to 1
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the 512-dim audio embedding (before the classifier head).
        Used by NabatiMultimodalFusion to extract audio features.

        Args:
            x: Tensor of shape (Batch, 128, Time)
        Returns:
            Tensor of shape (Batch, 512)
        """
        x = self.features(x)       # (Batch, 512, 1)
        return torch.flatten(x, 1) # (Batch, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 128, Time)
        Returns:
            Tensor of shape (Batch, num_classes) containing raw logits
        """
        emb = self.embed(x)          # (Batch, 512)
        return self.classifier(emb)  # (Batch, num_classes)

if __name__ == "__main__":
    # Configure logger for the test
    logger.add("logs/models.log", rotation="10 MB")
    
    # Instantiate the model for the 12 emotion classes
    model = Emotion1DCNN(num_classes=12)
    
    # Create a dummy tensor that mimics the shape from our dataset (Batch=16, Mels=128, Time=157)
    dummy_input = torch.randn(16, 128, 157) 
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model instantiated successfully.")
    logger.info(f"Output shape: {output.shape} (Expected: torch.Size([16, 12]))")
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    if total_params < 50_000_000:
        logger.success("Model is well under the 50M parameter limit for the scratch model!")
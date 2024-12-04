import torch          # type: ignore
import torch.nn as nn # type: ignore
import timm           # type: ignore

    
class TimmModel(nn.Module):
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
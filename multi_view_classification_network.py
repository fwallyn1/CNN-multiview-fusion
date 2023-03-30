import torch
import torch.nn as nn
from typing import Dict, List, Optional
torch.manual_seed(0)
torch.cuda.manual_seed(0)
class MultiInputModel(nn.Module):
    """
    Multi input model (need to be adapted for more that two views)
    """
    def __init__(self,models:List[Dict],classifier:Optional[torch.nn.Sequential]=torch.nn.Sequential(),pretrained=True, fine_tune=False, num_classes=7) -> None:
        """
        Constructs Two Input model with two model dictionnary, like those in config.py

        Args:
            models (List[Dict]): models descriptions
            pretrained (bool, optional): pretrained model or not. Defaults to True.
            fine_tune (bool, optional): fine tune during training or not. Defaults to False.
            num_classes (int, optional): number of classes to predict. Defaults to 7.
        """
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        super().__init__()
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        else:
            print('[INFO]: Not loading pre-trained weights')
        #self.modelRecto = model_recto['model']
        self.models_avg_pool = torch.nn.ModuleList()
        self.models_name = []
        self.models_features = torch.nn.ModuleList()
        self.classifier = classifier
        self.model_last_layer_size = 0 
        for model in models :
            self.models_avg_pool.append(model['avgpool'])
            self.model_last_layer_size += model['last_layer_size']
            self.models_name.append(model['model_name'])
            self.models_features.append(model['model'].features if not model['features'] else model['features'])
        
        input_tensor = torch.randn(1, self.model_last_layer_size)
        # Pass the input tensor through the sequential module
        output_tensor = classifier(input_tensor)
        # Get the output size of the sequential module
        self.model_last_layer_size = output_tensor.size(-1)
        self.drop = nn.Dropout(0.2)
        self.name = "_".join(self.models_name)
        self.classifier.append(nn.Linear(self.model_last_layer_size,num_classes))
        #self.classifier = nn.Sequential(nn.Linear(self.last_layer_size_recto+self.last_layer_size_verso,1024), 
                       #                 nn.ReLU(), nn.Dropout(0.2),nn.Linear(1024,num_classes))
        if fine_tune:
            print('[INFO]: Fine-tuning last features layers...')
            for params in self.models_features.parameters():
                params.requires_grad = True
            for params in self.models_features.parameters():
                params.requires_grad = True
            #self.modelRecto_features[-1].requires_grad_(requires_grad=True)
            #self.modelVerso_features[-3:].requires_grad_(requires_grad=True)
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.models_features.parameters():
                params.requires_grad = False
            for params in self.models_features.parameters():
                params.requires_grad = False
            
    def forward(self, views:List[torch.Tensor]):
        """
        Args:
            two_views (List[torch.Tensor]): Lis of the two images tensors

        Returns:
            tensor: outputs of model
        """
        z_V = []
        for i, (v,feature,avg_pool) in enumerate(zip(views,self.models_features,self.models_avg_pool)) :
            z = feature(v) 
            z = avg_pool(z) if avg_pool else z
            z = torch.flatten(z,1)
            z = self.drop(z)
            z_V.append(z)
        z_cat = torch.cat(z_V, dim=1)  
        z_fc = self.classifier(z_cat)
        return z_fc
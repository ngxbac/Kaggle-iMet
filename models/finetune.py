import torch
import torch.nn as nn
from cnn_finetune import make_model



"""
        self.model = make_model(
            model_name=arch,
            num_classes=n_class,
            pretrained=pretrained,
            input_size=(image_size, image_size),
        )
        
        self.embedding_hour = nn.Sequential(
            nn.Embedding(num_embeddings=n_hour+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )
        
        self.embedding_day = nn.Sequential(
            nn.Embedding(num_embeddings=n_day+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )
            
        self.embedding_month = nn.Sequential(
            nn.Embedding(num_embeddings=n_month+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )

"""


class Finetune(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        pretrained=True,
        image_size=256,
        **kwargs
    ):
        super(Finetune, self).__init__()
        self.model = make_model(
            model_name=arch,
            num_classes=n_class,
            pretrained=pretrained,
            input_size=(image_size, image_size),
        )

    def freeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        return self.model(x)

    
class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # xx = x.unsqueeze_(-1)
        return nn.functional.avg_pool1d(
            input=x,
            kernel_size=x.shape[1],
            padding=x.shape[1]//2
        )


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # xx = x.unsqueeze_(-1)
        return nn.functional.max_pool1d(
            input=x,
            kernel_size=x.shape[1],
            padding=x.shape[1]//2
        )


class GlobalConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = GlobalAvgPool1d()
        self.max = GlobalMaxPool1d()

    def forward(self, x):
        x = x.unsqueeze_(-1)
        return torch.cat([self.avg(x), self.max(x)], 1)
    
    
class FinetuneEmbedding(nn.Module):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        pretrained=True,
        image_size=256,
        n_hour=24,
        n_day=31,
        n_month=12,
        embedding_dim=10,
        **kwargs
    ):
        super(FinetuneEmbedding, self).__init__()
        self.model = make_model(
            model_name=arch,
            num_classes=n_class,
            pretrained=pretrained,
            input_size=(image_size, image_size),
        )
#         print(self.model)
        
        output_num = self.model._classifier.in_features
        
        self.embedding_hour = nn.Sequential(
            nn.Embedding(num_embeddings=n_hour+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )
        
        self.embedding_day = nn.Sequential(
            nn.Embedding(num_embeddings=n_day+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )
            
        self.embedding_month = nn.Sequential(
            nn.Embedding(num_embeddings=n_month+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )
        
        output_num += embedding_dim * 3 * 2
        self.fc = nn.Linear(output_num, n_class)

    def freeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x, x_hour, x_day, x_month):
        x = self.model._features(x)
        x = self.model.pool(x)
        x = x.view(x.size(0), -1)
        
        x_hour = self.embedding_hour(x_hour)
        x_hour = x_hour.view(x_hour.size(0), -1)
        
        x_day = self.embedding_day(x_day)
        x_day = x_day.view(x_day.size(0), -1)
        
        x_month = self.embedding_month(x_month)
        x_month = x_month.view(x_month.size(0), -1)
        
        x = torch.cat([x, x_hour, x_day, x_month], 1)
        x = self.fc(x)
        return x


def finetune(params):
    return Finetune(**params)


def finetune_embedding(params):
    return FinetuneEmbedding(**params)
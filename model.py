import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
    
class unet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,features=[64,128,256,512]):
        super(unet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(doubleconv(in_channels, feature))
            in_channels = feature
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)  #can use bilinear then conv layer
            )
            self.decoder.append(doubleconv(feature*2, feature))
        
        self.bottleneck = doubleconv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x,size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
            
        return self.final_conv(x)
    

def test():
    x = torch.randn((3, 3, 161, 161))
    model = unet(in_channels=3, out_channels=5)
    preds = model(x)
    print(preds.shape)
    
if __name__ == "__main__":
    test()
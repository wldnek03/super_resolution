
from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return x
        from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # 첫 번째 Conv 계층: 9x9 필터, 출력 채널 64
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 두 번째 Conv 계층: 1x1 필터, 출력 채널 32
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 세 번째 Conv 계층: 5x5 필터, 출력 채널 1 (밝기 정보만 복원)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # 순차적으로 Conv -> ReLU 적용
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)  # 마지막 계층에는 ReLU 없음
        return x
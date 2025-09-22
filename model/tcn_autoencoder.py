import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    """
    TCN(Temporal Convolutional Network)의 기본 구성 블록입니다.
    Causal Convolution을 적용하여 시간적 순서를 보존합니다.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, dilation: int = 1, dropout: float = 0.2):
        """
        Args:
            in_channels (int): 입력 채널의 수.
            out_channels (int): 출력 채널의 수.
            kernel_size (int): 컨볼루션 커널의 크기.
            dilation (int): 컨볼루션의 팽창(dilation) 정도.
            dropout (float): 드롭아웃 비율.
        """
        super().__init__()
        
        # Causal Convolution을 위해 필요한 패딩 계산
        self.padding = (kernel_size - 1) * dilation
        
        # 1D 컨볼루션 레이어
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        # Weight Normalization 적용
        self.conv1 = nn.utils.weight_norm(self.conv1)
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Residual connection을 위한 1x1 conv (채널 수가 다를 경우)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서 (Batch, Channels, Length)

        Returns:
            torch.Tensor: TCN 블록의 출력 텐서 (Batch, Channels, Length)
        """
        # 컨볼루션 및 활성화 함수 적용
        out = self.conv1(x)
        
        # Causal Convolution: 미래의 정보가 현재에 영향을 주지 않도록 끝부분을 잘라냄
        out = out[:, :, :-self.padding]
        out = self.relu1(out)
        out = self.dropout1(out)

        # Residual connection (잔차 연결)
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu_out(out + res)


class TCNEncoder(nn.Module):
    """
    TCN 블록을 사용하여 시계열 데이터의 포인트별 임베딩 시퀀스를 생성하는 인코더입니다.
    """
    def __init__(self, in_channels: int, embedding_dim: int, hidden_channels: int, depth: int = 3, kernel_size: int = 4):
        """
        Args:
            in_channels (int): 입력 시계열의 변수 차원.
            embedding_dim (int): 출력 임베딩 벡터의 차원.
            hidden_channels (int): TCN 블록 내부의 은닉 채널 수.
            depth (int): TCN 블록을 쌓을 깊이.
            kernel_size (int): 컨볼루션 커널의 크기.
        """
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        # TCN 블록을 점진적으로 증가하는 dilation과 함께 쌓음
        for i in range(depth):
            dilation = 2**i
            layers.append(
                TCNBlock(
                    current_channels, hidden_channels, 
                    kernel_size=kernel_size, dilation=dilation
                )
            )
            current_channels = hidden_channels
            
        # 마지막 레이어는 최종 임베딩 차원으로 매핑
        layers.append(nn.Conv1d(current_channels, embedding_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        🔥 핵심: Temporal Pooling을 적용하지 않아 포인트별 임베딩 시퀀스를 반환합니다.

        Args:
            x (torch.Tensor): 입력 텐서 (Batch, Length, Channels_in)

        Returns:
            torch.Tensor: 포인트별 임베딩 시퀀스 (Batch, Length, Channels_out)
        """
        # (B, L, C) -> (B, C, L) 형태로 변환하여 Conv1d에 입력
        x = x.permute(0, 2, 1)
        
        embedding_sequence = self.network(x)
        
        # (B, C, L) -> (B, L, C) 형태로 다시 변환하여 반환
        return embedding_sequence.permute(0, 2, 1)


class TCNDecoder(nn.Module):
    """
    포인트별 임베딩 시퀀스를 받아 원본 시계열을 재구성하는 디코더입니다.
    인코더와 대칭적인 구조를 가집니다.
    """
    def __init__(self, embedding_dim: int, out_channels: int, hidden_channels: int, depth: int = 3, kernel_size: int = 4):
        """
        Args:
            embedding_dim (int): 입력 임베딩 벡터의 차원.
            out_channels (int): 출력 시계열의 변수 차원.
            hidden_channels (int): TCN 블록 내부의 은닉 채널 수.
            depth (int): TCN 블록을 쌓을 깊이.
            kernel_size (int): 컨볼루션 커널의 크기.
        """
        super().__init__()

        layers = []
        current_channels = embedding_dim
        
        # TCN 블록을 점진적으로 감소하는 dilation과 함께 쌓음 (인코더의 역순)
        for i in range(depth - 1):
            dilation = 2**(depth - 1 - i)
            layers.append(
                TCNBlock(
                    current_channels, hidden_channels, 
                    kernel_size=kernel_size, dilation=dilation
                )
            )
            current_channels = hidden_channels
        
        # 마지막 레이어는 dilation이 1인 TCN 블록
        layers.append(TCNBlock(current_channels, hidden_channels, kernel_size=kernel_size, dilation=1))
        
        # 최종 출력 채널로 매핑
        layers.append(nn.Conv1d(hidden_channels, out_channels, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): 포인트별 임베딩 시퀀스 (Batch, Length, Channels_in)

        Returns:
            torch.Tensor: 재구성된 시계열 텐서 (Batch, Length, Channels_out)
        """
        # (B, L, C) -> (B, C, L) 형태로 변환하여 Conv1d에 입력
        z = z.permute(0, 2, 1)
        
        reconstructed_sequence = self.network(z)
        
        # (B, C, L) -> (B, L, C) 형태로 다시 변환하여 반환
        return reconstructed_sequence.permute(0, 2, 1)


class TCNAutoEncoder(nn.Module):
    """
    TCN 인코더와 디코더를 결합한 최종 AutoEncoder 모델입니다.
    """
    def __init__(self, in_channels: int, embedding_dim: int = 16, hidden_channels: int = 64, depth: int = 3, kernel_size: int = 4):
        """
        Args:
            in_channels (int): 입력 시계열의 변수 차원.
            embedding_dim (int): 임베딩 벡터의 차원.
            hidden_channels (int): TCN 블록 내부의 은닉 채널 수.
            depth (int): TCN 블록의 깊이.
            kernel_size (int): 컨볼루션 커널의 크기.
        """
        super().__init__()
        self.encoder = TCNEncoder(in_channels, embedding_dim, hidden_channels, depth, kernel_size)
        self.decoder = TCNDecoder(embedding_dim, in_channels, hidden_channels, depth, kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        모델의 순전파 로직.

        Args:
            x (torch.Tensor): 원본 시계열 텐서 (Batch, Length, Channels)

        Returns:
            z (torch.Tensor): 포인트별 임베딩 시퀀스 (Batch, Length, embedding_dim)
            x_recon (torch.Tensor): 재구성된 시계열 텐서 (Batch, Length, Channels)
        """
        # 1. 인코더를 통해 포인트별 임베딩 시퀀스 z를 생성
        z = self.encoder(x)
        
        # 2. 디코더를 통해 z로부터 원본 시계열 x_recon을 재구성
        x_recon = self.decoder(z)
        
        return z, x_recon
import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, TestDataset
from utils import AverageMeter, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 학습 데이터 경로
    parser.add_argument('--train-dir', type=str, default='./db/train')
    # 테스트 데이터 경로
    parser.add_argument('--test-dir', type=str, default='./db/test')
    # 모델 파일 저장 경로
    parser.add_argument('--outputs-dir', type=str, default='./outputs')
    # 학습률
    parser.add_argument('--lr', type=float, default=1e-4)
    # 미니 배치 크기
    parser.add_argument('--batch-size', type=int, default=16)
    # 에포크 수 설정
    parser.add_argument('--num-epochs', type=int, default=400)
    # 4배 업스케일
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    
    # 추가된 인자: 시작 에포크와 체크포인트 파일 경로
    parser.add_argument('--start-epoch', type=int, default=0, help='학습을 시작할 에포크 번호')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 파일 경로')

    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # MPS 또는 CPU 디바이스 설정
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    # 모델 정의 및 디바이스로 이동
    model = SRCNN().to(device)

    # 체크포인트에서 모델 가중치 로드 (추가된 부분)
    if args.resume:
        print(f"체크포인트에서 학습 재개: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 데이터 로더 정의
    train_dataset = TrainDataset(args.train_dir, is_train=1, scale=args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = TestDataset(args.test_dir, scale=args.scale)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # 학습 루프: 시작 에포크부터 시작하도록 수정됨
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description(f'Epoch: {epoch}/{args.num_epochs - 1}')

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                t.update(len(inputs))

        torch.save(model.state_dict(), f"{args.outputs_dir}/epoch_{epoch}.pth")

        model.eval()
        epoch_psnr = AverageMeter()

        if (epoch + 1) % 5 == 0:
            epoch_psnr_values = []

            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)
                    psnr_value = calc_psnr(preds, labels).item()
                    epoch_psnr_values.append(psnr_value)

            epoch_psnr_avg = sum(epoch_psnr_values) / len(epoch_psnr_values)
            print(f'Eval PSNR: {epoch_psnr_avg:.2f}')

            if epoch_psnr_avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr_avg
                best_weights = copy.deepcopy(model.state_dict())

    print(f'Best Epoch: {best_epoch}, PSNR: {best_psnr:.2f}')
    
    # 최적의 모델 저장
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

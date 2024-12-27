import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리 추가

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from datasets import TestDataset
from torch.utils.data.dataloader import DataLoader

# 입력/예측/정답 이미지를 비교하고 저장하는 함수
def compare_images(input_image, predicted_image, ground_truth_image, index, save_dir="comparisons"):
    """
    input_image: 입력 영상 (저해상도)
    predicted_image: 모델이 복원한 예측 영상
    ground_truth_image: 고해상도 정답 영상
    index: 이미지 번호
    save_dir: 비교 이미지를 저장할 디렉토리 경로
    """
    os.makedirs(save_dir, exist_ok=True)  # 저장 디렉토리 생성

    plt.figure(figsize=(15, 5))

    # 입력 영상 (저해상도)
    plt.subplot(1, 3, 1)
    plt.title("Input (Low-Res)")
    plt.imshow(input_image, cmap="gray")
    plt.axis("off")

    # 예측 영상 (복원된 고해상도)
    plt.subplot(1, 3, 2)
    plt.title("Predicted (Super-Res)")
    plt.imshow(predicted_image, cmap="gray")
    plt.axis("off")

    # 정답 영상 (고해상도)
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth (High-Res)")
    plt.imshow(ground_truth_image, cmap="gray")
    plt.axis("off")

    # 결과 저장 또는 화면에 표시
    save_path = os.path.join(save_dir, f"comparison_{index}.png")
    plt.tight_layout()
    plt.savefig(save_path)  # 비교 이미지를 저장
    print(f"Saved comparison image at {save_path}")
    # plt.show()  # 주석 해제 시 화면에 표시 가능
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 가중치 파일 경로 (최고 성능을 갖는 모델을 불러옴)
    parser.add_argument('--weights-file', type=str, default='./outputs/epoch_0.pth')
    # 테스트 데이터셋 경로
    parser.add_argument('--test-dir', type=str, default='./db/test')
    # 출력 영상 경로
    parser.add_argument('--outimg-dir', type=str, default='outimg/')
    # 입력 영상 경로
    parser.add_argument('--orgimg-dir', type=str, default='orgimg/')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.orgimg_dir, exist_ok=True)
    os.makedirs(args.outimg_dir, exist_ok=True)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    loaded_state_dict = torch.load(args.weights_file, map_location='cpu')

    for n, p in loaded_state_dict.items():
        if n in state_dict.keys() and state_dict[n].size() == p.size():
            state_dict[n].copy_(p)
        else:
            print(f"Skipped loading parameter: {n} (size mismatch or not found in model)")

    model.load_state_dict(state_dict)

    test_dataset = TestDataset(args.test_dir, scale=args.scale)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model.eval()

    img_num = len(test_dataset.hr_list)
    psnr = torch.zeros(img_num)

    n = -1
    for data in test_dataloader:
        n += 1
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        psnr[n] = calc_psnr(preds, labels)
        print('{:d}/{:d} PSNR: {:.2f}'.format(n + 1, len(test_dataset), psnr[n]))

        preds_np = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)  # 예측 영상 (numpy 배열로 변환)

        # 원본 이미지 저장 (입력 이미지와 동일하게 처리)
        image = pil_image.open(test_dataset.hr_list[n]).convert('RGB')
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

        _, fname = os.path.split(test_dataset.hr_list[n])
        filename_orgimg = "{}/{}".format(args.orgimg_dir, fname)
        image.save(filename_orgimg)

        # YCbCr 변환 및 모델 출력 저장
        image_np = np.array(image).astype(np.float32)  # 입력 이미지 (numpy 배열로 변환)
        ycbcr = convert_rgb_to_ycbcr(image_np)

        output_np = np.array([preds_np, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output_np = np.clip(convert_ycbcr_to_rgb(output_np), 0.0, 255.0).astype(np.uint8)

        output_img = pil_image.fromarray(output_np)
        filename_outimg = "{}/{}".format(args.outimg_dir, fname)
        output_img.save(filename_outimg)

        # 정답 이미지 (numpy 배열로 변환)
        ground_truth_np = labels.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        # 입력/예측/정답 이미지를 비교하고 저장
        compare_images(image_np[..., 0], preds_np, ground_truth_np, n)

    mean_psnr = torch.mean(psnr)
    print('Total PSNR: {:.2f}'.format(mean_psnr))

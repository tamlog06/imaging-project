from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# 画像ファイル→PyTorchテンソル
def load_tensor(img_path):
    with Image.open(img_path) as img:
        array = np.asarray(img, np.float32).transpose([2, 0, 1]) / 255.0
        tensor = torch.as_tensor(np.expand_dims(array, axis=0))  # rank 4
    return tensor

# ピラミッドを1枚の画像に結合して保存するための関数
def tile_pyramid(imgs):
    height, width = imgs[0].size()[2:]
    with torch.no_grad():
        canvas = torch.zeros(1, 3, height * 3 // 2, width)
        x, y = 0, 0
        for i, img in enumerate(imgs):
            h, w = img.size()[2:]
            canvas[:,:, y:(y + h), x:(x + w)] = img            
            if i % 2 == 0:
                x += width // (2 ** (i + 3))                    
                y += height // (2 ** i) # 0, 2, 4..でy方向にシフト
            else:
                x += width // (2 ** i)  # 1, 3, 5..でx方向にシフ
                y += height // (2 ** (i + 3))
        return canvas

# ガウシアンぼかしのカーネル
def get_gaussian_kernel():
    # [out_ch, in_ch, .., ..] : channel wiseに計算
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5))
    return gaussian_k

def pyramid_down(image):
    with torch.no_grad():
        gaussian_k = get_gaussian_kernel()        
        # channel-wise conv(大事)
        multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2) for i in range(3)]
        down_image = torch.cat(multiband, dim=1)
    return down_image

def pyramid_up(image):
    with torch.no_grad():
        gaussian_k = get_gaussian_kernel()
        upsample = F.interpolate(image, scale_factor=2)
        multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2) for i in range(3)]
        up_image = torch.cat(multiband, dim=1)
    return up_image

def gaussian_pyramid(original, n_pyramids):
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):
        x = pyramid_down(x)
        pyramids.append(x)
    return pyramids

def laplacian_pyramid(original, n_pyramids):
    # gaussian pyramidを作る
    pyramids = gaussian_pyramid(original, n_pyramids)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1])
        laplacian.append(diff)
    return laplacian


def normalize_pyramids(pyramids):
    result = []
    for diff in pyramids:
        diff_min = torch.min(diff)
        diff_max = torch.max(diff)
        diff_normalize = (diff - diff_min) / (diff_max - diff_min)
        result.append(diff_normalize)
    return result

# 見やすいようにMin-Maxでスケーリングする# ラプラシアンブレンディング
def laplacian_blending(img1, img2):
    img1 = load_tensor(img1)
    img2 = load_tensor(img2)
    img1_lap = laplacian_pyramid(img1, 5)
    img2_lap = laplacian_pyramid(img2, 5)

    # ラプラシアンピラミッドを左右にブレンドする
    blend_lap = []
    for x, y in zip(img1_lap, img2_lap):
        # width = x.size(3)
        # b = torch.cat([x[:,:,:,:width // 2], y[:,:,:, width // 2:]], dim=3)
        # blend_lap.append(b)

        hb, wb = x.size()[2:]
        hf, wf = y.size()[2:]
        # print(h, w)
        b = torch.cat([x[:, :, hb//4:hb//4+hf//2, wb//4:wb//4+wf//2], y[:, :, :, :]], dim=3)
        blend_lap.append(b)

    # 最高レベルのガウシアンピラミッドのブレンド
    img1_top = gaussian_pyramid(img1, 5)[-1]
    img2_top = gaussian_pyramid(img2, 5)[-1]
    # out = torch.cat([img1_top[:,:,:,:img1_top.size(3) // 2],
                     # img2_top[:,:,:, img2_top.size(3) // 2:]], dim=3)

    out = torch.cat([img1_top[:, :, hb//4:hb//4+hf//2, wb//4:wb//4+wf//2], img2_top[:, :, :, :]], dim=3)

    # ラプラシアンピラミッドからの再構築
    for lap in blend_lap[::-1]:
        out = pyramid_up(out) + lap
    torchvision.utils.save_image(out, "laplacian_blend.png")

    # 比較例：ダイクトにブレンド
    # direct = torch.cat([img1[:,:,:,:img1.size(3) // 2],
                        # img2[:,:,:, img2.size(3) // 2:]], dim=3)
    direct = torch.cat([img1[:, :, hb//4:hb//4+hf//2, wb//4:wb//4+wf//2], img2[:, :, :, :]], dim=3)
    torchvision.utils.save_image(direct, "direct_blend.png")

if __name__ == "__main__":
    original = load_tensor("../image/train.jpg")
    pyramid = gaussian_pyramid(original, 6)
    torchvision.utils.save_image(tile_pyramid(pyramid), "gaussian_pyramid.jpg")
    normalize = normalize_pyramids(pyramid)
    torchvision.utils.save_image(tile_pyramid(normalize), "laplacian_pyramid.jpg")
    img1 = "../image/cropped.png"
    img2 = "../image/sample.png"
    laplacian_blending(img1, img2)

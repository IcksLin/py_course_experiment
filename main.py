import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(path):
    img = cv2.imread(path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, img


def manual_3x3_conv(img, kernel):
    H, W = img.shape
    img_float = img.astype(np.float32)

    output = np.zeros((H - 2, W - 2), dtype=np.float32)
    for i in range(H - 2):
        for j in range(W - 2):
            patch = img_float[i:i + 3, j:j + 3]
            output[i, j] = np.sum(patch * kernel)
    return output


def advanced_edge_enhancement(img, kernel_type='sobel'):
    kernels = {
        'sobel': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
        'prewitt': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    }

    kernel = kernels.get(kernel_type, kernels['sobel'])

    # 1. 边缘检测
    edges = manual_3x3_conv(img, kernel)
    edges_abs = np.abs(edges)
    edges_norm = cv2.normalize(edges_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # # 2. 二值化
    # edges_binary = cv2.adaptiveThreshold(edges_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY, 11, 2)

    # # 3. 形态学操作
    # kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cleaned_edges = cv2.morphologyEx(edges_norm, cv2.MORPH_OPEN, kernel_3x3)
    #
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # enhanced_edges = cv2.dilate(cleaned_edges, kernel_dilate, iterations=2)

    return {
        'original_edges': edges_norm,
        # 'binary_edges': edges_binary,
        # 'cleaned_edges': cleaned_edges,
        # 'enhanced_edges': enhanced_edges
    }


def visualize_enhancement_results(original, results_dict):

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # 原始图像
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 边缘检测结果
    axes[1].imshow(results_dict['original_edges'], cmap='gray')
    axes[1].set_title('Edge Detection')
    axes[1].axis('off')

    # # 二值化边缘
    # axes[0, 2].imshow(results_dict['binary_edges'], cmap='gray')
    # axes[0, 2].set_title('Binary Edges')
    # axes[0, 2].axis('off')

    # # 清理后的边缘
    # axes[1, 0].imshow(results_dict['cleaned_edges'], cmap='gray')
    # axes[1, 0].set_title('Cleaned Edges')
    # axes[1, 0].axis('off')
    #
    # # 增强后的边缘
    # axes[1, 1].imshow(results_dict['enhanced_edges'], cmap='gray')
    # axes[1, 1].set_title('Enhanced Edges')
    # axes[1, 1].axis('off')

    # # 空子图
    # axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 读取图像
    gray, color_img = read_image("asset/test2.jpg")

    # 检查图像是否成功读取
    if gray is None:
        print("错误：无法读取图像文件，请检查文件路径")
        gray = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        print("使用随机生成的测试图像")

    print(f"图像尺寸: {gray.shape}")
    print(f"图像数据类型: {gray.dtype}")

    for kernel_type in ['sobel', 'laplacian', 'prewitt']:
        print(f"\n使用 {kernel_type} 卷积核:")
        result = advanced_edge_enhancement(gray, kernel_type=kernel_type)
        visualize_enhancement_results(gray, result)
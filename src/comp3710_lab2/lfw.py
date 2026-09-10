from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import ensure_dir, write_json

#PCA主成分分析，会寻找数据变化最明显的方向，并把人脸压缩到更少的维度
#函数作用从 scikit-learn 加载 LFW 人脸数据集，同时筛选人物并缩小图片尺寸。
def load_lfw(data_home: str | Path | None = None, min_faces: int = 70, resize: float = 0.4):
    from sklearn.datasets import fetch_lfw_people

    # min_faces=70 只保留样本足够多的人物，避免类别太稀疏。
    # resize=0.4 把图片缩小，既保留脸部结构，又让 PCA/CNN 训练速度更适合 demo。
    kwargs = {"min_faces_per_person": min_faces, "resize": resize}
    if data_home is not None:
        kwargs["data_home"] = str(data_home)
    return fetch_lfw_people(**kwargs)

#作用是把多张图片排列成网格并保存，只负责保存图片。
def plot_gallery(images, titles, h: int, w: int, path: str | Path, n_row: int = 3, n_col: int = 4):
    import matplotlib.pyplot as plt

    # 这个函数把一组向量重新 reshape 成灰度图片，用于保存 eigenfaces 或预测示例。
    path = Path(path)
    ensure_dir(path.parent)
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    fig.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(min(n_row * n_col, len(images))):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        ax.set_title(titles[i], size=12)
        ax.set_xticks(())
        ax.set_yticks(())
    fig.savefig(path, dpi=160)
    plt.close(fig)

#负责从加载 LFW 到输出准确率的完整流程
def run_eigenfaces(
        #load parameter
    output_dir: str | Path,
    data_home: str | Path | None = None,
    n_components: int = 150,
    n_estimators: int = 150,
    max_depth: int = 15,
    seed: int = 42,
    min_faces: int = 70,
    resize: float = 0.4,
) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    total_start = time.perf_counter()
    output_dir = ensure_dir(output_dir)
    lfw_people = load_lfw(data_home=data_home, min_faces=min_faces, resize=resize)
    n_samples, h, w = lfw_people.images.shape
    # Part 2 的传统方法先把每张脸展平成向量，再做 PCA。
    # CNN 直接吃二维图像；Eigenfaces 方法则先把 HxW 图片拉平成长度为 H*W 的特征向量。
    x = lfw_people.data.astype(np.float64)
    y = lfw_people.target
    target_names = lfw_people.target_names

    # stratify=y 保证训练集/测试集中每个人物类别比例接近，避免小类被随机分没了。
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )

    # PCA 需要先减去训练集均值，让主成分学习“脸部变化方向”而不是整体亮度偏移。
    # 注意测试集也只能减训练集的 mean，不能用测试集信息，避免数据泄漏。
    mean = np.mean(x_train, axis=0)
    x_train_centered = x_train - mean
    x_test_centered = x_test - mean

    # SVD 得到主成分；这些主成分 reshape 回图片后就是 eigenfaces。
    # v 的每一行都是一个 principal component，代表数据方差最大的方向。
    pca_start = time.perf_counter()
    u, s, v = np.linalg.svd(x_train_centered, full_matrices=False)
    n_components = min(n_components, v.shape[0])
    components = v[:n_components]
    eigenfaces = components.reshape((n_components, h, w))

    # 把原始高维像素投影到 eigenface 空间，分类器只需要处理 n_components 维特征。
    x_train_transformed = np.dot(x_train_centered, components.T)
    x_test_transformed = np.dot(x_test_centered, components.T)
    pca_seconds = time.perf_counter() - pca_start

    eigenface_titles = [f"eigenface {i}" for i in range(n_components)]
    plot_gallery(
        eigenfaces,
        eigenface_titles,
        h,
        w,
        output_dir / "part2_eigenfaces.png",
    )

    explained_variance = (s**2) / (x_train_centered.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    ratio_cumsum = np.cumsum(explained_variance_ratio)
    # compactness 图说明前 N 个 eigenfaces 已经解释了多少训练集方差。
    # 这是 Part 2 里证明 PCA 起到压缩作用的重要输出。
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(n_components), ratio_cumsum[:n_components])
    ax.set_title("PCA compactness")
    ax.set_xlabel("Number of eigenfaces")
    ax.set_ylabel("Cumulative explained variance")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "part2_compactness.png", dpi=160)
    plt.close(fig)

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=min(150, n_components),
        random_state=seed,
        n_jobs=-1,
    )
    # Random Forest 在 PCA 特征上分类，和 Part 3.1 的端到端 CNN 形成对比。
    # 这里没有反向传播，属于传统机器学习 pipeline：手工特征降维 + 分类器。
    classifier_start = time.perf_counter()
    estimator.fit(x_train_transformed, y_train)
    classifier_training_seconds = time.perf_counter() - classifier_start
    inference_start = time.perf_counter()
    predictions = estimator.predict(x_test_transformed)
    inference_seconds = time.perf_counter() - inference_start
    correct = predictions == y_test
    report_text = classification_report(
        y_test, predictions, target_names=target_names, zero_division=0
    )
    (output_dir / "part2_classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_test, predictions, target_names=target_names, zero_division=0, output_dict=True
    )
    summary = {
        # metrics.json 记录数据形状、类别数、准确率和 report，方便之后不用重新训练也能汇报。
        "n_samples": int(n_samples),
        "height": int(h),
        "width": int(w),
        "n_features": int(x.shape[1]),
        "n_classes": int(target_names.shape[0]),
        "n_components": int(n_components),
        "train_shape": list(x_train_transformed.shape),
        "test_shape": list(x_test_transformed.shape),
        "total_test": int(len(y_test)),
        "total_correct": int(np.sum(correct)),
        "accuracy": float(np.mean(correct)),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "pca_seconds": float(pca_seconds),
        "classifier_training_seconds": float(classifier_training_seconds),
        "inference_seconds": float(inference_seconds),
        "seconds": float(time.perf_counter() - total_start),
        "target_names": [str(name) for name in target_names],
        "classification_report": report_dict,
    }
    write_json(output_dir / "part2_metrics.json", summary)
    return summary


class LfwCnn:
    """Factory wrapper to delay importing torch until needed."""

    @staticmethod
    def build(height: int, width: int, n_classes: int):
        import torch.nn as nn

        # 两次 MaxPool2d 会让高和宽各缩小 4 倍，所以全连接层输入维度要按 pooled_h/w 计算。
        pooled_h = height // 4
        pooled_w = width // 4
        return nn.Sequential(
            # 任务要求：两层 3x3 convolution，每层 32 filters。
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * pooled_h * pooled_w, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

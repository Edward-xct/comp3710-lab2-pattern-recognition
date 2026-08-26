from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import ensure_dir, write_json


def load_lfw(data_home: str | Path | None = None, min_faces: int = 70, resize: float = 0.4):
    from sklearn.datasets import fetch_lfw_people

    kwargs = {"min_faces_per_person": min_faces, "resize": resize}
    if data_home is not None:
        kwargs["data_home"] = str(data_home)
    return fetch_lfw_people(**kwargs)


def plot_gallery(images, titles, h: int, w: int, path: str | Path, n_row: int = 3, n_col: int = 4):
    import matplotlib.pyplot as plt

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


def run_eigenfaces(
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

    output_dir = ensure_dir(output_dir)
    lfw_people = load_lfw(data_home=data_home, min_faces=min_faces, resize=resize)
    n_samples, h, w = lfw_people.images.shape
    x = lfw_people.data.astype(np.float64)
    y = lfw_people.target
    target_names = lfw_people.target_names

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )

    mean = np.mean(x_train, axis=0)
    x_train_centered = x_train - mean
    x_test_centered = x_test - mean

    u, s, v = np.linalg.svd(x_train_centered, full_matrices=False)
    n_components = min(n_components, v.shape[0])
    components = v[:n_components]
    eigenfaces = components.reshape((n_components, h, w))

    x_train_transformed = np.dot(x_train_centered, components.T)
    x_test_transformed = np.dot(x_test_centered, components.T)

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
    estimator.fit(x_train_transformed, y_train)
    predictions = estimator.predict(x_test_transformed)
    correct = predictions == y_test
    report_text = classification_report(
        y_test, predictions, target_names=target_names, zero_division=0
    )
    (output_dir / "part2_classification_report.txt").write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_test, predictions, target_names=target_names, zero_division=0, output_dict=True
    )
    summary = {
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

        pooled_h = height // 4
        pooled_w = width // 4
        return nn.Sequential(
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

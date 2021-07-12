import torch


def generate_batches(y_batch):
    y_batch_circle = torch.tensor([1 if i in [0, 2, 6, 8, 9] else 0 for i in y_batch.tolist()], dtype=torch.float32)
    y_batch_curve = torch.tensor([1 if i in [2, 3, 5, 6, 9] else 0 for i in y_batch.tolist()], dtype=torch.float32)
    y_batch_line = torch.tensor([1 if i in [1, 2, 4, 5, 7] else 0 for i in y_batch.tolist()], dtype=torch.float32)

    return y_batch_circle, y_batch_curve, y_batch_line


def split_batches(x, y_pred):
    X_circle, X_curve, X_line, y_circle, y_curve, y_line = [], [], [], [], [], []
    for i, y in enumerate(y_pred):
        if y in [0, 2, 6, 8, 9]:
            X_circle.append(x[i])
            y_circle.append(y)
        if y in [2, 3, 5, 6, 9]:
            X_curve.append(x[i])
            y_curve.append(y)
        if y in [1, 2, 4, 5, 7]:
            X_line.append(x[i])
            y_line.append(y)

    return torch.stack(y_circle + y_curve + y_line), \
           torch.stack(X_circle) if len(X_circle) > 0 else None, \
           torch.stack(X_curve) if len(X_curve) > 0 else None, \
           torch.stack(X_curve) if len(X_curve) > 0 else None

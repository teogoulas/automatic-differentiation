import torch


def generate_batches(y_batch):
    y_batch_circle = torch.tensor([1 if i in [0, 2, 6, 8, 9] else 0 for i in y_batch.tolist()], dtype=torch.float32)
    y_batch_curve = torch.tensor([1 if i in [2, 3, 5, 6, 9] else 0 for i in y_batch.tolist()], dtype=torch.float32)
    y_batch_line = torch.tensor([1 if i in [3, 5, 6, 9] else 0 for i in y_batch.tolist()], dtype=torch.float32)

    return y_batch_circle, y_batch_curve, y_batch_line


def split_batches(x, y_pred):
    X_batch_circle = []
    X_batch_curve = []
    X_batch_line = []
    for i, y in enumerate(y_pred):
        predicted_digit = torch.argmax(y)
        if predicted_digit in [0, 2, 6, 8, 9]:
            X_batch_circle.append(x[i])
        if predicted_digit in [2, 3, 5, 6, 9]:
            X_batch_curve.append(x[i])
        if predicted_digit in [1, 2, 4, 5, 7]:
            X_batch_line.append(x[i])

    return torch.stack(X_batch_circle) if len(X_batch_circle) > 0 else None, \
           torch.stack(X_batch_curve) if len(X_batch_curve) > 0 else None, \
           torch.stack(X_batch_line) if len(X_batch_line) > 0 else None



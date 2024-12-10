import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from itertools import product
from matplotlib.pyplot import Axes
from matplotlib.patches import Rectangle
from torch import Tensor

from models.yolo import DetectionModel
from utils.dataloaders import create_dataloader
from utils.general import colorstr, labels_to_class_weights
from utils.loss import ComputeLoss

OBJECTNESS = 4
DIMS_PER_BOX = 5


def draw_box(
        ax: Axes,
        xy: tuple[float, float],
        wh: tuple[float, float],
        imgsz: tuple[int, int],
        box_type: str = 'cell',
):
    """

    Parameters
    ----------
    ax:
        matplotlib axis
    xy:
        normalized 0-to-1 xy coordinates of upper-left of box
    wh:
        normalized 0-to-1 wh size of box
    imgsz:
        image size as (rows, cols)
    box_type:
        choices ('obj', 'cell')

    Returns
    -------
    Draws a rectangle to the given axis
    """

    # bounding box around object
    if box_type == 'obj':
        edgecolor = 'magenta'
    else:  # cell
        edgecolor = 'cyan'

    rows, cols = imgsz

    rect = Rectangle(
        xy=[xy[0] * cols, xy[1] * rows],
        width=wh[0] * cols,
        height=wh[1] * rows,
        linewidth=1,
        edgecolor=edgecolor,  # 'red' if b == 0 else 'magenta',
        facecolor='none',
    )
    ax.add_patch(rect)


def visualize_batch_yolov1(
        imgs: Tensor,
        targets: Tensor,
        label_annotations: list[str],
):
    """
    Visualize the data and annotations from a Yolo V1 object detection dataset. This also helps determine how the cell
    size is affecting the normalization.

    Parameters
    ----------
    imgs:
        (N, 3, height, width) tensor of RGB images scaled from [0.0, 1.0].
            N: batch size
    targets:
        (N, S, S, B * 5 + C) tensor of truth boxes and classification labels.txt
            N: batch size
            S: number of cells vertically
            S: number of cells horizontally
            B: number of boxes per cell
            C: number of classes
    label_annotations:
        list of C label names
    """

    # extract dimensions from inputs
    batch_size = imgs.size(0)  # N
    num_cells = targets.size(1)  # S
    num_classes = len(label_annotations)  # C
    num_boxes = (targets.size(3) - num_classes) // DIMS_PER_BOX  # B

    # for each image
    for n in range(batch_size):
        rows = imgs.size(2)
        cols = imgs.size(3)
        fig, ax = plt.subplots()
        plt.imshow(imgs[n, :, :, :].permute(1, 2, 0))

        # cell size
        cell_width = 1 / num_cells
        cell_height = 1 / num_cells

        # for each cell(i, j) and box(b) within cell
        for i, j, b in product(range(num_cells), range(num_cells), range(num_boxes)):
            if targets[n, i, j, b * DIMS_PER_BOX + OBJECTNESS] <= 0:
                # skip truths with zero objectness
                continue

            # cell upper-left location
            cell_ulx = j / num_cells
            cell_uly = i / num_cells
            draw_box(ax, xy=(cell_ulx, cell_uly), wh=(cell_width, cell_height), imgsz=(rows, cols))

            # extract truth bounding box
            box_index = b * DIMS_PER_BOX
            x, y, w, h = targets[n, i, j, box_index:box_index + OBJECTNESS]

            # rescale x and y (?)
            x_prime = x * cell_width + cell_ulx
            y_prime = y * cell_height + cell_uly

            # center xy to upper-left xy
            ulx = (x_prime - w / 2.0)
            uly = (y_prime - h / 2.0)
            draw_box(ax, xy=(ulx, uly), wh=(w, h), imgsz=(rows, cols), box_type='obj')

            # class label
            label = label_annotations[
                torch.where(targets[n, i, j, num_boxes * DIMS_PER_BOX:] > 0.5)[0].item()
            ]
            ax.text(ulx * cols, uly * rows, label, color='magenta')

        plt.axis('equal')
        plt.show()
        if True:
            print()


def visualize_batch_yolov5(
        imgs: Tensor,
        targets: Tensor,
        label_annotations: list[str],
):
    """
    Visualize the data and annotations from a Yolo V5 object detection dataset. This also helps determine how the cell
    size is affecting the normalization.

    Parameters
    ----------
    imgs:
        (N, 3, height, width) tensor of RGB images scaled from [0.0, 1.0].
            N: batch size
    targets:
        (V, 6) tensor of truth boxes
    label_annotations:
        list of C label names
    """

    # extract dimensions from inputs
    batch_size = imgs.size(0)  # N

    # for each image
    for n in range(batch_size):
        rows = imgs.size(2)
        cols = imgs.size(3)
        fig, ax = plt.subplots()
        plt.imshow(imgs[n, :, :, :].permute(1, 2, 0))

        if len(targets) == 0:
            continue

        print(targets.shape)

        # for each cell(i, j) and box(b) within cell
        for v, (image, label_index, x, y, w, h) in enumerate(targets):
            if n != image:
                continue
            # center xy to upper-left xy
            ulx = (x - w / 2.0)
            uly = (y - h / 2.0)
            draw_box(ax, xy=(ulx, uly), wh=(w, h), imgsz=(rows, cols), box_type='obj')

            # class label
            label = label_annotations[label_index.int().item()]
            ax.text(ulx * cols, uly * rows, label, color='magenta')

        plt.axis('equal')
        plt.show()
        # if False:
        #     print()


def visualize_image_yolov5(
        img: np.ndarray,
        targets: np.ndarray,
        label_annotations: list[str],
):
    """
    Visualize the data and annotations from a Yolo V5 object detection dataset. This also helps determine how the cell
    size is affecting the normalization.

    Parameters
    ----------
    img:
        (3, height, width) array of RGB images scaled from [0.0, 1.0].
            N: batch size
    targets:
        (V, 14) array of truth boxes
    label_annotations:
        list of C label names
    """

    rows = img.shape[1]
    cols = img.shape[2]
    fig, ax = plt.subplots()
    plt.imshow(img.transpose(1, 2, 0))

    if len(targets) == 0:
        return

    # pxy, pwh, _, pcls = targets[image_index, anchor_index, cell_y, cell_x].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
    # # Regression
    # pxy = pxy.sigmoid() * 2 - 0.5
    # pwh = (pwh.sigmoid() * 2) ** 2 * anchors[layer_i]
    # pbox = torch.cat((pxy, pwh), 1)  # predicted box

    # for each cell(i, j) and box(b) within cell
    for v, (x, y, w, h, conf, *cls) in enumerate(targets):

        # pxy = torch.tensor([x, y]).sigmoid() * 2 - 0.5
        # pwh = (torch.tensor([w, h]).sigmoid() * 2) ** 2 * (1.0 / 7.0)
        pxy = torch.tensor([x, y]).sigmoid() * 2 - 0.5
        pwh = (torch.tensor([w, h]).sigmoid() * 2) ** 2 * (1.0 / 20.0)

        x, y = pxy
        w, h = pwh

        # center xy to upper-left xy
        ulx = (x - w / 2.0)
        uly = (y - h / 2.0)
        draw_box(ax, xy=(ulx, uly), wh=(w, h), imgsz=(rows, cols), box_type='obj')

        # class label
        class_index = np.array(cls).argmax()
        label = label_annotations[class_index]
        ax.text(ulx * cols, uly * rows, label, color='magenta')

    plt.axis('equal')
    # plt.show(block=False)
    import time
    start = time.time() % 10000
    plt.savefig(f"detection_{start:.0f}.jpeg")
    plt.close()
    if True:
        pass


if __name__ == "__main__":
    na = 3  # number of anchors
    nc = 9  # number of classes
    nl = 3  # number of layers / scales
    bs = 16
    imgsz = 640

    # hyperparameters
    hyp = {'anchor_t': 4.0, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'copy_paste': 0.0, 'degrees': 0.0, 'fl_gamma': 0.0, 'fliplr': 0.5, 'flipud': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'iou_t': 0.2, 'lr0': 0.01, 'lrf': 0.01, 'mixup': 0.0, 'momentum': 0.937, 'mosaic': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'perspective': 0.0, 'scale': 0.5, 'shear': 0.0, 'translate': 0.1, 'warmup_bias_lr': 0.1, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'weight_decay': 0.0005}
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl * 2 # scale to image size and layers

    # class labels
    annotations = ['player', 'ref', 'ball', 'hoop', 'mid', 'circle', 'corner', 'baseline', 'r_area']

    # device
    device = torch.device('cpu')  # mps is much slower

    # ['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx',
    matplotlib.use('macosx')

    # model
    model = DetectionModel(cfg="models/yolov5l.yaml", ch=3, nc=nc, anchors=None).to(device)
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = annotations

    # Trainloader
    train_loader, dataset = create_dataloader(
        path='/Users/merrillmck/source/github/nbaPlayerAndFeatureDetection/data/nba1022/train/images',
        imgsz=640,
        batch_size=bs,
        stride=32,
        single_cls=False,
        hyp=hyp,
        augment=False,
        cache=True,
        rect=False,
        rank=-1,
        workers=8,
        image_weights=False,
        quad=False,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=0,
    )
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    compute_loss = ComputeLoss(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = smart_optimizer(model, name="SGD", lr=hyp["lr0"], momentum=hyp["momentum"], decay=hyp["weight_decay"])

    epochs = 10
    model.train()
    for e in range(epochs):
        optimizer.zero_grad()
        mloss = torch.zeros(3, device=device)  # mean losses
        for i, (imgs, targets, img_paths, what_is) in enumerate(train_loader):
            # pre-process images
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # forward
            preds = model(imgs)

            # loss and backpropagation
            loss, loss_items = compute_loss(preds, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

            # logging
            print(loss)
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            print(mloss)

            # for each image, visualize top detection at top layer
            for image_index, image_preds in enumerate(preds[0]):
                # image_preds: (3, 80, 80, 14)
                num_cells = image_preds.shape[1]
                image_preds_2 = image_preds.reshape(na * num_cells * num_cells, 5 + nc)

                # sort by object confidence
                image_preds_3 = image_preds_2[image_preds_2[:, 4].argsort(descending=True)]

                if image_index == 0:
                    img = imgs[image_index]
                    img_copy = img.detach().cpu().numpy()
                    image_preds_3_copy = image_preds_3[:2, :].detach().cpu().numpy()
                    visualize_image_yolov5(img_copy, image_preds_3_copy, annotations)

            # pred_2 = preds[0]  # (16, 3, 80, 80, 14)
            # pred_2 = pred_2.reshape(bs, na * pred_2.shape[2] * pred_2.shape[3], 5 + nc)  # (16, 3*80*80, 14)
            # pred_3 = pred_2[pred_2[:, 4].argsort(descending=True)]
            # pred_4 = pred_3[:, :2]

            # print(pred_2.shape)

            # visualization
            # visualize_batch_yolov5(imgs, pred_2, annotations)

    # for imgs, targets, img_paths, _ in train_loader:
    #     plot_images(imgs, targets)

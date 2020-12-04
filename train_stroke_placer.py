#from paint import *
import paint
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import datetime
from utils.tensorboard import TensorBoard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


date_and_time = datetime.datetime.now()
run_name = 'StrokePlacer_' + date_and_time.strftime("%m_%d__%H_%M_%S")
writer = TensorBoard('train_log/{}'.format(run_name))

from stroke_placement.stroke_placement import StrokePlacer

stroke_placer = StrokePlacer(pretrained=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(stroke_placer.parameters(), lr=0.001)

epoch = 0
# Redefine this function for training
def paint_layer_train(canvas, reference_image, r, T, curved, pix_diff_thresh=30):
    """
    Go through the pixels and paint a layer of strokes with a given radius

    args:
        canvas (np.array[width, height, 3]) : Current painting canvas 0-1 RGB
        reference_image (np.array[width, height, 3]) :  Reference image 0-255 RGB
        r (int) : Brush radius to use
        T (int) : Max attempts to generate strokes
        curved (bool) : Whether to use curved or straight brush strokes
    kwargs:
        # smooth_loss (int) : How many bad strokes before giving up
        pix_diff_thresh (int) : small value reduces likely hood of picking a dark color

    return:
        np.array[width, height, 3] : Painting in 0-1 RGB format
        List[Tuple 13] : List of strokes. x0,y0,x1,y1,x2,y2,r0,r1,opaq0,opaq1,B,G,R
    """
    S = []

    width, height, _ = canvas.shape

    # Keep track of where you've already painted
    already_painted = torch.zeros((width, height), dtype=torch.bool, device=device)

    d_losses = []

    ############
    global epoch
    train_loss_sum = 0.
    ################

    for i in range(T):
        # sum the error near (x,y)
        D = torch.sum(torch.abs(canvas*255. - reference_image), dim=2)

        # D = gaussian_filter(D, sigma=max(r//4, 2)) # Blur it

        D = D * (1-already_painted.float()) # Don't paint same are twice

        # Pick starting point where error is largest
        noise = torch.randn(D.shape[0], D.shape[1], device=device)*0.001
        x, y = np.unravel_index((torch.clamp(D, 0, pix_diff_thresh) + noise).argmax().cpu(), D.shape)

        color = reference_image[x,y,:] / 255.

        K = paint.make_stroke(x, y, r, reference_image, canvas)
        if not curved:
            K, robot_stroke = paint.curved_stroke_to_straight(K, color*255., r)
        s = 1 - paint.draw_spline_stroke(K, r, width=width, height=height)

        loss_before_stroke = torch.mean(torch.abs(reference_image - canvas*255.))

        canvas_hat = paint.apply_stroke(canvas, s, color)

        loss_after_stroke = torch.mean(torch.abs(reference_image - canvas_hat*255.))

        ############################################
        sample = stroke_placer.create_sample(reference_image, canvas, r, x, y, color)

        pred = stroke_placer(sample.unsqueeze(0))
        true = torch.from_numpy(np.asarray(robot_stroke[2:6], dtype=np.float32)).unsqueeze(0).to(device) / reference_image.shape[0]
        # print(pred, true, sep="\n")
        # print()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(pred, true)
        loss.backward()
        optimizer.step()

        writer.add_scalar('Train_loss_stroke', loss, epoch*T + i)

        train_loss_sum += loss
        ############################################

        stroke_loss = loss_after_stroke - loss_before_stroke

        # Reject strokes that increase loss
        if stroke_loss > 0:
            continue

        # Accept the stroke
        if not curved:
            S.append(robot_stroke)
        else:
            S.append(K)
        canvas = canvas_hat
        already_painted = already_painted | (s==1)
    ##################
    writer.add_scalar('Train_loss', train_loss_sum, epoch)
    writer.add_image('Train_Image', canvas.cpu().numpy()[:,:,::-1] * 255., epoch)
    print('Image:', epoch, 'Train Loss:', float(train_loss_sum), sep="\t")
    epoch += 1
    ###################
#         if len(S) % 50 == 0:
#             plt.imshow(canvas)
#             plt.show()
#     print(loss_after_stroke)
    return canvas, S

paint.paint_layer = paint_layer_train

data_dir = 'C:/Users/Peter/HumanoidPainter/data/img_align_celeba/'

for img_fn in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_fn)

    R = 6
    paint.paint_fn(img_path, [R], T=[800], n_colors=6, ignore_whites=True, output_dir='stroke_placer_training_output', w=200, h=200)

    stroke_placer.save()
    # break
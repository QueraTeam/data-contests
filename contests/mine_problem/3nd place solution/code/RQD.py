""" compute RQD by R boxes """
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms import functional as F

# MIN_PROP = 0.2
# MIN_PROP_S = 0.01
# MIN_LEN_S = 0.06
# MIN_LEN = 0.03


MIN_PROP = 0.2
MIN_PROP_S = 0.01
MIN_LEN_S = 0.06
MIN_LEN = 0.04


def compute(image, prediction, l_t, name=None):
    ind = torch.argsort(prediction[0]['scores'], descending=True)[0:45]
    prop = prediction[0]['scores'][ind].detach()
    labels = prediction[0]['labels'][ind].detach()
    boxes = prediction[0]['boxes'][ind].detach()

    fig, ax = plt.subplots()
    ax.imshow(image)

    print(prop)
    df = pd.DataFrame({'prop': prop, 'label': labels, 'x0': boxes[:, 0],
                       'x1': boxes[:, 2], 'y0': boxes[:, 1], 'y1': boxes[:, 3]})
    df['coy'] = df['y0'] + ((df['y1'] - df['y0']) / 2.0)
    df['cox'] = df['x0'] + ((df['x1'] - df['x0']) / 2.0)
    df['A'] = (df['y1'] - df['y0']) * (df['x1'] - df['x0'])
    df['delta'] = df['y1'] - df['y0']
    # df = df[df['prop'] >= MIN_PROP]
    df = df[(df['prop'] >= MIN_PROP) | (df['label'] == 2)]
    df = df[df['prop'] >= MIN_PROP_S]
    df = df[df['label'] != 3]

    w, h = image.size
    dr = h / 10.1
    dw = 1.1 / w
    r = h / 1.99
    delta = dr + (dr * 0.3)

    df['val'] = (df['x1'] - df['x0']) * dw
    df = df[(df['val'] >= MIN_LEN_S) | (df['label'] == 1)]
    df = df[df['delta'] <= delta]

    # for i in range(prop.shape[0]):
    #
    #     if prop[i].item() > MIN_PROP:
    #         x = round(boxes[i, 0].item())
    #         y = round(boxes[i, 1].item())
    #         dx = round(boxes[i, 2].item()) - round(boxes[i, 0].item())
    #         dy = round(boxes[i, 3].item()) - round(boxes[i, 1].item())
    #
    #         val = dx * dw
    #         if val >= MIN_LEN:
    #             color = 'g'
    #         else:
    #             color = 'r'
    #
    #         rect = patches.Rectangle((x, y), dx, dy, linewidth=1, edgecolor=color, facecolor='none')
    #         ax.add_patch(rect)

    sum_l = 0.0
    run_rqd = []
    kk = 0
    tl = []
    for _ in range(6):

        plt.plot([10, w - 10], [r, r], color='w', linewidth=1)
        r += dr

        lb, ub = r - dr, r
        c = df[df['coy'] > lb]
        c = c[ub > c['coy']].sort_values(by=['cox'])
        # print(c)
        for index, row in c.iterrows():

            # if row['label'] == 2:
            #     ov = df[df['x1'] >= row['x0']]
            #     ov = ov[ov['x1'] < row['x1']]
            #     ov = ov[ov['x0'] < row['x0']]
            #     ov = ov[ov['y0'] <= row['coy']]
            #     ov = ov[ov['y1'] >= row['coy']]
            #     ov = ov[ov['cox'] != row['cox']]
            #     ov = ov[ov['coy'] != row['coy']]
            #     ov = ov[ov['label'] == 2]
            #
            #     if len(ov) > 0:
            #         ov = ov.iloc[0]
            #         nx = ov['x1'] + 10
            #         if (row['x1'] - nx) >= 70:
            #             row['x0'] = nx
            #             row['cox'] = row['x0'] + ((row['x1'] - row['x0']) / 2.0)
            #         else:
            #             continue
            #
            # ov = df[df['x0'] <= row['cox']]
            # ov = ov[ov['x1'] >= row['cox']]
            # ov = ov[ov['y0'] <= row['coy']]
            # ov = ov[ov['y1'] >= row['coy']]
            # ov = ov[ov['cox'] != row['cox']]
            # ov = ov[ov['coy'] != row['coy']]
            # ov = ov[ov['val'] >= row['val']]
            # ov = ov[ov['label'] == row['label']]
            #
            # if len(ov) > 0:
            #     continue

            if row['label'] == 2:
                ov = df[df['x1'] > row['x0']]
                ov = ov[ov['x0'] <= row['x0']]
                ov = ov[ov['y0'] <= row['coy']]
                ov = ov[ov['y1'] >= row['coy']]
                ov = ov[ov['cox'] != row['cox']]
                ov = ov[ov['coy'] != row['coy']]
                ov = ov[ov['label'] == 2]

                if len(ov) > 0:
                    if len(ov) == 1:
                        ov = ov.iloc[0]
                    else:
                        print('kkkk', ov['x1'].idxmax(), len(ov))
                        # ov = ov.iloc[ov['x1'].idxmax()]
                        ov = ov.loc[ov['x1'].idxmax()]
                    nx = min(ov['x1'] + 10, row['x1'])
                    if (row['x1'] - nx) >= 40:
                        row['x0'] = nx
                        row['cox'] = row['x0'] + ((row['x1'] - row['x0']) / 2.0)
                    else:
                        continue
            else:
                ov = df[df['x0'] <= row['cox']]
                ov = ov[ov['x1'] >= row['cox']]
                ov = ov[ov['y0'] <= row['coy']]
                ov = ov[ov['y1'] >= row['coy']]
                ov = ov[ov['cox'] != row['cox']]
                ov = ov[ov['coy'] != row['coy']]
                ov = ov[ov['val'] >= row['val']]
                ov = ov[ov['label'] == row['label']]

                if len(ov) > 0:
                    continue

            if row['label'] == 1:
                color = 'r'
            else:
                val = (row['x1'] - row['x0']) * dw
                if val >= MIN_LEN:
                    color = 'g'
                else:
                    color = 'c'

            rect = patches.Rectangle((row['x0'], row['y0']), row['x1'] - row['x0'],
                                     row['y1'] - row['y0'], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            if row['label'] == 2:
                val = (row['x1'] - row['x0']) * dw
                # if val > 10:
                #     val -= (val * 0.1)
                if val >= MIN_LEN:
                    sum_l += val
                    kk += val
            else:
                if sum_l >= 0.1 and len(run_rqd) < len(l_t):
                    run_rqd.append(min(1.0, sum_l / l_t[len(run_rqd)]) * 100)
                else:
                    run_rqd.append(1.0)
                sum_l = 0

    if sum_l >= 0.1 and len(run_rqd) < len(l_t):
        run_rqd.append(min(1.0, sum_l / l_t[len(run_rqd)]) * 100)
    else:
        run_rqd.append(1.0)

    if name is not None:
        plt.savefig(f'det/{name}.png')

    return run_rqd, plt


def rqd(model):
    df = pd.read_excel('data/from-to-rqd.xlsx')
    out = pd.DataFrame({'RunId': df['RunId'], 'Prediction': [1] * len(df['RunId'])})
    names = [k.split('\\')[1] for k in glob.glob('data/test-rqd/*')]
    names = [k.split('.')[0] for k in names]

    for name in names:
        img = Image.open(f'data/test-rqd/{name}.jpg').convert('RGB')
        img_t = F.to_tensor(img)
        # img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = [img_t]
        prediction = model(x)

        l_t, res = [], []
        for index, row in df[df['RunId'].str.startswith(f'{name}-')].iterrows():
            l_t.append(row['to'] - row['from'])
            res.append([row['RunId'], 1])

        run_rqd, plt = compute(img, prediction, l_t, name)

        v = 3
        for i, k in enumerate(res):
            if i < len(run_rqd):
                v = run_rqd[i]
                if 25 >= v >= 0:
                    v = 1
                elif 50 >= v > 25:
                    v = 2
                elif 75 >= v > 50:
                    v = 3
                elif 90 >= v > 75:
                    v = 4
                else:
                    v = 5
            res[i][1] = v
            out.loc[df.RunId == res[i][0], 'Prediction'] = v

        # for i, v in enumerate(run_rqd):
        #     if i >= len(res):
        #         break
        #     if 25 >= v >= 0:
        #         v = 1
        #     elif 50 >= v > 25:
        #         v = 2
        #     elif 75 >= v > 50:
        #         v = 3
        #     elif 90 >= v > 75:
        #         v = 4
        #     else:
        #         v = 5
        #     res[i][1] = v
        #     out.loc[df.RunId == res[i][0], 'Prediction'] = v
        print(len(l_t), l_t)
        print(len(run_rqd), run_rqd)
        print(res)

        # try:
        #     plt.title(str(res))
        #     plt.show()
        # except:
        #     pass
        # M3-BH3301-15   - 5
        # M3-BH3301-17   - 5
        # M3-BH3301-19   - 5

    out.to_csv('output.csv', index=False)

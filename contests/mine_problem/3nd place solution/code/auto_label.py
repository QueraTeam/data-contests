import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from bbaug import policies
from random import shuffle

aug_policy = policies.policies_custom()
policy_container = policies.PolicyContainer(aug_policy)


def main():
    names = [k.split('\\')[1] for k in glob.glob('data/train/*')]
    lbx = pd.read_excel('data/label.xlsx')

    shuffle(names)
    for name in names:
        img = Image.open(f'data/train/{name}').convert('RGB')
        lbl = lbx[lbx['image_name'] == name]
        print(lbl)
        df = pd.DataFrame({'label': lbl['label_name'], 'x0': lbl['xmin'], 'x1': lbl['xmin'] + lbl['width'],
                           'y0': lbl['ymin'], 'y1': lbl['ymin'] + lbl['height']})
        df['coy'] = df['y0'] + ((df['y1'] - df['y0']) / 2.0)
        df['cox'] = df['x0'] + ((df['x1'] - df['x0']) / 2.0)
        df['A'] = (df['y1'] - df['y0']) * (df['x1'] - df['x0'])

        w, h = img.size
        dr = h / 10.1
        dw = 1.1 / w
        r = h / 1.99

        fig, ax = plt.subplots()
        ax.imshow(img)

        for _ in range(6):

            bx = 90

            # plt.plot([10, w - 10], [r, r], color='blue', linewidth=2)
            r += dr

            lb, ub = r - dr, r
            c = df[df['coy'] > lb]
            c = c[ub > c['coy']].sort_values(by=['cox'])
            # print(c)
            for index, row in c.iterrows():

                ov = df[df['x0'] <= row['cox']]
                ov = ov[ov['x1'] >= row['cox']]
                ov = ov[ov['y0'] <= row['coy']]
                ov = ov[ov['y1'] >= row['coy']]
                ov = ov[ov['cox'] != row['cox']]
                ov = ov[ov['coy'] != row['coy']]
                ov = ov[ov['A'] >= row['A']]
                ov = ov[ov['label'] != 'wood']

                if len(ov) > 0:
                    continue

                if row['label'] == 'wood':
                    color = 'r'
                else:
                    color = 'g'

                rect = patches.Rectangle((row['x0'], row['y0']), row['x1'] - row['x0'],
                                         row['y1'] - row['y0'], linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                if (row['x0'] - bx - 30) >= 120:
                    rect = patches.Rectangle((bx, lb), row['x0'] - bx - 30, dr, linewidth=2, edgecolor='c',
                                             facecolor='none')
                    ax.add_patch(rect)
                    lbx.loc[len(lbx)] = [name.lower(), 'empty', round(bx), round(lb), round(row['x0'] - bx - 30), round(dr), 0,
                                         0]
                bx = row['x1'] + 30

                # print(ov)
                # print(row['cox'], row['coy'])
                # input('l')

            if (w - 90 - bx) >= 120:
                rect = patches.Rectangle((bx, lb), w - 90 - bx, dr, linewidth=2, edgecolor='c', facecolor='none')
                ax.add_patch(rect)
                lbx.loc[len(lbx)] = [name.lower(), 'empty', round(bx), round(lb), round(w - 50 - bx), round(dr), 0, 0]
        plt.savefig(f'det/{name}.png')
    lbx.to_excel('labelE.xlsx', sheet_name='Sheet_name_1', index=False)


def aug():
    names = [k.split('\\')[1] for k in glob.glob('data/train/*')]
    lbx = pd.read_excel('data/label.xlsx')

    k = 0
    shuffle(names)
    for name in names:

        _img = Image.open(f'data/train/{name}').convert('RGB')

        boxes, labels = [], []
        image_df = lbx[lbx['image_name'] == name]
        for it in zip(image_df['label_name'], image_df['xmin'], image_df['ymin'], image_df['width'],
                      image_df['height']):
            boxes.append([it[1], it[2], it[1] + it[3], it[2] + it[4]])
            if it[0] == 'wood':
                labels.append(1)
            # elif it[0] != 'empty':
            #     labels.append(2)
            else:
                labels.append(2)

        error = False
        random_policy = policy_container.select_random_policy()
        img, temp = policy_container.apply_augmentation(random_policy, np.array(_img), boxes, labels)

        for i in temp:
            if (i[3] - i[1]) <= 10 or (i[4] - i[2]) <= 10:
                # img = F.to_tensor(_img)
                error = True
                break

        if not error:

            k += 1

            new_name = f'{k}_{name}'

            fig, ax = plt.subplots()
            ax.imshow(img)

            for i in boxes:
                rect = patches.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1], linewidth=1, edgecolor='g',
                                         facecolor='none')
                ax.add_patch(rect)

            boxes.clear()
            labels.clear()
            for i in temp:
                boxes.append([i[1], i[2], i[3], i[4]])
                labels.append(i[0])

                if i[0] == 1:
                    lb = 'wood'
                # elif i[0] == 2:
                #     lb = 'empty'
                else:
                    lb = '+10cm rock'
                lbx.loc[len(lbx)] = [new_name, lb, i[1], i[2], i[3] - i[1], i[4] - i[2], 0, 0]

                rect = patches.Rectangle((i[1], i[2]), i[3] - i[1], i[4] - i[2], linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)
            im = Image.fromarray(img)
            im.save(f'data/trainA/{new_name}')
        print(k, len(names), name)
        # plt.show()
        if k >= 90:
            break
    lbx.to_excel('data/labelA.xlsx', sheet_name='Sheet_name_1', index=False)


if __name__ == '__main__':
    # main()
    aug()

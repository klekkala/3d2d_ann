dates = []
dates.append('2012-01-08')
dates.append('2012-01-15')
dates.append('2012-01-22')
dates.append('2012-02-02')
dates.append('2012-02-04')
dates.append('2012-02-05')
dates.append('2012-02-12')
dates.append('2012-02-18')
dates.append('2012-02-19')
dates.append('2012-03-17')
dates.append('2012-03-25')
dates.append('2012-03-31')
dates.append('2012-04-29')
dates.append('2012-05-11')
dates.append('2012-05-26')
dates.append('2012-06-15')
dates.append('2012-08-04')
dates.append('2012-08-20')
dates.append('2012-09-28')
dates.append('2012-10-28')
dates.append('2012-11-04')
dates.append('2012-11-16')
dates.append('2012-11-17')
dates.append('2012-12-01')
dates.append('2013-01-10')
dates.append('2013-02-23')
dates.append('2013-04-05')

import sys
import matplotlib.pyplot as plt
import numpy as np
def main(args):

    gt=None
    for date in dates:
        temp_gt = np.loadtxt('./simplified/groundtruth_%s.csv' % (date), delimiter = ",")
        if gt is None:
            gt = temp_gt
        else:
            gt = np.concatenate((gt, temp_gt))


    x = gt[:, 1]
    y = gt[:, 2]
    z = gt[:, 3]


    plt.figure()
    plt.scatter(y, x, 1, c=-z, linewidth=0)    # Note Z points down
    plt.axis('equal')
    plt.title('Ground Truth Position of Nodes in SLAM Graph')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.colorbar()

    plt.savefig('test2.png')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

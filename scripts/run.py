import os
import argparse
from os import system


def runcmd(cmd):
    print('[cmd]', cmd)
    system(cmd)


def get_cp_path(args, alt):
    return '../checkpoint/{pre}{alt}-h{hid}-b{batch}-norm{lam}'.format(
            pre=args.prefix,
            alt=alt,
            hid=args.hid_dim,
            batch=args.batch,
            lam=args.lamb
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('dictionary generation')
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('-i', '--hid_dim', type=int, default=512)
    parser.add_argument('-c', '--basecheckpoint', type=str, required=True)

    parser.add_argument('-l', '--lamb', type=float, default=0.0)
    parser.add_argument('-b', '--batch', type=int, default=4)
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, default=10)
    args = parser.parse_args()
    system('cd ../src')

    outdir = '../output/'
    datadir = '../data/alt/'
    outpath = outdir + '{}-h{}.txt'.format(args.prefix, args.hid_dim)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    for alt in range(args.start, args.end):
        data = datadir + '{pre}{next_alt}-n{lamb}'.format(pre=args.prefix, next_alt=alt + 1, lamb=args.lamb)
        if alt == 0:
            rank_cp = args.basecheckpoint
        else:
            rank_cp = get_cp_path(args, alt)

        if not os.path.exists(data):
            cmd = 'python rank.py --data {basedata} -e {check} -g {gpu} -o {data} -l {lamb}'.format(basedata=args.data, check=rank_cp, data=data, gpu=args.gpu, lamb=args.lamb)
            runcmd(cmd)

        checkpoint = get_cp_path(args, alt + 1)

        cmd = 'python ../src/train.py --data {data} -g {gpu} -b {batch} -i {hid} -c {cp}'.format(hid=args.hid_dim, data=data, gpu=args.gpu, batch=args.batch, cp=checkpoint)
        runcmd(cmd)

        cmd = 'python rvd.py -c {check} -p 20 >> {outpath}'.format(gpu=args.gpu, outpath=outpath, check=checkpoint)
        runcmd(cmd)




import os
from os.path import join, basename, split, splitext
from os import makedirs
from glob import glob
from pyntcloud import PyntCloud
import pandas as pd
import argparse
import functools
from tqdm import tqdm
from multiprocessing import Pool
from utils.octree_partition import partition_octree

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def process(path,args):
    ori_path = join(args.source, path)
    target_path, _ = splitext(join(args.dest, path))
    target_path += (str(args.portion_points)+'rm'+args.target_extension)
    target_folder, _ = split(target_path)
    makedirs(target_folder, exist_ok=True)
    pc = PyntCloud.from_file(ori_path)
    pc.points = pc.points.astype('float64', copy=False)
    coords = ['x', 'y', 'z']
    points = pc.points[coords]
    points=points.sample(frac=(1-args.portion_points), axis=0)
    if(len(points)>0):
        cloud=PyntCloud(points)
        cloud.to_file(target_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--level', type=int, help='Number of level to downsample', default=1)
    parser.add_argument('--source_extension', help='Mesh files extension', default='.ply')
    parser.add_argument('--target_extension', help='Point cloud extension', default='.ply')
    parser.add_argument("-portion", '--portion_points', type=float,
                        default=1,
                        help='percent of point to be removed')

    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'

    paths = glob(join(args.source, '**', f'*{args.source_extension}'), recursive=True)
    files = [x[len(args.source) :] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    with Pool() as p:
        process_f = functools.partial(process, args=args)
        list(tqdm(p.imap(process_f, files), total=files_len))
    logger.info(f'{files_len} models written to {args.dest}')

#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 1/23/24 8:10 PM
import argparse
import os
import sys
sys.path.append('.')

from spring_amr.train_utils import compute_smatch

argparser = argparse.ArgumentParser()
argparser.add_argument('-p', '--pred', help='prediction file', required=True)
argparser.add_argument('-g', '--gold', help='ground truth file', required=True)
argparser.add_argument('--f_only', action='store_true', help='whether to only show F Score')
argparser.add_argument('--verbose', action='store_true', help='whether to set verbose=True for smatch')

def main():
  args = argparser.parse_args()
  print("Smatch with `smatch` package")

  assert os.path.exists(args.gold)
  print("Gold File: {}".format(args.gold))
  assert os.path.exists(args.pred)
  print("Pred File: {}".format(args.pred))

  p,r,f = compute_smatch(args.gold, args.pred, return_all=not args.f_only, verbose=args.verbose)
  print("Score:")
  print(f' Precision: {p:.5f}')
  print(f' Recall: {r:.5f}')
  print(f' F-Score: {f:.5f}')

  ### Jan 23, 2024
  # Score (LeakDistill on R3):
  #  Precision: 0.84965
  #  Recall: 0.83850
  #  F-Score: 0.84404

  # Score (LeakDistill with BLINK on R3):
  #  Precision: 0.85183
  #  Recall: 0.84065
  #  F-Score: 0.84620


if __name__ == '__main__':
  main()

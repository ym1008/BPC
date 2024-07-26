import os
import glob
import random
import argparse
import numpy as np
import soundfile as sf


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('root', help='root directory containing flac files')
	parser.add_argument('--outputDir', default='.', type=str, help='output directory')
	parser.add_argument('--fname', default='train.tsv', type=str, help='output filename')
	parser.add_argument('--sample-size', default=50000, type=int, help='input to pre-training')
	return parser


def compute_std(search_path, sample_size, mu, n):

	std = 0.0
	for fname in glob.iglob(search_path, recursive=True):
		file_path = os.path.realpath(fname)
		frames = sf.info(fname).frames
	
		if frames >= sample_size:
			data, _ = sf.read(fname)
			std = std + np.sum((data - mu)**2)	

	std = np.sqrt(std/n)
	return std



def main(args):

	dir_path = os.path.realpath(args.root)
	search_path = os.path.join(dir_path, '**/*.flac')
	
	mu = 0.0
	n = 0
	
	with open(os.path.join(args.outputDir, args.fname), 'w') as outfile:
		print(dir_path, file=outfile)

		for fname in glob.iglob(search_path, recursive=True):
			file_path = os.path.realpath(fname)
			frames = sf.info(fname).frames
			
			if frames >= args.sample_size:
				print('{}\t{}'.format(os.path.relpath(file_path, dir_path), frames), file=outfile)

				# compute mean
				data, _ = sf.read(fname)
				mu = mu + np.sum(data)
				n = n + frames

	mu = mu / n
	std = compute_std(search_path, args.sample_size, mu, n)

	print('{}\t{}'.format(mu, std))



if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()
	main(args)

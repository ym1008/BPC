import os
import copy
import torch
import augment
import argparse
import random
import numpy as np
from os.path import join



def add_args():
	parser = argparse.ArgumentParser()

	# data and data handling
	parser.add_argument('datatrain', help='tsv files with paths to each .flac data file for training')
	parser.add_argument('--sample-rate', default=16000, type=int, help='target sample rate. audio files will be up/down sampled to this rate')
	parser.add_argument('--sample-size', default=50000, type=int, help='sample size to crop to for batching.')
	parser.add_argument('--num-workers', default=8, type=int, help='Number of workers for dataloader')
	parser.add_argument('--prefetch', default=100, type = int, help='Number of workers for dataloader')
	parser.add_argument('--pin-memory', action='store_true', help='pin memory in dataloader')
	parser.add_argument('--no-input-norm', action='store_true', help='Turn off normalising input with mean and variance of each input sample if true, otherwise normalise with population statistics')
	parser.add_argument('--load-classifier', default=None, type=str, help='path to checkpoint for supervised downstream classifier')
	parser.add_argument('--load-chk-path', default=None, type=str, help='path to folder containing folder "checkpoints" with saved checkpoints')
	parser.add_argument('--load-chk-name', default='', type=str, help='name of saved checkpoint')
	parser.add_argument('--seed', default=0, type=int, help='seed')

	# logging and outputs
	parser.add_argument('--save-chk-steps', default=5000, type=int, help='step frequency to save parameters')
	parser.add_argument('--check-val-steps', default=100, type=int, help='step frequency to validate downstream model')
	parser.add_argument('--log-steps', default=50, type=int, help='step frequency to logg metrics (loss)')
	parser.add_argument('--log-dir', default='./Experiments', type=str, help='step frequency to logg metrics (loss)')
	parser.add_argument('--log-name', default='', type=str, help='step frequency to logg metrics (loss)')
	parser.add_argument('--log-version', default=None, help='step frequency to logg metrics (loss)')
	parser.add_argument('--log-subdir', default=None, help='step frequency to logg metrics (loss)')
	
	# pre-training model architecture
	parser.add_argument('--cnn', action='store_true')
	parser.add_argument('--lstm', action='store_true')
	parser.add_argument('--transformer', action='store_true')
	
	parser.add_argument('--conv-encoder-layers', default='[(512, 10, 5)] + [(512, 8, 4)] + [(512, 4, 2)] * 3', type=str, help='convolutional layers for encoder [(dim, kernel_size, stride), ...]')
	parser.add_argument('--conv-aggregator-layers', default='[(512, 3, 1)] * 9', type=str, help='convolutional layers for aggregator [(dim, kernel_size, stride), ...]')
	parser.add_argument('--skip-connections-enc', action='store_true', help='if set, adds skip connections to the encoder (feature extractor)')
	parser.add_argument('--skip-connections-agg', action='store_true', help='if set, adds skip connections to the aggregator')
	parser.add_argument('--log-compression', action='store_true', help='if set, log compression on encoder output')
	parser.add_argument('--no-bias-enc', action='store_true', help='if set, turns off bias in encoder conv layers')
	parser.add_argument('--padding-agg', action='store_true', help='if set, use padding in aggregator')
	parser.add_argument('--residual-scale', default=0.5, type=float, help='scales residual by sqrt(value)')
	
	parser.add_argument('--lstm-dim', default=128,type=int)
	parser.add_argument('--lstm-nlayers', default=3,type=int)
	
	parser.add_argument('--transformer-dim', default=512,type=int)
	parser.add_argument('--transformer-ffn', default=512,type=int)
	parser.add_argument('--transformer-nhead', default=8,type=int)
	parser.add_argument('--transformer-nlayers', default=12,type=int)
	parser.add_argument('--pos-conv-layers', default=5,type=int)
	parser.add_argument('--pos-conv-filter', default=95,type=int)
	parser.add_argument('--pos-conv-norm', default='ln_c',type=str)
	parser.add_argument('--dropout-transformer', default=0.1, type=float, help='dropout to apply to output of encoder')
	parser.add_argument('--layerdrop', default=0.0, type=float, help='dropout to apply within the model')
	
	parser.add_argument('--dropout', default=0.0, type=float, help='dropout to apply within the model')
	parser.add_argument('--activation', default='relu', type=str, help='activation to use throughout model. Transformer default = gelu')
	
	# try out different normalisation strategies at some point. Will need to edit this flag and the code... 
	parser.add_argument('--initialise', action='store_true', help='If true, uses kaimin_normal initialisation for conv weights in enc and agg')
	
	parser.add_argument('--norm-enc', default='in', type=str, help = '"bn", "in", "ln", or "ln_c"')
	parser.add_argument('--norm-agg', default='bn', type=str, help = '"bn", "in", "ln", or "ln_c"')
	parser.add_argument('--norm-pred', default='bn', type=str, help = '"bn", "in", "ln", or "ln_c"')
	parser.add_argument('--norm-target', default='in', type=str, help = '"bn", "in", "ln", or "ln_c"')
	parser.add_argument('--norm-online', default='other', type=str, help = '"bn", "in", "ln", or "ln_c"')
	parser.add_argument('--no-l2-target', action='store_true', help='disable l2 on target')
	parser.add_argument('--no-l2-online', action='store_true', help='disable l2 on online')
	

	# prediction
	parser.add_argument('--min-layer', default=-1, type=int, help='First layer to begin averaging from for target avg prediction')
	parser.add_argument('--max-layer', default=1, type=int, help='Last layer to begin averaging from for target avg prediction')
	parser.add_argument('--avg-layers', action='store_true', help='avg aggregator layers')
	parser.add_argument('--no-predictor', action='store_true', help='Turn off predictor')
	parser.add_argument('--output-layer', default='act', type=str, help='"conv", "norm" or "act"')
	parser.add_argument('--prediction-steps', default=1, type=int, help='Number of steps into future to predict')
	parser.add_argument('--prediction-dim', default=512, type=int, help='Dimension of frame predictor output (matching target)')
	parser.add_argument('--prediction-nlayer', default=1, type=int, help='Whether to have the first activation and normalisation block')
	

	# contrastive 
	parser.add_argument('--contrastive', action='store_true')
	parser.add_argument('--n-negatives', default=10, type=int, help='number of negative examples')
	parser.add_argument('--cross-sample-negatives', action='store_true', help='whether to sample negatives across examples in the same batch')


	# optimisation/learning
	parser.add_argument('--batch-size', default=128, type=int, help='batch size.')
	parser.add_argument('--tune-lr', action='store_true', help='tune learning rate')
	parser.add_argument('--lr', default=1e-3, type=float, help='max learning rate')
	parser.add_argument('--wd', default=0, type=float, help='adam optimiser weight decay')
	parser.add_argument('--eps', default=1e-8, type=float, help='adam optimiser epsilon')
	parser.add_argument('--beta1', default=0.9, type=float, help='adam optimiser beta1')
	parser.add_argument('--beta2', default=0.999, type=float, help='adam optimiser beta2')
	parser.add_argument('--warmup-steps', default=12000, type=int, help='number of updates to linearly increase lr to max value')
	parser.add_argument('--lr-cosine', action='store_true', help='Use cosine anealing schedule for LR')	
	parser.add_argument('--lr-cosine-max-steps', default=400000,type=int, help='Max steps for cosine annealing of LR')	
	parser.add_argument('--max-steps', default=40000, type=int, help='number of total gradient step updates to make')
	parser.add_argument('--max-epochs', default=-2, type=int, help='number of total gradient step updates to make')
	parser.add_argument('--update-grad-freq', default=1, type=int, help='accumulate gradients')
	parser.add_argument('--celoss', action='store_true', help='cross entropy loss instead of MSE default')
	parser.add_argument('--lamb', default=1, type=float, help='Scaling factor for predictor faster learning rate')
	parser.add_argument('--ema-aggregator-only', action='store_true', help='Ema of aggregator only')
	parser.add_argument('--ema-layers-only', action='store_true', help='Ema of transformer layers in aggregator only')
	parser.add_argument('--tau', default=0.999, type=float, help='starting decay rate. Increases to 1 during training following cosine annealing')
	parser.add_argument('--tau-max', default=0.9999, type=float, help='Max tau')
	parser.add_argument('--tau-anneal-steps', default=20000, type=int, help='Max steps for cosine annealing of tau')
	parser.add_argument('--anneal-fn', default='linear', type=str, help='Linear, cosine, constant')
	parser.add_argument('--precision', default='16-mixed', type=str, help='Precision of models')
	parser.add_argument('--no-syncBN', action='store_true', help='Turn off BN syncing across GPUS')
	parser.add_argument('--eman', action='store_true', help='EMAN on batchnorm in target network')

	
	# pre-text task and augmentations
	parser.add_argument('--offset', default=3, type=int, help='Offset between context in present and predicted context in future')
	parser.add_argument('--mask', action='store_true', help='activates time-masking in online network')	
	parser.add_argument('--mask-prob', default=0.65, type=float, help='Probability of span being masked: prob of each frame = prob/span')
	parser.add_argument('--mask-span', default=10, type=int, help='Span of frames to mask')
	parser.add_argument('--mask-sampling', action='store_true', help='randomly samples mask ids')	
	parser.add_argument('--mask-type', default=0, type=int, help='Type of mask/pretext')
	parser.add_argument('--pitch-past', action='store_true')
	parser.add_argument('--reverb-past', action='store_true')
	parser.add_argument('--add-past', action='store_true')
	parser.add_argument('--pitch-future', action='store_true')
	parser.add_argument('--reverb-future', action='store_true')
	parser.add_argument('--add-future', action='store_true')


	# downstream supervised task hyperparameters
	parser.add_argument('--datatest', help='tsv files with paths to each .flac data file for testing')
	parser.add_argument('--labelstrain', help='file with phone labels for training')
	parser.add_argument('--labelstest', help='file with phone labels for testing')
	parser.add_argument('--nsn', action='store_true', help='include NSN label')
	
	parser.add_argument('--nclasses', default=41, help='No. of phone classes')
	parser.add_argument('--bilstm-dim', default=256, type=int, help='Hidden dimension size of BiLSTM')
	parser.add_argument('--use-lstm', action='store_true', help='Lstm layer in sequence classifier')
	parser.add_argument('--ctc', action='store_true', help='use ctc sequence loss instead of frame loss')
	
	parser.add_argument('--finetune-all', action='store_true', help='train entire network')
	parser.add_argument('--finetune-agg', action='store_true', help='train aggregator (transformer or cnn) only')
	parser.add_argument('--freeze-first-nsteps', default=10000, type=int, help='How long to freeze network before fine-tuning')
	
	parser.add_argument('--use-z', action='store_true', help='use encoding, z')
	parser.add_argument('--use-target', action='store_true', help='use target network instead of online network')
	
	# pretraining model architecture
	parser.add_argument('--d2v-fs', action='store_true')
	parser.add_argument('--w2v2-fs', action='store_true')
	parser.add_argument('--fs-model', default='data2vec-audio-base', type=str, help='the model to download: [data2vec-audio][wav2vec]-[base][large][-960]')


	return parser.parse_args()


def	writeout_args(args, dir_path):

	if args.log_version not in dir_path:
		setattr(args, 'outputDir', os.path.join(dir_path, args.log_version))
	else:
		setattr(args, 'outputDir', dir_path)

	if not os.path.isdir(args.outputDir):
		os.mkdir(args.outputDir)
	
	opts =''.join(''.join(str(args).split('(')[1:]).split(')')[:-1]).replace(',','\n')
	f = open(join(args.outputDir, 'config.txt'),'w')
	f.write('Using {} GPUs \n'.format(torch.cuda.device_count()))
	f.write(opts)
	f.close()


def get_args():
	args = add_args()

	# for dataloader: get receptive field and striding of encoder
	jin = 1
	rin = 1
	for _, k, s in eval(args.conv_encoder_layers):
		rin = rin + (k - 1) * jin
		jin *= s

	setattr(args, 'rfs_enc', rin)
	setattr(args, 'stride_enc', jin)

	# identifying offset between encodings and context vectors due to no padding in aggregator cnn
	if args.cnn:
		jin = 1
		rin = 1
		for _, k, s in eval(args.conv_aggregator_layers):
			rin = rin + (k - 1) * jin
			jin *= s
		receptive_field = int(rin) - 1 
		setattr(args, 'receptive_field', receptive_field)
		if args.padding_agg:
			setattr(args, 'receptive_field', 0)			
	else:
		setattr(args, 'receptive_field', 0)

	
	if args.d2v_fs or args.w2v2_fs:
		setattr(args, 'rfs_enc', 400)
		setattr(args, 'stride_enc', 320)
		setattr(args, 'classifier_dim', 768)


	if args.num_workers == 0 :
		setattr(args, 'prefetch', None)

	if args.min_layer == -1:
		setattr(args, 'min_layer', None)
	
	if args.max_layer == -1:
		setattr(args, 'max_layer', None)

	if args.max_epochs == -2:
		setattr(args, 'max_epochs', None)

	if args.save_chk_steps == -1:
		setattr(args, 'save_chk_steps', None)

	return args



def compute_per(r,h):
	# https://martin-thoma.com/word-error-rate-calculation/
   
    d = np.zeros((len(r) + 1) * (len(h) + 1))
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return torch.tensor(d[len(r)][len(h)])


class ChainRunner:
	def __init__(self, chain):
		self.chain = chain

	def __call__(self, x):

		src_info = {'channels': x.size(0), 'length': x.size(1), 'rate': 16000.0}
		target_info = {'channels': 1, 'length': x.size(1), 'rate': 16000.0}

		y = self.chain.apply(x, src_info=src_info, target_info=target_info)

		if torch.isnan(y).any() or torch.isinf(y).any():
			y = x.clone()

		return y


def get_augmentations(args):

	random_pitch = lambda: np.random.randint(-300, +300)
	random_room_size = lambda: np.random.randint(0, 100)
	random_noise = lambda: torch.zeros((1, args.sample_size)).uniform_()

	chain_past = augment.EffectChain()
	chain_past.additive_noise(random_noise, snr = 15) if args.add_past else None
	chain_past.pitch(random_pitch).rate(args.sample_rate) if args.pitch_past else None
	chain_past.reverb(50, 50, random_room_size).channels(1) if args.reverb_past else None
	augment_past = ChainRunner(chain_past)

	chain_future = augment.EffectChain()
	chain_future.additive_noise(random_noise, snr = 15) if args.add_future else None
	chain_future.pitch(random_pitch).rate(args.sample_rate) if args.pitch_future else None
	chain_future.reverb(50, 50, random_room_size).channels(1) if args.reverb_future else None
	augment_future = ChainRunner(chain_future)

	return augment_past, augment_future



def get_state_dicts(chk, use_target = False):
	keys = []
	for i in chk:
		keys.append(i)

	for key in keys:
		key_split = key.split('pred')
		if len(key_split) == 2:
			del chk[key]

		else:
			key_split = key.split('ema_')

			if use_target:
				if len(key_split) == 2:
					
					# label of online layer
					key_new = key_split[1]
					
					# delete ema.n_averaged fields
					key_new_split = key_new.split('n_averaged')
					if len(key_new_split) == 2:
						del chk[key]   
					else:
						key_new_split = key_new.split('.module')
						key_new = ''.join(key_new_split)
						chk[key_new] = chk[key] # copy target to online and delete target
						del chk[key]
		
			else:
				# delete ema
				if len(key_split) == 2: 
					del chk[key]


	agg_chk = copy.deepcopy(chk) 

	keys = []
	for i in chk:
		keys.append(i)

	for key in keys:
		key_agg = key.split('agg.')
	
		if len(key_agg) == 2:
			agg_chk[key_agg[1]] = agg_chk[key]
		else:
			key_enc = key.split('enc.')

			if len(key_enc) == 2:
				chk[key_enc[1]] = chk[key]
				
		
		del chk[key]
		del agg_chk[key]
	

	return chk, agg_chk



def get_mask_ids(shape, mask_prob, mask_span, rfs, typeM = 0, sampling = False):

	# mask type
	# 0 : c vectors that have any z masked
	# 1 : c vectors with first half or second half of z frames masked (9 frames)
	# 2 : c vectors predicting final z frame if masked

	bsz, tsz = shape
	mask = torch.full((bsz, tsz), False)

	p = mask_prob
	M = mask_span
	num_spans = int(p * tsz / float(M) + np.random.rand())
	span_lengths = np.full(num_spans, M, dtype=int)

	mask_ids = []
	for i in range(bsz):
		ids = np.random.choice(tsz - min(span_lengths), num_spans, replace = False)  # starting indices of mask spans
		ids = np.asarray([ 
			ids[j] + m_count for j in range(num_spans)
				for m_count in range(int(span_lengths[j]))
			])                
		mask_ids.append(np.unique(ids[ids < tsz]))

	min_mask_len = min([len(m_id) for m_id in mask_ids])
	for i, mask_id in enumerate(mask_ids):
		if len(mask_id) > min_mask_len:
			startid = random.randint(0, len(mask_id) - min_mask_len) # random cropping 
			if sampling:
				mask_id = np.unique(np.random.choice(mask_id, min_mask_len, replace=False))
			else:
				mask_id = mask_id[startid : startid + min_mask_len]
            
		mask[i, mask_id] = True


	if typeM == 0:
		ymask = torch.full((bsz, tsz - rfs), False) # B x Tc
		ids = torch.arange(mask.size(1)) # time dimension of mask Tz
		off = torch.arange(rfs)

		for i, m in enumerate(mask):
			mids = ids[m]
			if rfs > 0:
				mids = torch.repeat_interleave(mids, rfs).reshape(mids.size(0), rfs) - off   
				mids = mids.view(-1).unique()
				mids = mids[mids >= 0]
				mids = mids[mids < tsz - rfs]
			ymask[i,mids] = True

	if typeM == 1:
		ymask = torch.full((bsz, tsz - rfs), False) # B x Tc
		ids = np.arange(mask.size(1)) # time dimension of mask Tz
		for i, m in enumerate(mask):
			mids = ids[m]
			mids = np.split(mids, np.where(np.diff(mids) != 1)[0] + 1)
			idc = []
			for m in mids:
				if len(m) >= rfs//2:
					idc.append(m[- rfs//2])
					idc.append(m[0] - rfs//2)

			idc = np.array(idc)
			idc = idc[idc >= 0]
			idc = idc[idc < tsz - rfs]
			ymask[i, idc] = True

	if typeM == 2:
		ymask = mask[:, rfs :]

	return mask, ymask


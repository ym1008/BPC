import os
import shutil
import torch
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from dataload import LibriSpeech
from models import encoder, aggregator, predictor, BPC, wav2vec
from utils import get_args, get_augmentations, writeout_args


if __name__=='__main__': 
	
	args = get_args()
	pl.seed_everything(args.seed, workers=True)
	
	# set up augmentations
	augment_past, augment_future = get_augmentations(args)
	
	# set up data
	trainset = LibriSpeech(args.datatrain, args.no_input_norm, args.sample_size, train=True, augment=(augment_past, augment_future))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle=True, drop_last=True, 
											num_workers=args.num_workers, prefetch_factor = args.prefetch, pin_memory=args.pin_memory)

	# latent embedding sizes
	feature_embed_dim = eval(args.conv_encoder_layers)[-1][0] 
	if args.cnn:
		context_embed_dim = eval(args.conv_aggregator_layers)[-1][0] 
	elif args.transformer:
		context_embed_dim = args.transformer_dim
	elif args.lstm:
		context_embed_dim = args.lstm_dim


	enc = encoder(
		enc_layers = args.conv_encoder_layers, 
		normalisation = args.norm_enc, 
		activation = args.activation, 
		initialise = args.initialise,
		skiplayers = args.skip_connections_enc, 
		scale = args.residual_scale,
		dropout = args.dropout,
		log_compression = args.log_compression,
		bias = not args.no_bias_enc, 
		Cfinal = context_embed_dim 
	)

	is_causal = True if args.transformer and not args.mask else False
	agg = aggregator(
		feature_embed_dim, 
		transformer = (args.transformer, args.transformer_dim, args.transformer_ffn, args.transformer_nhead, 
						args.transformer_nlayers, args.pos_conv_layers, args.pos_conv_filter, args.pos_conv_norm),
		lstm = (args.lstm, args.lstm_dim, args.lstm_nlayers), 
		cnn = (args.cnn, args.conv_aggregator_layers, args.skip_connections_agg, args.residual_scale, args.padding_agg), 
		normalisation = args.norm_agg,
		activation = args.activation, 
		initialise = args.initialise, 
		is_causal = is_causal,
		dropout = (args.dropout, args.dropout_transformer)
	)


	if args.no_predictor:
		pred = lambda x: x
	else:
		pred = predictor(
			Cin = context_embed_dim, 
			Cout = args.prediction_dim, 
			ksteps = args.prediction_steps, 
			nlayers = args.prediction_nlayer, 
			normalisation = args.norm_pred,
			norm_final = args.norm_online,
			activation = args.activation
		)

	if args.contrastive: 
		model = wav2vec(
	        encoder = enc, 
	        aggregator = agg, 
	        predictor = pred, 
	        prediction_embed_dim = args.prediction_dim,
	        norm_target = args.norm_target,
	        l2_target = not args.no_l2_target, 
	        l2_online = not args.no_l2_online, 
	        rfs_agg = args.receptive_field,
	        offset_pred = args.offset, 
	        mask = args.mask, 
			mask_prob = args.mask_prob,
			mask_span = args.mask_span,
			mask_sampling = args.mask_sampling,
			mask_type = args.mask_type,
			n_negatives = args.n_negatives, 
	        learning_rate = args.lr,
	        weight_decay = args.wd, 
	        adam_betas = (args.beta1, args.beta2),
	        adam_eps = args.eps,
	        warmup_steps = args.warmup_steps,
	        lr_cosine = args.lr_cosine,
	        lr_cosine_max_steps = args.lr_cosine_max_steps
		)
	else:
		model = BPC(
			encoder = enc, 
			aggregator = agg, 
			predictor = pred, 
			feature_embed_dim = context_embed_dim,
			prediction_embed_dim = args.prediction_dim,
			ema_aggregator_only = args.ema_aggregator_only,
			ema_layers_only = args.ema_layers_only,
			layerdrop = args.layerdrop,
			min_layer = args.min_layer,
			max_layer = args.max_layer, 
			avg_layers = args.avg_layers,
			output_layer = args.output_layer,
			use_z = args.use_z,
			norm_target = args.norm_target,
			l2_target = not args.no_l2_target, 
			l2_online = not args.no_l2_online, 
			rfs_agg = args.receptive_field,
			offset_pred = args.offset, 
			mask = args.mask, 
			mask_prob = args.mask_prob,
			mask_span = args.mask_span,
			mask_sampling = args.mask_sampling,
			mask_type = args.mask_type,
			celoss = args.celoss,
			learning_rate = args.lr,
			weight_decay = args.wd, 
			adam_betas = (args.beta1, args.beta2),
			adam_eps = args.eps,
			warmup_steps = args.warmup_steps,
			lamb = args.lamb,
			tau = args.tau, 
			tau_max = args.tau_max,
			tau_anneal_steps = args.tau_anneal_steps,
			anneal_fn = args.anneal_fn,
			eman = args.eman
		)
	

	trainer = pl.Trainer(
		accumulate_grad_batches = args.update_grad_freq, 
		precision = args.precision, 
		strategy = 'ddp_find_unused_parameters_true', 
		max_steps = args.max_steps, 
		max_epochs = args.max_epochs,
		enable_model_summary = True, 
		logger = TensorBoardLogger(save_dir = args.log_dir, name = args.log_name, version = args.log_version, sub_dir = args.log_subdir), 
		log_every_n_steps = args.log_steps,
		callbacks=[
			ModelSummary(max_depth = -1), 
			ModelCheckpoint(save_top_k = -1, every_n_train_steps = args.save_chk_steps),
			LearningRateMonitor(logging_interval='step')
			], 
		sync_batchnorm = not args.no_syncBN
	)

	if args.tune_lr:
		tuner = Tuner(trainer)
		lr_finder = tuner.lr_find(model, train_dataloaders = trainloader, max_lr=10.0, num_training=100)
		print(lr_finder.results)
		fig = lr_finder.plot(suggest=True)
		fig.savefig(os.path.join(trainer.log_dir, 'lr_tune.png'))
	else:	
		writeout_args(args, args.log_dir)
		if args.load_chk_path is not None:
			trainer.fit(model, train_dataloaders = trainloader, ckpt_path = os.path.join(args.load_chk_path, 'checkpoints', args.load_chk_name))
		else:
			trainer.fit(model = model, train_dataloaders = trainloader)
		
		#os.rename(os.path.join(args.log_dir, args.log_version + 'config.txt'), args.path.join(trainer.log_dir, 'config.txt'))
		#shutil.move(os.path.join(args.log_dir, args.log_version+'config.txt'), os.path.join(trainer.log_dir, 'config.txt'))
		#writeout_args(args, trainer.log_dir)


	# Enable Stochastic Weight Averaging using the callback
	# trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
	
	


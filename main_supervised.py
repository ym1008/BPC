import os
import torch
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl

from torch import optim
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from functools import partial
from dataload import LibriSpeech
from models import encoder, aggregator, BPC, frame_classifier, sequence_classifier
from utils import get_args, get_augmentations, writeout_args, get_state_dicts, get_mask_ids

from transformers import Wav2Vec2Model, Data2VecAudioModel


class BPC_finetuned(pl.LightningModule):
	def __init__(
		self, 
		pretrained_model,
		classifier,
		min_layer = None,
		max_layer = None,
		learning_rate = 0.0005,
		weight_decay = 0.01, 
		warmup_steps = 12000,
		freeze_first_nsteps = 10000,
		fs_model = False
	):

		super().__init__()
		self.save_hyperparameters(ignore=['pretrained_model', 'classifier'])
		
		self.pt_model = pretrained_model
		self.classifier = classifier

		self.min_layer = min_layer
		self.max_layer = max_layer
		
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.warmup_steps = warmup_steps
		self.freeze_updates = freeze_first_nsteps
		
		self.fs_model = fs_model

	def training_step(self, batch, batch_idx):
		
		x, y = batch
		x = x[0] # raw waveform 
		ny = [0] # no. of frames - needed for ctc loss
		y = y[1] # phone ids

		if self.freeze_updates >= self.trainer.global_step:
			self.pt_model.eval()
			with torch.no_grad():
				if self.fs_model:
					x = torch.squeeze(x)
					x = self.pt_model(x, output_hidden_states=True)
					x = x.hidden_states[self.max_layer + 1]
				else:
					x = self.pt_model(x, min_layer = self.min_layer, max_layer = self.max_layer)
					x = x.transpose(1,2) # BCT -> BTC
	
		else:
			self.pt_model.train()
			if self.fs_model:
				x = torch.squeeze(x)
				x = self.pt_model(x, output_hidden_states=True)
				x = x.hidden_states[self.max_layer + 1]
			else:
				x = self.pt_model(x, min_layer = self.min_layer, max_layer = self.max_layer)
				x = x.transpose(1,2) # BCT -> BTC
	
		loss, acc = self.classifier(x, (ny, y))
		loss = loss.mean()
		acc = acc.mean()

		self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
		self.log("train_accuracy", acc, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        
		return loss


	def configure_optimizers(self):

		def fn(warmup_steps, step):
			if step < warmup_steps:
				return float(step) / float(max(1, warmup_steps))
			else:
				return 1.0

		def linear_warmup(warmup_steps):
			return partial(fn, warmup_steps)


		def scheduler_fn(optimizer):

			scheduler = {
				"scheduler": optim.lr_scheduler.LambdaLR(
					optimizer,
					linear_warmup(self.warmup_steps),
				),
				"interval": "step",
				"frequency": 1,
			}
			return scheduler


		opt = optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		sch = scheduler_fn(opt)
		return ([opt], [sch])



	def evaluate(self, batch, loss_field, acc_field):
		x, y = batch
		x = x[0] # raw waveform 
		ny = [0] # no. of frames - needed for ctc loss
		y = y[1] # phone ids

		if self.fs_model:
			x = torch.squeeze(x)
			x = self.pt_model(x, output_hidden_states=True)
			x = x.hidden_states[self.max_layer + 1]
		else:
			x = self.pt_model(x, min_layer = self.min_layer, max_layer = self.max_layer)
			x = x.transpose(1,2) # BCT -> BTC
	
		loss, acc = self.classifier(x, (ny, y))
		loss = loss.mean()
		acc = acc.mean()

		self.log(loss_field, loss, prog_bar = True, sync_dist=True)
		self.log(acc_field, acc, prog_bar = True, sync_dist=True)

        
	def validation_step(self, batch, batch_idx):
		self.evaluate(batch, 'val_loss', 'val_accuracy')


	def test_step(self, batch, batch_idx):
		self.evaluate(batch, 'test_loss', 'test_accuracy')




if __name__=='__main__': 
	
	args = get_args()
	pl.seed_everything(args.seed, workers=True)
	
	# set up augmentations
	augment_past, augment_future = get_augmentations(args)
	
	# set up data
	trainset = LibriSpeech(
		args.datatrain, args.no_input_norm, args.sample_size, 
		train = True, augment = (augment_past, augment_future), label_fname = args.labelstrain, 
		rfs = args.rfs_enc, stride = args.stride_enc, ctcloss = args.ctc, nsn = args.nsn
	)

	validset = LibriSpeech(
		args.datatest, args.no_input_norm, args.sample_size, 
		train = False, augment = (augment_past, augment_future), label_fname = args.labelstest, 
		rfs = args.rfs_enc, stride = args.stride_enc, ctcloss = args.ctc, nsn = args.nsn
	)

	trainloader = torch.utils.data.DataLoader(
		trainset, batch_size = args.batch_size, shuffle = True, drop_last = True, 
		num_workers=args.num_workers, prefetch_factor = args.prefetch, pin_memory=args.pin_memory
	)

	validloader = torch.utils.data.DataLoader(
		validset, batch_size = args.batch_size, shuffle = False, drop_last = True, 
		num_workers=args.num_workers, prefetch_factor = args.prefetch, pin_memory=args.pin_memory
	)


	# latent embedding sizes
	feature_embed_dim = eval(args.conv_encoder_layers)[-1][0] 
	embed_dim = eval(args.conv_aggregator_layers)[-1][0] 
	if args.transformer:
		embed_dim = args.transformer_dim
	elif args.lstm:
		embed_dim = args.lstm_dim



	# encoder
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
		Cfinal = embed_dim
	)

	# aggregator, if using
	if args.use_z:	
		agg = None 
		embed_dim = feature_embed_dim
	else:
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
	

	# load up pre-trained weights
	if args.load_chk_path is not None:
		chk = torch.load(os.path.join(args.load_chk_path, 'checkpoints', args.load_chk_name))
		chk = chk['state_dict']
		chk, agg_chk = get_state_dicts(chk, args.use_target)

		enc.load_state_dict(chk) 
		agg.load_state_dict(agg_chk) if not args.use_z else None


	# Turn off gradients if not fine-tuning
	if not args.finetune_all:
		for param in enc.parameters():
			param.requires_grad = False

		if not args.finetune_agg and not args.use_z:
			for param in agg.parameters():
				param.requires_grad = False


	# ------------------#
	# pre-trained model #
	# ------------------#
	if args.d2v_fs:
		pretrained_model = Data2VecAudioModel.from_pretrained("facebook/" + args.fs_model)
	elif args.w2v2_fs:
		pretrained_model = Wav2Vec2Model.from_pretrained("facebook/" + args.fs_model)
	else:
		pretrained_model = BPC(
				encoder = enc, 
				aggregator = agg, 
				predictor = None, 
				feature_embed_dim = feature_embed_dim,
				mask = args.mask, 
				mask_prob = args.mask_prob,
				mask_span = args.mask_span
		)


	if args.d2v_fs or args.w2v2_fs:
		embed_dim = args.classifier_dim
	
		# Turn off gradients if not fine-tuning
		if not args.finetune_all:
			for param in pretrained_model.parameters():
				param.requires_grad = False

	
	# -----------#
	# classifier #
	# -----------#
	if args.ctc:
		classifier = sequence_classifier(embed_dim, args.nclasses, (args.use_lstm, args.bilstm_dim))
		early_stopping_mode = 'min'
	else:
		classifier = frame_classifier(embed_dim, args.nclasses)
		early_stopping_mode = 'max'


	# -----------------#
	# fine-tuned model #
	# -----------------#
	model = BPC_finetuned(
		pretrained_model = pretrained_model,
		classifier = classifier,
		min_layer = args.min_layer,
		max_layer = args.max_layer,
		learning_rate = args.lr,
		weight_decay = args.wd, 
		warmup_steps = args.warmup_steps,
		freeze_first_nsteps = args.freeze_first_nsteps,
		fs_model = args.d2v_fs or args.w2v2_fs
	)



	if args.load_chk_path is not None:
		log_dir = args.load_chk_path.split('Experiments/')
		log_dir = os.path.join(log_dir[0], 'Experiments/downstream_classification', log_dir[1])
		setattr(args, 'log_dir', log_dir)
	else:
		setattr(args, 'log_dir', os.path.join(args.log_dir, 'downstream_classification'))
		if args.d2v_fs:
			setattr(args, 'log_name', 'd2v-fs')
		elif args.w2v2_fs:
			setattr(args, 'log_name', 'w2v2-fs')
		else:
			setattr(args, 'log_name', 'no_pretraining')




	trainer = pl.Trainer(
		accumulate_grad_batches = args.update_grad_freq, 
		precision = args.precision, 
		strategy = 'ddp_find_unused_parameters_true', 
		max_steps = args.max_steps, 
		enable_model_summary = True, 
		logger = TensorBoardLogger(save_dir = args.log_dir, name = args.log_name, version = args.log_version), 
		log_every_n_steps = args.log_steps,
		callbacks = [
			ModelCheckpoint(monitor='val_accuracy', mode = early_stopping_mode),
			EarlyStopping(monitor='val_accuracy', mode = early_stopping_mode, patience = 20)
		], 
		val_check_interval = args.check_val_steps,
		check_val_every_n_epoch = None,
		sync_batchnorm = not args.no_syncBN
	)


	if args.tune_lr:
		tuner = Tuner(trainer)
		tuner.lr_find(model, train_dataloaders = trainloader)
		
	if args.load_classifier:
		trainer.test(model = model, dataloaders = validloader, ckpt_path = args.load_classifier, verbose=True)
	else:
		trainer.fit(model = model, train_dataloaders = trainloader, val_dataloaders = validloader)
		setattr(args, 'log_dir', trainer.log_dir)
		setattr(args, 'lr', model.learning_rate)
		writeout_args(args, trainer.log_dir)

	#trainer.test(model, dataloaders = testloader)

	# Enable Stochastic Weight Averaging using the callback
	# trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
	
	


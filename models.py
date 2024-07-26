import math
import torch
import numpy as np
import torch.nn as nn
import lightning.pytorch as pl
from functools import partial
from os.path import join
from utils import compute_per, get_mask_ids
from torch.optim.swa_utils import AveragedModel
from torch import optim



class EMA(AveragedModel):
    def train(self, mode:bool):
        return super().train(False)



class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(-2,-1)
        return x



class Normalise(nn.Module):
    def __init__(self, Cdim, norm):
        super().__init__()

        # Assume input is B C T
        self.normlayer = False

        if norm == 'bn':
            self.normlayer = nn.BatchNorm1d(Cdim)
        if norm == 'ln':
            self.normlayer = nn.GroupNorm(1, Cdim)
        if norm == 'ln_c': 
            self.normlayer = nn.Sequential(TransposeLast(), nn.LayerNorm(Cdim), TransposeLast()) # input needs to be B T C
        if norm == 'in':
            self.normlayer = nn.InstanceNorm1d(Cdim)
        if norm == 'in_affine':
            self.normlayer = nn.InstanceNorm1d(Cdim, affine=True)
        if norm == 'in_eman':
            self.normlayer = nn.InstanceNorm1d(Cdim, track_running_stats=True) 
        if norm == 'in_full':
            self.normlayer = nn.InstanceNorm1d(Cdim, affine=True, track_running_stats=True) 


    def forward(self, x):
        if self.normlayer:
            x = self.normlayer(x)
        return x




class encoder(nn.Module):
    def __init__(self, enc_layers = '[(512, 10, 5)] + [(512, 8, 4)] + [(512, 4, 2)] * 3', 
                       normalisation = 'in', 
                       activation = 'relu', 
                       initialise = False, 
                       skiplayers = False, 
                       scale = 0.5, 
                       dropout = 0.0,
                       log_compression = False,
                       bias = True, 
                       Cfinal = 512
        ):

        super().__init__()

        enc_layers = eval(enc_layers)
        layers = nn.ModuleList()
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        Cin = 1
        for (Cout, filt, stride) in enc_layers:
            conv = nn.Conv1d(Cin, Cout, filt, stride = stride, bias = bias)
            nn.init.kaiming_normal_(conv.weight) if initialise else None
            
            layers.append(
                nn.Sequential(
                conv, 
                nn.Dropout(dropout), 
                Normalise(Cout, normalisation), 
                act),         
            )
            Cin = Cout

        if Cfinal != Cout:
            self.linear = nn.Sequential

        self.linear_proj = nn.Sequential(Normalise(Cin, normalisation), nn.Conv1d(Cin, Cfinal, 1)) if Cin != Cfinal else None
        self.layers = layers
        self.skiplayers = skiplayers
        self.scale = math.sqrt(scale)
        self.log_compression = log_compression


    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)

        for layer in self.layers:
            residual = x
            x = layer(x)

            if self.skiplayers and x.size(1) == residual.size(1):
                ratio = residual.size(2) // x.size(2)
                res_downsampled = residual[..., ::ratio][..., :x.size(2)]
                x = (x + res_downsampled) * self.scale

        if self.log_compression:
            x = (x.abs() + 1).log()

        if self.linear_proj:
            x = self.linear_proj(x)

        return x



class cnn_agg(nn.Module):
    def __init__(self, Cin, agg_layers, normalisation, activation, initialise, skiplayers, scale, dropout, padding):
        super().__init__()

        agg_layers = eval(agg_layers)
        layers = nn.ModuleList()
        rproj_layers = nn.ModuleList()
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        for (Cout, filt, stride) in agg_layers:
            pad = filt // 2 if padding else 0

            conv = nn.Conv1d(Cin, Cout, filt, stride=stride, padding=pad)
            nn.init.kaiming_normal_(conv.weight) if initialise else None
            
            layers.append(
                nn.Sequential(
                conv,
                nn.Dropout(dropout), 
                Normalise(Cout, normalisation),
                act),          
            )
            
            projfn = nn.Conv1d(Cin, Cout, 1, bias=False) if Cin != Cout else None
            rproj_layers.append(projfn)
            
            Cin = Cout     

        self.layers = layers
        self.rproj_layers = rproj_layers
        self.skiplayers = skiplayers
        self.scale = math.sqrt(scale)
        self.padding = padding


    def forward(self, x, avg_layers=False, min_layer = None, max_layer = None, output_layer='act'):
        layer_out = [] 


        for i, (layer, reslayer) in enumerate(zip(self.layers, self.rproj_layers)):
            residual = x
            #x = layer(x)

            for j, module in enumerate(layer):
                x = module(x)
                if output_layer == 'conv' and j == 0:
                    output = x
                elif output_layer == 'norm' and j == 2:
                    output = x
                elif output_layer == 'act' and j == 3:
                    output = x

            
            if avg_layers:
                if i >= min_layer and i <= max_layer:
                    layer_out.append(output)  # used to be append(x)

            if max_layer is not None:
                if i == max_layer:
                    x = output # added for output layer testing
                    break

            if self.skiplayers:
                if reslayer:
                    residual = reslayer(residual)

                if x.size(2) != residual.size(2):
                    ratio = residual.size(2) // x.size(2)
                    residual = residual[..., ::ratio][..., :x.size(2)]
                 
                x = (x + residual) * self.scale


        # to deal with different sequence lengths at different layers of cnn (without padding): average frames correspoding to receptive field size of highest layer. 
        if avg_layers:
            if not self.padding:
                min_len = layer_out[-1].size(2)
                for i, lo in enumerate(layer_out[:-1]):
                    offset = lo.size(2) - min_len + 1
                    for j in range(min_len):
                        lo[:,:,j] = lo[:,:, j : j+offset].mean(dim=2)
                    layer_out[i] = lo[:,:, : min_len]

            return layer_out
        else:
            return x



class positional_embedder(nn.Module):
    def __init__(self, Cout, pos_conv_k, pos_conv_depth, normalisation, activation = 'relu', dropout=0.1):
        super().__init__()

        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        k = pos_conv_k // pos_conv_depth
        self.positional_enc = nn.Sequential(*[nn.Sequential(nn.Conv1d(Cout, Cout, k, padding = k //2, groups = 16), 
                                                           Normalise(Cout, normalisation), act) for _ in range(pos_conv_depth)])

        self.norm = Normalise(Cout, normalisation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.positional_enc(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x



class transformer_layer(nn.Module):
    def __init__(self, normalisation, embed_dim, ffn_dim, nhead, is_causal=False, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = Normalise(embed_dim, normalisation)
        self.ffn1 = nn.Conv1d(embed_dim, ffn_dim, 1) # nn.linear(embed_dim, ffn_dim)
        self.act = nn.GELU()
        self.ffn2 = nn.Conv1d(ffn_dim, embed_dim, 1) # nn.linear(ffn_dim, embed_dim)
        self.ff_dropout = nn.Dropout(dropout)
        self.norm2 = Normalise(embed_dim, normalisation)
        self.is_causal = is_causal


    def forward(self,x):
        
        # B C T >> B T C
        x = x.transpose(1,2)
        x = x + self.attn_dropout(self.attn(x,x,x, need_weights=False, is_causal = self.is_causal)[0]) 
        x =  self.norm1(x.transpose(1,2)) # B T C >> B C T

        residual = x 
        x = self.ffn2(self.act(self.ffn1(x)))
        layer_out = x
        x = self.ff_dropout(x)
        x = self.norm2(x + residual)

        return x, layer_out



class transformer_agg(nn.Module):
    def __init__(self, embed_dim, ffn_dim, nhead, nlayers, normalisation, is_causal = False, dropout=0.1):
        super().__init__()

        self.transformer = nn.ModuleList([transformer_layer(normalisation, embed_dim, ffn_dim, nhead, is_causal, dropout) for _ in range(nlayers) ])


    def forward(self, x, avg_layers = False, min_layer = None, max_layer = None, output_layer='act', layerdrop = 0):

        layer_out = []
        for i, layer in enumerate(self.transformer):

            do_prob = np.random.random() if layerdrop > 0 else 1

            if do_prob > layerdrop:
                x, lo = layer(x)

                if avg_layers:
                    if i >= min_layer and i <= max_layer:
                        layer_out.append(lo) 
        
            if max_layer is not None:
                if i == max_layer: 
                    break

        if avg_layers:
            return layer_out
        else:
            return x



class lstm_agg(nn.Module):
    def __init__(self, Cin, Hdim, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(Cin, Hdim, num_layers, batch_first=True)

        # FIGURE OUT: get intermediate layer outputs, in line with cnn and transformer method of averaging across layers. 
        #             perhaps meaningless for lstm. 

    def forward(self, x):
        x = x.transpose(1,2) # BCT -> BTC

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        
        x = x.transpose(1,2) # BTH -> BHT

        return x




class aggregator(nn.Module):
    def __init__(self, feature_embed_dim = 512, transformer = (False, 512, 512, 12, 12, 5, 128, 'other'), lstm = (False,128,3), 
                        cnn = (True, '[(512, 3, 1)] * 9', False, 0.5, False), normalisation = 'bn', activation = 'relu', 
                        initialise = False, is_causal = False, dropout = (0, 0.1)):
        super().__init__()
        self.transformer = transformer[0]
            
        if lstm[0]: 
            embed_dim = lstm[1]
            nlayers = lstm[2]
            self.model = lstm_agg(feature_embed_dim, lstm[1], lstm[2])
            
        elif transformer[0]:
            embed_dim = transformer[1]
            ffn_dim = transformer[2]
            nhead = transformer[3]
            nlayers = transformer[4]
            pos_conv_depth = transformer[5]
            pos_conv_filter = transformer[6]
            pos_conv_norm = transformer[7]

            self.proj_enc = positional_embedder(embed_dim, pos_conv_filter, pos_conv_depth, pos_conv_norm, activation, dropout[1])
            self.model = transformer_agg(embed_dim, ffn_dim, nhead, nlayers, normalisation, is_causal, dropout[1])

        elif cnn[0]:
            layers = cnn[1]
            skiplayers = cnn[2]
            scale = cnn[3]
            padding = cnn[4]
            self.model = cnn_agg(feature_embed_dim, layers, normalisation, activation, initialise, skiplayers, scale, dropout[0], padding)

            
    def forward(self, x, avg_layers = False, min_layer = None, max_layer = None, output_layer='act', layerdrop=0):
        if self.transformer:
            x = self.proj_enc(x) 
            x = self.model(x, avg_layers, min_layer, max_layer, layerdrop)
        else:
            x = self.model(x, avg_layers, min_layer, max_layer, output_layer)
        
        return x




class projector(nn.Module):
    def __init__(self, Cin, Cout, normalisation, activation = 'ReLU'):
        super().__init__()

        Cmid = Cout*2 if Cout < Cin//2 else Cout
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        # BYOL: linear, BN, ReLU, linear (256)
        self.nn = nn.Sequential(
                        nn.Conv1d(Cin, Cmid, 1),
                        Normalise(Cmid, normalisation),
                        act,
                        nn.Conv1d(Cmid, Cout, 1)
                    )

    def forward(self, x):
        x = self.nn(x)
        return x




class predictor(nn.Module):
    def __init__(self, Cin, Cout, ksteps, nlayers, normalisation, norm_final, activation = 'relu'):
        super().__init__()

        Cmid = Cout*2 if Cout < Cin//2 else Cout
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        # BYOL: linear, BN, ReLU, linear (256)
        if nlayers > 1:
            self.nn_base = nn.ModuleList()
            for k in range(nlayers - 1):
                self.nn_base.append(
                    nn.Sequential(
                        nn.Conv1d(Cin, Cmid, 1),
                        Normalise(Cmid, normalisation),
                        act
                    )
                )
                Cin = Cmid
        
        self.nn_final = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(Cmid, Cout, 1), 
                Normalise(Cout, norm_final)
            ) 
            for _ in range(ksteps) ]) 
    
        self.nlayers = nlayers

    def forward(self, x):

        if self.nlayers != 1:
            for layer in self.nn_base:
                x = layer(x)

        x_steps = []
        for layer in self.nn_final: 
            x_steps.append(layer(x))
        x = x_steps

        return x





class BPC(pl.LightningModule):
    def __init__(
        self, 
        encoder, 
        aggregator, 
        predictor, 
        feature_embed_dim = 512,
        prediction_embed_dim = 512,
        ema_aggregator_only = False,
        ema_layers_only = False,
        layerdrop = 0,
        min_layer = None,
        max_layer = None, 
        avg_layers = False,
        output_layer = 'act',
        use_z = False,
        norm_target = 'in',
        l2_target = True, 
        l2_online = True, 
        rfs_agg = 18,
        offset_pred = 3, 
        mask = False, 
        mask_prob = 0.65,
        mask_span = 10,
        mask_sampling = False,
        mask_type = 0,
        celoss = False,
        learning_rate = 0.0005,
        weight_decay = 0.01, 
        adam_betas = (0.9, 0.999),
        adam_eps = 1e-8,
        warmup_steps = 12000,
        lamb = 1,
        tau = 0.999, 
        tau_max = 0.9999,
        tau_anneal_steps = 30000,
        anneal_fn = 'linear',
        eman = False
    ):

        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'aggregator', 'predictor'])
        
        self.enc = encoder
        self.agg = aggregator
        self.pred = predictor
        self.norm_target = Normalise(prediction_embed_dim, norm_target)
        self.l2_target = l2_target
        self.l2_online = l2_online
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.avg_layers = avg_layers
        self.output_layer = output_layer
        self.use_z = use_z
        self.ema_aggregator_only = ema_aggregator_only
        self.ema_layers_only = ema_layers_only
        self.layerdrop = layerdrop
       
        self.rfs_agg = rfs_agg
        self.offset_pred = rfs_agg + offset_pred
        self.celoss = celoss
    
        self.mask = mask      
        self.mask_prob = mask_prob
        self.mask_span = mask_span   
        self.mask_sampling = mask_sampling  
        self.mask_type = mask_type

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = adam_betas
        self.eps = adam_eps
        self.warmup_steps = warmup_steps
        self.lamb = lamb

        
        # initialise EMA model (target network)
        def tau_fn(num_averaged, anneal_fn):
            if num_averaged < tau_anneal_steps:
                if anneal_fn == 'linear':
                    return tau + (tau_max - tau) * num_averaged / tau_anneal_steps
            
                elif anneal_fn == 'cosine':
                    return 1 - (1 - tau) * 0.5 * (torch.cos(np.pi * num_averaged / tau_anneal_steps) + 1 )
                
                else:
                    return tau
            else:
                return tau_max


        ema_fn = lambda averaged_model_parameter, model_parameter, num_averaged:\
                tau_fn(num_averaged, anneal_fn) * averaged_model_parameter + (1 - tau_fn(num_averaged, anneal_fn)) * model_parameter 
    
        self.ema_enc = EMA(encoder, avg_fn = ema_fn, use_buffers = eman).eval() if not ema_aggregator_only else None
        
        if self.ema_layers_only:
            self.ema_agg = EMA(aggregator.model, avg_fn = ema_fn, use_buffers = eman).eval() 
        else:
            self.ema_agg = EMA(aggregator, avg_fn = ema_fn, use_buffers = eman).eval() 
        

        if not ema_aggregator_only:
            for p in self.ema_enc.parameters():
                p.requires_grad = False

        for p in self.ema_agg.parameters():
            p.requires_grad = False


        if mask:
            self.maskW = nn.Parameter(torch.randn(feature_embed_dim))
            self.maskW.requires_grad = True

   

    def apply_mask(self, x, mask):
        # B C T 
        x = x.transpose(1,2)
        xm = x.clone()
        xm[mask] = self.maskW.to(dtype = xm.dtype)
        x = xm.transpose(1,2)
        return x




    def forward(self, x, min_layer = None, max_layer = None):
        x = self.enc(x)

        if self.mask:
            mask, ymask = get_mask_ids((x.size(0), x.size(2)), 
                            self.mask_prob, self.mask_span, self.rfs_agg, 
                            self.mask_type, self.mask_sampling) 
            x = self.apply_mask(x, mask)

        if self.agg is not None:
            x = self.agg(x, min_layer = min_layer, max_layer = max_layer)

        return x



    def compute_loss(self, xsteps, ylabel, mask=None):
        # x:    [B x C x T]     list of K items (k steps)
        # ylabel:   B x C x T

        if self.l2_online:
            xsteps = [x / torch.norm(x, dim = 1, keepdim = True) for x in xsteps] 

        if self.l2_target:
            ylabel = ylabel / torch.norm(ylabel, dim = 1, keepdim = True)

        
        tm = mask.size(1) if mask is not None else None
        diff = ylabel.size(2) - xsteps[0].size(2)
        off = self.offset_pred - diff

        loss = 0.0
        n = 0

        for k, x in enumerate(xsteps):
            t = x.size(2) - (k + off)

            x = x[:,:, : t]
            y = ylabel[:,:, diff + off + k : diff + off + k + t]

            if mask is not None:
                mask_tmp = mask[:, : tm - (k + off)]
                x = x.transpose(1,2)[mask_tmp]
                y = y.transpose(1,2)[mask_tmp]
               

            if self.celoss: 
                loss += (- (y * nn.functional.log_softmax(x, dim = 1)).sum(dim=1)).sum(dim=1)
                n += torch.numel(x[0,0,:])
            else:
                loss += nn.functional.mse_loss(x, y, reduction='none').sum() # have to do this because T dimension changes with every step k if doing multi-step prediction
                n += torch.numel(x)

        loss = loss / n
        
        return loss



    def training_step(self, batch, batch_idx):
        x, _ = batch
        y = x[1] # future (may have beeen augmented)
        x = x[0] # past (may have been augmented)

        with torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            # online
            x = self.enc(x)
            
            if self.mask:
                mask, ymask = get_mask_ids((x.size(0), x.size(2)), 
                                self.mask_prob, self.mask_span, self.rfs_agg, 
                                self.mask_type, self.mask_sampling) 
                mask = mask.to(self.device)
                ymask = ymask.to(self.device)
                
                x = self.apply_mask(x, mask)
            else:
                ymask = None


            x = self.agg(x, layerdrop = self.layerdrop)
            x = self.pred(x)
           
            # target
            self.ema_enc.update_parameters(self.enc) if not self.ema_aggregator_only else None
            self.ema_agg.update_parameters(self.agg) if not self.ema_layers_only else self.ema_agg.update_parameters(self.agg.model)


            with torch.no_grad():
                if self.ema_aggregator_only:
                    self.enc.eval()
                    y = self.enc(y)
                    self.enc.train()
                else:
                    y = self.ema_enc(y)
                
                if not self.use_z:
                    if self.ema_layers_only:
                        self.agg.eval()
                        y = self.agg.proj_enc(y)
                        self.agg.train()

                    y = self.ema_agg(y, self.avg_layers, self.min_layer, self.max_layer, self.output_layer)


                if self.avg_layers:
                    y = [self.norm_target(ylayer) for ylayer in y]
                    y = sum(y) / len(y) 
                else:
                    y = self.norm_target(y)
             
            # loss objective 
            loss = self.compute_loss(x, y.detach(), ymask)
            self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
            
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


        if self.lamb == 1:
            opt = optim.AdamW(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay, betas = self.betas, eps = self.eps)
            sch = scheduler_fn(opt)
            return ([opt], [sch])
        else:
            parameters = [param for name, param in self.named_parameters() if 'pred' not in name]
            opt = optim.AdamW(parameters, lr = self.learning_rate, weight_decay = self.weight_decay, betas = self.betas, eps = self.eps)
            opt_pred = optim.AdamW(self.pred.parameters(), lr = self.learning_rate * self.lamb, weight_decay = self.weight_decay, betas = self.betas, eps = self.eps)
        
            sch = scheduler_fn(opt)
            sch_pred = scheduler_fn(opt_pred)

            return ([opt, opt_pred], [sch, sch_pred])



class wav2vec(pl.LightningModule):
    def __init__(
        self, 
        encoder, 
        aggregator, 
        predictor, 
        prediction_embed_dim = 512,
        norm_target = 'other',
        l2_target = False, 
        l2_online = False, 
        rfs_agg = 18,
        offset_pred = 3, 
        mask = False, 
        mask_prob = 0.65,
        mask_span = 10,
        mask_sampling = False,
        mask_type = 0,
        n_negatives = 10, 
        learning_rate = 0.0005,
        weight_decay = 0.01, 
        adam_betas = (0.9, 0.999),
        adam_eps = 1e-8,
        warmup_steps = 12000,
        lr_cosine = False,
        lr_cosine_max_steps = 400000
):

        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'aggregator', 'predictor'])
        
        self.enc = encoder
        self.agg = aggregator
        self.pred = predictor
        self.norm_target = Normalise(prediction_embed_dim, norm_target)
        self.l2_target = l2_target
        self.l2_online = l2_online
       
        self.rfs_agg = rfs_agg
        self.offset_pred = rfs_agg + offset_pred
        self.n_negatives = n_negatives

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = adam_betas
        self.eps = adam_eps
        self.warmup_steps = warmup_steps
        self.lr_cosine = lr_cosine
        self.lr_cosine_max_steps = lr_cosine_max_steps

        self.mask = mask      
        self.mask_prob = mask_prob
        self.mask_span = mask_span   
        self.mask_sampling = mask_sampling  
        self.mask_type = mask_type

        if mask:
            self.maskW = nn.Parameter(torch.randn(prediction_embed_dim))
            self.maskW.requires_grad = True



    def forward(self, x, min_layer = None, max_layer = None):
        x = self.enc(x)
        if self.agg is not None:
            x = self.agg(x, min_layer = min_layer, max_layer = max_layer)
        return x


    def apply_mask(self, x, mask):
        # B C T 
        x = x.transpose(1,2)
        xm = x.clone()
        xm[mask] = self.maskW.to(dtype = xm.dtype)
        x = xm.transpose(1,2)
        return x


    def sample_negs(self, z):
        bdim, cdim, tdim = z.size()
        
        z = z.transpose(0,1).contiguous() # BCT -> CBT
        z = z.view(cdim, -1) # 3D to 2D
        
        #maxval = bdim * tdim if self.cross_sample_negs else tdim
        ids = torch.randint(0, tdim, (bdim, self.n_negatives * tdim))  
        #if not self.cross_sample_negs:
        for i in range(1, bdim):
            ids[i] += i*tdim
        ids = ids.view(-1)

        z = z[..., ids]
        z = z.view(cdim, bdim, self.n_negatives, tdim).permute(2, 1, 0, 3)  # change to LxBxCxT (L for lambda)
        return z



    def compute_loss(self, xsteps, ylabel, ynegs, mask=None):
        # x:           [B x C x T]     list of K items (k steps)
        # ylabel:       B x C x T       
        # ynegs:   L x B x C x T

        if self.l2_online:
            xsteps = [x / torch.norm(x, dim = 1, keepdim = True) for x in xsteps] 

        if self.l2_target:
            ylabel = ylabel / torch.norm(ylabel, dim = 1, keepdim = True)

        tm = mask.size(1) if mask is not None else None
        diff = ylabel.size(2) - xsteps[0].size(2)
        off = self.offset_pred - diff

        loss = 0.0
        n = 0
        for k, x in enumerate(xsteps):
            t = x.size(2) - (k + off)

            x = x[:, :, :t]
            y = ylabel[:, :, diff + off + k : diff + off + k + t]
            yn = ynegs[:, :, :, :t] 

            if mask is not None:
                mask_tmp = mask[:, : tm - (k + off)]
                x = x.transpose(1,2)[mask_tmp]
                y = y.transpose(1,2)[mask_tmp]
                yn = yn.transpose(2,3).flatten(start_dim=1, end_dim=2)
                yn = yn[:, : x.size(0), :]
            else:
                x = x.transpose(1,2).flatten(end_dim=1)
                y = y.transpose(1,2).flatten(end_dim=1)
                yn = yn.transpose(2,3).flatten(start_dim=1, end_dim=2)
         
            t = x.size(0)
            predictions = torch.zeros((x.size(0) * (1 + ynegs.size(0)))).to(self.device)
            labels = torch.zeros_like(predictions)
            weights = torch.full_like(predictions, 1/ynegs.size(0))
                
            predictions[: t] = (x[:, :] * y).sum(dim=1) 
            predictions[t :] = (x[:, :] * yn).sum(dim=2).flatten()
            labels [: t] = 1
            weights[: t] = 1
            

#            predictions = torch.zeros((x.size(0), t * (1 + ynegs.size(0)))).to(self.device)
#            labels = torch.zeros_like(predictions)
#            weights = torch.full_like(predictions, 1/ynegs.size(0))
                
#            predictions[:, : t] = (x[:, :, :] * y).sum(dim=1) # Becomes: B x T (T changing with k)
#            predictions[:, t :] = (x[:, :, :] * yn).sum(dim=2).transpose(0,1).flatten(1) # becomes: L x B x T then transposed to B x L x T then flattened to B x LT
#            labels [:, : t] = 1
#            weights[:, : t] = 1

            loss += nn.functional.binary_cross_entropy_with_logits(predictions, labels, weight = weights, reduction='none').sum()
            n += torch.numel(predictions)

        loss = loss / n
        
        return loss

               


    def training_step(self, batch, batch_idx):

        x, _ = batch
        x = x[0] 

        y = self.enc(x)

        if self.mask:
            mask, ymask = get_mask_ids((y.size(0), y.size(2)), 
                            self.mask_prob, self.mask_span, self.rfs_agg, 
                            self.mask_type, self.mask_sampling) 
            mask = mask.to(self.device)
            ymask = ymask.to(self.device)
            
            x = self.apply_mask(y, mask)
        else:
            ymask = None
            x = y

        x = self.agg(x)
        x = self.pred(x)
    
        ynegs = self.sample_negs(y)
        y = self.norm_target(y)

        # loss objective 
        loss = self.compute_loss(x, y.detach(), ynegs.detach(), mask = ymask) 
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        
        return loss


    def configure_optimizers(self):

        def fn(warmup_steps, step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                if self.lr_cosine:
                    ratio = float(step - warmup_steps) / float(self.lr_cosine_max_steps - warmup_steps)
                    lr = 0.5 * (np.cos(ratio * np.pi) + 1)
                    return 0.5 * (np.cos(ratio * np.pi) + 1)
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


        opt = optim.AdamW(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay, betas = self.betas, eps = self.eps)
        sch = scheduler_fn(opt)
        return ([opt], [sch])
    



class frame_classifier(nn.Module):
    def __init__(self, input_dim, nclasses):
        super().__init__()

        self.nclasses = nclasses
        self.linear = nn.Linear(input_dim, nclasses)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x, y, test=True):
        x    = self.linear(x)
        tdim = x.size(1)
        x = x.reshape(x.size(0)*x.size(1), self.nclasses) # BT x nclass
        _, pred = torch.max(x, 1)
        
        y = y[1][:, -tdim:]
        y = y.reshape(-1) 
        
        loss = self.criterion(x, y)
        acc = torch.sum(pred == y).float()/y.size(0)

        return (loss, acc)


    def save_params(self, exDir, n_updates):
        outDict = {'updateNo': n_updates, 'linear': self.linear.state_dict()}
        torch.save(outDict, join(exDir, 'classifier_params_'+str(n_updates)+'.tar'))
        print('Saving model at {} updates'.format(n_updates))


    def load_params(self, exDir, n_updates, device):
        checkpoint = torch.load(join(exDir, 'classifier_params_'+str(n_updates)+'.tar'), map_location=device)
        self.linear.load_state_dict(checkpoint['linear'])


    
class sequence_classifier(nn.Module):
    def __init__(self, input_dim, nclasses, lstm=(False, 256)):
        super().__init__()

        if lstm[0]:
            self.lstm = nn.LSTM(input_dim, lstm[1], 2, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(lstm[1]*2, nclasses + 1)
        
        else:
            self.linear = nn.Linear(input_dim, nclasses + 1)
            
        self.criterion = nn.CTCLoss()
        self.lstmF = lstm[0]

    def decode(self, x, y, ny):
        
        x = x.transpose(0,1)
        acc = torch.tensor(0.0).to(x.device)
        prediction = torch.argmax(x, dim=2)
        
        for b in range(x.size(0)):
            pred = torch.unique_consecutive(prediction[b,:])
            pred = pred[pred != 0]
            acc += compute_per(pred, y[b, : ny[b]])

        return acc/x.size(0)


    def forward(self, x, y, test=False):

        if self.lstmF:
            x, _ = self.lstm(x)
        x    = self.linear(x)
        
        nlabels, labels = y
        labels = labels + 1

        x = nn.functional.log_softmax(x, dim=2)
        x = x.transpose(0,1) # BTC -> TBC for ctc loss function

        input_lengths = torch.full(size=(x.size(1),), fill_value = x.size(0), dtype=labels.dtype)

        loss = self.criterion(x, labels, input_lengths, nlabels)
        acc = self.decode(x, labels, nlabels) if test else None
        
        return (loss, acc)


    def save_params(self, exDir, n_updates):
        if self.lstmF:
            outDict = {'updateNo': n_updates, 'linear': self.linear.state_dict(), 'lstm': self.lstm.state_dict()}
        else:
            outDict = {'updateNo': n_updates, 'linear': self.linear.state_dict()}
        
        torch.save(outDict, join(exDir, 'classifier_params_'+str(n_updates)+'.tar'))
        print('Saving model at {} updates'.format(n_updates))


    def load_params(self, exDir, n_updates, device):
        checkpoint = torch.load(join(exDir, 'classifier_params_'+str(n_updates)+'.tar'), map_location=device)
        self.linear.load_state_dict(checkpoint['linear'])
        self.lstm.load_state_dict(checkpoint['lstm'])

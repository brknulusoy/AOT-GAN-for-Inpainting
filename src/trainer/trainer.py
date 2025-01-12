import os
import wandb
import importlib
from tqdm import tqdm
from glob import glob

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from data import create_loader 
from loss import loss as loss_module
from .common import timer, reduce_loss_dict


class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        # Detect anomalies
        torch.autograd.set_detect_anomaly(True)

        # setup data set and data loader
        self.dataloader = create_loader(args)

        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        self.adv_loss = getattr(loss_module, args.gan_type)()

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        net = importlib.import_module('model.'+args.model)
        
        self.netG = net.InpaintGenerator(args).cuda()
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))

        self.netD = net.Discriminator().cuda()
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))

        self.load()
        if args.distributed:
            self.netG = DDP(self.netG, device_ids= [args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids= [args.local_rank], output_device=[args.local_rank])
        
        if args.tensorboard: 
            wandb.tensorboard.patch(root_logdir=os.path.join(args.save_dir, 'log'), pytorch=True)
            wandb.init(project="tds-aotgan")
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            

    def load(self):
        try: 
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'G*.pt'))))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0: 
                print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 
        
        try: 
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'D*.pt'))))[-1]
            self.netD.load_state_dict(torch.load(dpath, map_location='cuda'))
            if self.args.global_rank == 0: 
                print(f'[**] Loading discriminator network from {dpath}')
        except: 
            pass

        try: 
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.global_rank == 0: 
                print(f'[**] Loading optimizer from {opath}')
        except: 
            pass


    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            if self.args.distributed:
                torch.save(self.netG.module.state_dict(), 
                    os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
                torch.save(self.netD.module.state_dict(), 
                    os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            else:
                torch.save(self.netG.state_dict(), 
                    os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
                torch.save(self.netD.state_dict(), 
                    os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()}, 
                os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))

    def _fix(self, x):
        IS = x.shape[-1]
        return x.reshape(self.args.block_dimension, 1, IS, IS)

    def add_images(self, images, masks, comps):
        self.writer.add_image('mask', make_grid(self._fix(masks), normalize=True), self.iteration)
        self.writer.add_image('orig', make_grid((self._fix(images)+1.0)/2.0, normalize=True), self.iteration)
        self.writer.add_image('comp', make_grid((self._fix(comps)+1.0)/2.0, normalize=True), self.iteration)  

    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        if self.args.global_rank == 0: 
            pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
            timer_data, timer_model = timer(), timer()
        
        for idx in pbar:
            self.iteration += 1
            images, masks = next(self.dataloader)
            images, masks = images.cuda(), masks.cuda()
            images_masked = (images * (1 - masks).float()) + masks

            if self.args.global_rank == 0: 
                timer_data.hold()
                timer_model.tic()

            pred_imgs = self.netG(images_masked, masks)
            comp_imgs = (1 - masks) * images + masks * pred_imgs

            # TODO: Convert to correct methods
            IS = images.shape[-1]
            images = images.reshape((1, 1, self.args.block_dimension * IS, IS))
            masks = masks.reshape((1, 1, self.args.block_dimension * IS, IS))
            pred_imgs = pred_imgs.reshape((1, 1, self.args.block_dimension * IS, IS))
            comp_imgs = comp_imgs.reshape((1, 1, self.args.block_dimension * IS, IS))

            losses = {}
            # reconstruction losses
            for name, weight in self.args.rec_loss.items(): 
                losses[name] = weight * self.rec_loss_func[name](pred_imgs, images)

            # adversarial loss 
            dis_loss, gen_loss = self.adv_loss(self.netD, comp_imgs, images, masks)
            losses[f"advg"] = gen_loss * self.args.adv_weight

            # temporal losses
            # TODO: Convert to correct methods
            comp_imgs = comp_imgs.reshape((1, self.args.block_dimension, 1, IS, IS))
            losses["temporal"] = self.args.temp_weight * torch.var(comp_imgs, axis=1).sum()
            
            # backforward 
            self.optimG.zero_grad()
            self.optimD.zero_grad()
            sum(losses.values()).backward()
            losses[f"advd"] = dis_loss 
            dis_loss.backward()
            self.optimG.step()
            self.optimD.step()

            if self.args.global_rank == 0:
                timer_model.hold()
                timer_data.tic()

            # logs
            scalar_reduced = reduce_loss_dict(losses, self.args.world_size)
            if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0): 
                pbar.update(self.args.print_every)
                description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                    if self.args.tensorboard: 
                        self.writer.add_scalar(key, val.item(), self.iteration)
                pbar.set_description((description))
                if self.args.tensorboard:
                    self.add_images(images, masks, comp_imgs)

            if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0: 
                self.save()


import sys
import os
from .base import Base
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

device = torch.device("cuda")

class AGEM(Base):

    def train(self):
        self.model.train()
        train_step_perepoch = len(self.train_loader)
        C, _ = self.train_loader.dataset.get_ClsName()
        C = C.to(device)

        criterion_XE = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam([{'params': self.model.parameters(),'lr': self.lr}])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10*train_step_perepoch, gamma=0.1)

        if self.mode == "mst":
            text_fmask = torch.zeros(len(C), dtype=torch.long).to(device)
            text_fmask[(len(C)//self.phase_matrix.shape[0])*self.phase: (len(C)//self.phase_matrix.shape[0])*(self.phase+1)] = 1
            text_fmask.view(1, -1)

        resume_ckpt = None
        for pt_file in os.listdir(self.logging_dir):
            if os.path.splitext(pt_file)[1] == ".pt":
                resume_ckpt = self.logging_dir + pt_file
        if resume_ckpt is not None:
            checkpoint = torch.load(resume_ckpt)
            self.model.load_state_dict(checkpoint['model'])
            if checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint['lr_scheduler'] is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] if self.mode == "ost" else -1
        else:
            start_epoch = -1

        scaler = GradScaler()
        replay_loader = iter(self.replay_loader)
        for j in range(start_epoch+1, self.epoch):
            running_loss = 0.0
            self.model.train()
            
            for k, data in enumerate(self.train_loader):
                global_step = self.step_start + j*train_step_perepoch + k

                I = data['image'].to(device)
                L = data['target'].to(device)
                try:
                    replay = next(replay_loader)
                except:
                    replay_loader = iter(self.replay_loader)
                    replay = next(replay_loader)
                RI = data['image'].to(device)
                RC = data['caption'].to(device)

                with autocast():
                    #replayed part
                    optimizer.zero_grad()
                    ref_image_features, _ = self.model.encode_image(RI)
                    ref_image_features = ref_image_features / ref_image_features.norm(dim=-1, keepdim=True)
                    ref_text_features, _ = self.model.encode_text(RC)
                    ref_text_features = ref_text_features / ref_text_features.norm(dim=-1, keepdim=True)

                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * ref_image_features @ ref_text_features.t()
                    logits_per_text = logits_per_image.t()

                    loss_ref = 1/2 *  \
                        (criterion_XE(logits_per_image, torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)).mean() \
                        + criterion_XE(logits_per_text, torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)).mean())
                    
                    #loss_ref.backward(retain_graph=True)
                    scaler.scale(loss_ref).backward(retain_graph=True)
                    reference_gradients_list = [
                        (
                            p.grad.view(-1)
                            if p.grad is not None
                            else torch.zeros(p.numel(), device=device)
                        )
                        for n, p in self.model.named_parameters()
                    ]
                    reference_gradients = torch.cat(reference_gradients_list)
                    optimizer.zero_grad()

                    image_features, _ = self.model.encode_image(I)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features, _ = self.model.encode_text(C)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    if self.mode == "mst":
                        logits_per_image = logits_per_image.masked_fill(text_fmask == 0, -1e4)
                    
                    loss = criterion_XE(logits_per_image, L).mean()

                optimizer.zero_grad()
                total_loss = loss
                running_loss += total_loss.data
                scaler.scale(total_loss).backward()
                #total_loss.backward(retain_graph=True)
                #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)
                with torch.no_grad():
                    current_gradients_list = [
                        (
                            p.grad.view(-1)
                            if p.grad is not None
                            else torch.zeros(p.numel(), device=device)
                        )
                        for n, p in self.model.named_parameters()
                    ]
                    current_gradients = torch.cat(current_gradients_list)
                    dotg = torch.dot(current_gradients, reference_gradients)
                    if dotg < 0:
                        alpha2 = dotg / torch.dot(
                            reference_gradients, reference_gradients
                        )
                        grad_proj = current_gradients - reference_gradients * alpha2

                        count = 0
                        for n, p in self.model.named_parameters():
                            n_param = p.numel()
                            if p.grad is not None:
                                p.grad.copy_(grad_proj[count : count + n_param].view_as(p))
                            count += n_param
                scaler.step(optimizer)
                scaler.update()

                self.writer.add_scalar('batch_loss', total_loss.data, global_step=global_step)
                self.writer.add_scalar('XE_loss', loss.data, global_step=global_step)
                if self.mode == "mst":
                    print('Gstep: %6d; epoch: %d; [%5d] loss: %.3f ; learning_rate: %f' 
                            % (global_step, (self.epoch*self.phase+j), k, total_loss.data,
                            optimizer.param_groups[0]['lr']))
                else:
                    print('Gstep: %6d; epoch: %d; [%5d] loss: %.3f; learning_rate: %f' 
                            % (global_step, j, k, total_loss.data, 
                            optimizer.param_groups[0]['lr']))
                    lr_scheduler.step()
            
            if self.mode == "ost":
                log_txt = self.logging_dir + f"{self.method}_{self.mode}.txt"
                f = open(log_txt, 'a')
                temp = sys.stdout
                sys.stdout = f                

                acc, _, _, _ = self.test(is_zs=False)
                zs_acc, _, _, _ = self.test(is_zs=True)
                self.writer.add_scalar('ut_acc', 100*acc, global_step=j)
                self.writer.add_scalar('zs_acc', 100*zs_acc, global_step=j)
                self.writer.add_scalar('a_acc', 100*(zs_acc+acc)/2, global_step=j)
                print('ut acc @ %d: %.2f' %(j, 100*acc))
                print('zs acc @ %d: %.2f' %(j, 100*zs_acc))
                print('a acc @ %d: %.2f' %(j, 100*(zs_acc+acc)/2))
                
                i2t1, i2t5, i2t10, t2i1, t2i5, t2i10 = self.retri(is_flickr=False)
        
                self.writer.add_scalar('coco_i2t@1', i2t1, global_step=j)
                self.writer.add_scalar('coco_i2t@5', i2t5, global_step=j)
                self.writer.add_scalar('coco_i2t@10', i2t10, global_step=j)
                self.writer.add_scalar('coco_t2i@1', t2i1, global_step=j)
                self.writer.add_scalar('coco_t2i@5', t2i5, global_step=j)
                self.writer.add_scalar('coco_t2i@10', t2i10, global_step=j)
                
                i2t1, i2t5, i2t10, t2i1, t2i5, t2i10 = self.retri(is_flickr=True)
                self.writer.add_scalar('flickr_i2t@1', i2t1, global_step=j)
                self.writer.add_scalar('flickr_i2t@5', i2t5, global_step=j)
                self.writer.add_scalar('flickr_i2t@10', i2t10, global_step=j)
                self.writer.add_scalar('flickr_t2i@1', t2i1, global_step=j)
                self.writer.add_scalar('flickr_t2i@5', t2i5, global_step=j)
                self.writer.add_scalar('flickr_t2i@10', t2i10, global_step=j)

                sys.stdout = temp
                #if self.save_dir is not None:
                #    self.save(j)
                self.save(j, optimizer, lr_scheduler)
                
        if self.mode == "mst":
            log_txt = self.logging_dir + f"{self.method}_{self.mode}.txt"
            f = open(log_txt, 'a')
            temp = sys.stdout
            sys.stdout = f   

            acc, cls_acc, _, _ = self.test(is_zs=False)
            
            acc, bwt, _ = self.get_acc_bwt(cls_acc)

            zs_acc, _, _, _ = self.test(is_zs=True)
            self.writer.add_scalar('ut_acc', 100*acc, global_step=self.phase)
            self.writer.add_scalar('ut_bwt', 100*bwt, global_step=self.phase)
            self.writer.add_scalar('zs_acc', 100*zs_acc, global_step=self.phase)
            self.writer.add_scalar('a_acc', 100*(zs_acc+acc)/2, global_step=self.phase)
            #writer.add_scalar('in_phase_acc', in_phase_acc, global_step=(epoch*phase+j))
            print('ut acc @ %d: %.2f' %(self.phase, 100*acc))
            print('ut bwt @ %d: %.2f' %(self.phase, 100*bwt))
            print('zs acc @ %d: %.2f' %(self.phase, 100*zs_acc))
            print('a acc @ %d: %.2f' %(self.phase, 100*(zs_acc+acc)/2))
            #print('in_phase_acc acc @ %d: %.3f' %((epoch*phase+j), in_phase_acc))
            i2t1, i2t5, i2t10, t2i1, t2i5, t2i10 = self.retri(is_flickr=False)

            self.writer.add_scalar('coco_i2t@1', i2t1, global_step=self.phase)
            self.writer.add_scalar('coco_i2t@5', i2t5, global_step=self.phase)
            self.writer.add_scalar('coco_i2t@10', i2t10, global_step=self.phase)
            self.writer.add_scalar('coco_t2i@1', t2i1, global_step=self.phase)
            self.writer.add_scalar('coco_t2i@5', t2i5, global_step=self.phase)
            self.writer.add_scalar('coco_t2i@10', t2i10, global_step=self.phase)
            
            i2t1, i2t5, i2t10, t2i1, t2i5, t2i10 = self.retri(is_flickr=True)
            self.writer.add_scalar('flickr_i2t@1', i2t1, global_step=self.phase)
            self.writer.add_scalar('flickr_i2t@5', i2t5, global_step=self.phase)
            self.writer.add_scalar('flickr_i2t@10', i2t10, global_step=self.phase)
            self.writer.add_scalar('flickr_t2i@1', t2i1, global_step=self.phase)
            self.writer.add_scalar('flickr_t2i@5', t2i5, global_step=self.phase)
            self.writer.add_scalar('flickr_t2i@10', t2i10, global_step=self.phase)  

            self.writer.add_figure('Phase%d'%self.phase, figure=self.plot_confusion_matrix())

            sys.stdout = temp
            #if self.save_dir is not None:
            self.save(self.phase)
            
            return self.model, global_step, self.phase_matrix
        else:
            return self.model
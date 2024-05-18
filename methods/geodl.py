import sys
from .base import Base
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from .GFK_distill_normalise import GFK

device = torch.device("cuda")

class GeoDL(Base):

    def train(self, geodl_alpha):
        gfk = GFK()

        self.model.train()
        train_step_perepoch = len(self.train_loader)
        C, _ = self.train_loader.dataset.get_ClsName()
        C = C.to(device)

        criterion_XE = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam([{'params': self.model.parameters(),'lr': self.lr}])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10*train_step_perepoch, gamma=0.1)

        with torch.no_grad():
            text_base, _ = self.teacher.encode_text(C)

        if self.mode == "mst":
            text_fmask = torch.zeros(len(C), dtype=torch.long).to(device)
            text_fmask[(len(C)//self.phase_matrix.shape[0])*self.phase: (len(C)//self.phase_matrix.shape[0])*(self.phase+1)] = 1
            text_fmask.view(1, -1)
        
        scaler = GradScaler()
        for j in range(self.epoch):
            running_loss = 0.0
            self.model.train()
            
            for k, data in enumerate(self.train_loader):
                global_step = self.step_start + j*train_step_perepoch + k

                I = data['image'].to(device)
                L = data['target'].to(device)

                with torch.no_grad():
                    image_base, _ = self.teacher.encode_image(I)
                
                with autocast():
                    image_features, _ = self.model.encode_image(I)
                    text_features, _ = self.model.encode_text(C)

                    try:
                        if args.part == "wm":
                            dist_image = gfk.fit(image_features, image_base)
                            if self.mode == "mst":
                                dist_text = gfk.fit(text_features[:10*(self.phase+1)], text_base[:10*(self.phase+1)])
                            else:
                                dist_text = gfk.fit(text_features, text_base)
                        elif args.part == "io":
                            dist_image = gfk.fit(image_features, image_base)
                            dist_text = torch.zeros_like(dist_image)
                        elif args.part == "to":
                            if self.mode == "mst":
                                dist_text = gfk.fit(text_features[:10*(self.phase+1)], text_base[:10*(self.phase+1)])
                            else:
                                dist_text = gfk.fit(text_features, text_base)
                            dist_image = torch.zeros_like(dist_text)                       
                        
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        logit_scale = self.model.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        if self.mode == "mst":
                            logits_per_image = logits_per_image.masked_fill(text_fmask == 0, -1e4)
                        loss = criterion_XE(logits_per_image, L).mean()
                        total_loss = loss + geodl_alpha*dist_image + geodl_alpha*dist_text
                    except:
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        logit_scale = self.model.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        if self.mode == "mst":
                            logits_per_image = logits_per_image.masked_fill(text_fmask == 0, -1e4)
                        loss = criterion_XE(logits_per_image, L).mean()
                        dist_image = torch.zeros_like(loss)
                        dist_text = torch.zeros_like(loss)
                        total_loss = loss
                
                optimizer.zero_grad()
                running_loss += total_loss.data
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()  

                self.writer.add_scalar('batch_loss', total_loss.data, global_step=global_step)
                self.writer.add_scalar('XE_loss', loss.data, global_step=global_step)
                self.writer.add_scalar('dist_image', dist_image.data, global_step=global_step)
                self.writer.add_scalar('dist_text', dist_text.data, global_step=global_step)
                self.writer.add_scalars('loss_compare', 
                    {'total_loss': total_loss.data,
                    'XE_loss': loss.data, 
                    'dist_image': dist_image.data,
                    'dist_text': dist_text.data,
                    }, 
                    global_step=global_step)
                if self.mode == "mst":
                    print('Gstep: %6d; epoch: %d; [%5d] loss: %.3f (%.2f, %.2f, %.2f); learning_rate: %f' 
                            % (global_step, (self.epoch*self.phase+j), k, total_loss.data, loss.data, dist_image.data, dist_text.data,
                            optimizer.param_groups[0]['lr']))
                else:
                    print('Gstep: %6d; epoch: %d; [%5d] loss: %.3f (%.2f, %.2f, %.2f); learning_rate: %f' 
                            % (global_step, j, k, total_loss.data, loss.data, dist_image.data, dist_text.data,
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
                if self.save_dir is not None:
                    self.save(j)
                
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
            if self.save_dir is not None:
                self.save(self.phase)
            
            return self.model, global_step, self.phase_matrix
        else:
            return self.model
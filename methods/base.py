from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

device = torch.device("cuda")

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])
        
        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                            torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
        
def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                            torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

class Base(object):
    def __init__(self, input_params):
        self.epoch = input_params["train_epoch"]
        self.step_start = input_params["global_step"]
        self.train_loader = input_params["train_loader"]
        self.replay_loader = input_params["replay_loader"]
        self.test_loader = input_params["test_loader"]
        self.zs_loader = input_params["zs_loader"]
        self.coco_loader = input_params["coco_loader"]
        self.flickr_loader = input_params["flickr_loader"]
        self.writer = input_params["writer"]
        self.model = input_params["model"]
        self.teacher = input_params["teacher"]
        self.embed_dim = input_params["embed_dim"]
        self.save_dir = input_params["save_dir"]
        self.logging_dir = input_params["logging_dir"]
        self.method = input_params["method"]
        self.mode = input_params["mode"]
        self.part = input_params["part"]
        self.lr = input_params["lr"]
        
        if "phase" in input_params.keys():
            self.phase = input_params["phase"]
        if "phase_matrix" in input_params.keys():
            self.phase_matrix = input_params["phase_matrix"]

    def save(self, epoch):
        model_dict = self.model.state_dict()
        data = {k: v for k, v in model_dict.items()}
        save_path = os.path.join(self.save_dir, str(epoch)+'.npz')
        print((" Saving the model to %s..." % (save_path)))
        np.savez(save_path, model=data)
        print("Model saved.")
        # rfile=os.path.join(save_dir, str(epoch-1)+".npz")
        # if os.path.exists(rfile):
        #     os.remove(rfile)

    #CLS and Retri metrics
    def test(self, is_zs=False):
        print("testing...")
        
        if is_zs:
            dataloader = self.zs_loader
        else:
            dataloader = self.test_loader
        
        C, _ = dataloader.dataset.get_ClsName()

        num_correct= 0
        cls_count = torch.zeros(len(C)).to(device)
        cls_correct = torch.zeros(len(C)).to(device)
        self.model.eval()
        with torch.no_grad():
            C = C.to(device)
            text_features, _ = self.model.encode_text(C)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            for k, data in tqdm(enumerate(dataloader)):
                I = data['image'].to(device)
                L = data['target'].to(device)
                
                image_features, _ = self.model.encode_image(I)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)
                pred = logits_per_image.argmax(dim=1)
                
                num_correct += torch.eq(pred, L).sum().float().item()
                
                for i in L:
                    cls_count[i] += 1
                for i,j in enumerate(torch.eq(pred, L)):
                    cls_correct[L[i]] += j
            
            acc = num_correct/len(dataloader.dataset)
            cls_acc = cls_correct/cls_count   
        return acc, cls_acc, cls_correct, cls_count

    def retri(self, is_flickr=False):
        print("retrieval...")
        
        if is_flickr:
            dataloader = self.flickr_loader
        else:
            dataloader = self.coco_loader
        embed_dim = self.embed_dim
        self.model.eval()
        
        datalen = len(dataloader.dataset)
        img_embs = np.zeros((datalen, embed_dim))
        cap_embs = np.zeros((datalen, embed_dim))
        batch_size = dataloader.batch_size
        
        with torch.no_grad():
            for k, data in enumerate(tqdm(dataloader)):
                I = data['image'].to(device)
                S = data['caption'].to(device)
                
                image_features, _ = self.model.encode_image(I)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features, _ = self.model.encode_text(S)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)   
                
                img_embs[k*batch_size: k*batch_size+image_features.shape[0]] = image_features.data.cpu().numpy().copy()
                cap_embs[k*batch_size: k*batch_size+text_features.shape[0]] = text_features.data.cpu().numpy().copy()

                del I, S, image_features, text_features
        
        results = []
        
        if is_flickr:
            r, rt = i2t(img_embs,
                        cap_embs, 
                        measure='cosine',
                        return_ranks=True)
            #print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti = t2i(img_embs,
                        cap_embs, 
                        measure='cosine',
                        return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            #print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]       
        else:
            for i in range(5):
                r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                            cap_embs[i * 5000:(i + 1) * 5000], 
                            measure='cosine',
                            return_ranks=True)
                #print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                            cap_embs[i * 5000:(i + 1) * 5000], 
                            measure='cosine',
                            return_ranks=True)
                if i == 0:
                    rt, rti = rt0, rti0
                #print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                #print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]
                
        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.2f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.2f" % mean_metrics[11])
        print("Image to text: %.2f %.2f %.2f %.2f %.2f" %
            mean_metrics[:5])
        print("Average t2i Recall: %.2f" % mean_metrics[12])
        print("Text to image: %.2f %.2f %.2f %.2f %.2f" %
            mean_metrics[5:10]) 
            
        i2t1 = mean_metrics[0]
        i2t5 = mean_metrics[1]
        i2t10 = mean_metrics[2]

        t2i1 = mean_metrics[5]
        t2i5 = mean_metrics[6]
        t2i10 = mean_metrics[7]
    
        return  i2t1, i2t5, i2t10, t2i1, t2i5, t2i10

    #This is for MST metrics
    def get_acc_bwt(self, cls_acc):
        BWT = np.zeros(self.phase+1)
        for i in range(8):
            self.phase_matrix[self.phase][i] = cls_acc[10*i: 10*(i+1)].mean()
            if i < self.phase:
                BWT[i] = self.phase_matrix[self.phase][i] - self.phase_matrix[i][i]
        acc = self.phase_matrix[self.phase][:self.phase+1].mean()
        bwt = BWT.mean()
        del BWT
        return acc, bwt, self.phase_matrix

    def plot_confusion_matrix(self):
        matplotlib.use('agg')
        fig = plt.figure()
        plt.imshow(self.phase_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.tight_layout()
        
        fmt = '.2f'
        thresh = self.phase_matrix.max() / 2.
        for i, j in itertools.product(range(self.phase_matrix.shape[0]), range(self.phase_matrix.shape[1])):
            plt.text(j, i, format(100*self.phase_matrix[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if self.phase_matrix[i, j] > thresh else "black")    
        return fig
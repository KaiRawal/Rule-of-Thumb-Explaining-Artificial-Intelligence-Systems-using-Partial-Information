import torch
import torch.nn as nn
import numpy as np
from numpy import prod
from torch.optim.swa_utils import AveragedModel, SWALR


class RoT(torch.nn.Module):
    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False, no_a_b=False):
        super().__init__()
        if not no_a_b:
            self.a = nn.Parameter(torch.zeros((classes,)+sample_shape, requires_grad=True))
            self.b = nn.Parameter(torch.zeros((classes,)+sample_shape, requires_grad=True))
        self.g = nn.Parameter(torch.zeros(classes, requires_grad=True))
        self.classes=classes
        if use_BCE_loss is False:
            self.objective = torch.nn.CrossEntropyLoss(reduction='sum')
        else:
            self.objective = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.dropout_rate = dropout_rate
        if not no_a_b:
            self.weights = (self.a, self.b, self.g)
        self.training_loss=None
        self.use_BCE_loss=use_BCE_loss
        self.swa_model=None

    def forward(self,x):
        "Warning. Forward should only be used at eval, at training use stochastic importance"
        return self.importance(x)

    mins=-np.inf
    maxs=np.inf
    def importance(self,points):
        return self.a[None]*(points[:,None]+self.b[None])

    def stochastic_importance(self,points):
        imp=self.importance(points)
        mask=(torch.rand(points.shape[0:1]+imp.shape[2:])>self.dropout_rate).float()
        return mask[:,None]*imp

    def pretrain_loss(self,points,classifier_response):
        imp=self.importance(points)
        imp=imp.reshape(imp.shape[0],imp.shape[1],-1)+self.g[:,None]
        cl=classifier_response.repeat(imp.shape[2],1).T
        loss=self.objective(imp,cl)
        loss/=prod(self.b.shape[1:])
        loss+=self.objective(self.g.repeat(imp.shape[0],1),classifier_response)
        return loss/2


    def loss(self,points,target):
        response=self.stochastic_importance(points)
        response=response.reshape(points.shape[0],self.classes,-1).sum(-1)
        response=response+self.g[None]
        print(f'{points.shape=}')
        print(f'{response.shape=}')
        print(f'{target.shape=}')
        print('SUPER\n')
        return self.objective(response,target)

    def fit_project(self,mins,maxs):
        self.mins=mins
        self.maxs=maxs

    def project(self):
        #self.b.data[:]=0
        self.b.data=torch.min(torch.max(self.b.data, self.mins), self.maxs)

    def training_loop(self,loss,points,classifier_response,optimiser,epochs=1,batch_size=200,scheduler=False, write_file=None):
        self.training_loss=np.zeros(epochs)
        burn_in=epochs//10+1
        for e in range(epochs):
            # print('l74')
            shuff=torch.randperm(classifier_response.shape[0])
            for i in range(points.shape[0] // batch_size + 1):
                # print('l77')
                upper = min(points.shape[0], batch_size * (i + 1))
                lower = batch_size * i
                if lower==upper:
                    break

                shuff_inner=shuff[lower:upper]
                target=classifier_response[shuff_inner]
                features=points[shuff_inner]

                l=loss(features,target)
                l/=batch_size
                if self.use_BCE_loss:
                    l/=self.classes
                self.training_loss[e]+=l
                l.backward()
                optimiser.step()
                optimiser.zero_grad()
                # self.project()
            if e < 10 or (e % 10 == 1 and e < 100) or (e % 50 == 1):  # print loss every 10 epochs
                print(f'\t\tEpoch: {e}, Loss: {self.training_loss[e]}')
                if write_file is not None:
                    with open(write_file, 'a') as f:
                        f.write(f'\tepoch-loss,{e},{self.training_loss[e]}\n')
            if e>burn_in:
                self.swa_model.update_parameters(self)
            elif e == burn_in:
                self.swa_model=AveragedModel(self)

        if scheduler:
            scheduler.step()
        result = self.training_loss
        self.training_loss/=features.shape[0]/batch_size
        return result

    def fit(self,points,classifier_response,epochs,batch_size,lr=1e-4, write_file=None):
        if write_file is not None:
            with open(write_file, 'w') as f:
                f.write(f"{points.shape=}\n")
                f.write(f"{classifier_response.shape=}\n")
                f.write(f"{epochs=}\n")
                f.write(f"{batch_size=}\n")
                f.write(f"{self.dropout_rate=}\n")
                f.write(f"{lr=}\n")
                if hasattr(self, 'use_sgd'):
                    f.write(f"{self.use_sgd=}\n")
                if hasattr(self, 'l1_penalty'):
                    f.write(f"{self.l1_penalty=}\n")
                if hasattr(self, 'cube'):
                    f.write(f"{self.cube=}\n")
                if hasattr(self, 'essay_length_mean'):
                    f.write(f"{self.essay_length_mean=}\n")
                f.write("==============\n\n")
        points = points.to(torch.float)
        assert points.shape[0]==classifier_response.shape[0]
        # assert points.shape[1]==self.a.shape[1]
        # self.fit_project(-points.max(0)[0],-points.min(0)[0])
        self.b[None,:].data= torch.zeros(points.shape[2]) # -torch.Tensor(points).mean(0)
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr)
        if self.use_sgd:
            optimiser = torch.optim.SGD(self.parameters(), lr=lr)
        drop_out = self.dropout_rate
        self.dropout_rate=0
        # print('TRAINING LOOP 1')
        self.training_loop(self.loss,points,classifier_response,optimiser,5,batch_size,write_file=write_file)
        self.dropout_rate=drop_out
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr,weight_decay=0.01)
        if self.use_sgd:
            optimiser = torch.optim.SGD(self.parameters(), lr=lr,weight_decay=0.01)
        # print('TRAINING LOOP 2')
        self.training_loop(self.loss,points,classifier_response,optimiser,epochs,batch_size,write_file=write_file)

    def continue_fit(self, points, classifier_response, epochs, batch_size, lr=1e-4, append_file=None):
        if append_file is not None:
            with open(append_file, 'a') as f:
                f.write("\n-------------\n")
                f.write(f"{points.shape=}\n")
                f.write(f"{epochs=}\n")
                f.write(f"{batch_size=}\n")
                f.write(f"{lr=}\n")
                if hasattr(self, 'use_sgd'):
                    f.write(f"{self.use_sgd=}\n")
                if hasattr(self, 'l1_penalty'):
                    f.write(f"{self.l1_penalty=}\n")
                if hasattr(self, 'cube'):
                    f.write(f"{self.cube=}\n")
                if hasattr(self, 'essay_length_mean'):
                    f.write(f"{self.essay_length_mean=}\n")
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        if self.use_sgd:
            optimiser = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=0.01)
        return self.training_loop(self.loss, points, classifier_response, optimiser, epochs, batch_size)

    def averaged_explainer(self):
        return list(self.swa_model.children())[0]

    def score(self,points):
        imp=self.importance(points).detach()
        score=imp.reshape(imp.shape[0],imp.shape[1],-1).sum(-1)
        score+=self.g[None,:]
        return score

    def predict(self,points):
        score=self.score(points)
        return score.argmax(1)

    def ordered_predict(self,points,order):
        import math
        assert order.min()==0
        imp=self.importance(points).detach()#.numpy()
        assert order.shape[0]==imp.shape[0]
        assert order.shape[1:]==imp.shape[2:]
        assert order.max()==prod(imp.shape[2:])-1

        acc=torch.zeros(imp.shape[0],self.classes)
        pred=torch.zeros((imp.shape[0],prod(imp.shape[2:])+1),dtype=int)
        imp=imp.reshape(imp.shape[0],imp.shape[1],-1)
        order=order.reshape(imp.shape[0],-1)
        imp=imp.permute(0,2,1)
        acc[:]=self.g[None].detach()
        pred[:,0]=acc.argmax(1)
        for i in range(order.shape[1]):
            tmp=imp[(np.arange(imp.shape[0]),order[:,i])]
            assert tmp.shape==acc.shape
            imp[(np.arange(imp.shape[0]),order[:,i])]=0
            acc+=tmp
            pred[:,i+1]=acc.argmax(1)
        return pred

    def get_order(self,points):
        imp=self.importance(points).detach().numpy()
        imp=np.abs(imp).sum(1)
        old_shape=imp.shape
        imp=imp.reshape(imp.shape[0],-1)
        order=np.argsort(imp,1)[:,::-1].copy()
        order=order.reshape(old_shape)
        return order

    def score_ordering(self,points,labels,order, metric=None):
        if metric is None:
            metric=lambda tp,fp,fn,tn: (tp+tn)/(tp+fp+fn+tn)
        pred=self.ordered_predict(points,order)
        tp=((pred==1).float()*(labels==1).float()[:,None]).sum(0)
        tn=((pred==0).float()*(labels==0).float()[:,None]).sum(0)
        fp=((pred==1).float()*(labels==0).float()[:,None]).sum(0)
        fn=((pred==0).float()*(labels==1).float()[:,None]).sum(0)
        return metric(tp,fp,fn,tn)


class RoT_text(RoT):
    """
       Assumes datapoints are in the form: token x embedding
     """
    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False, l1_penalty=0.0, sgd=False):
        super().__init__(classes,sample_shape,dropout_rate,use_BCE_loss,no_a_b=True)
        self.a = nn.Parameter(torch.zeros((classes, sample_shape[1]),requires_grad=True))
        self.b = nn.Parameter(torch.zeros((classes, 1),requires_grad=True))
        # print('INITIALISATION')
        # print(self.a.shape)
        # print(self.a.shape)
        # self.objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.weights=(self.a,self.b,self.g)
        self.l1_penalty = l1_penalty
        self.use_sgd = sgd
        self.cube = False
        self.essay_length_mean = False
        self.ttanh = False


    # def write_hyperparams_to_file(self, file_path):
    #     """
    #     Writes the hyperparameters of the model to a specified file.

    #     Args:
    #         file_path (str): The path to the file where hyperparameters will be written.
    #     """
    #     hyperparams = {
    #         'classes': self.classes,
    #         'dropout_rate': self.dropout_rate,
    #         'use_BCE_loss': self.use_BCE_loss,
    #         'l1_penalty': self.l1_penalty,
    #         'use_sgd': self.use_sgd
    #     }

    #     with open(file_path, 'w') as f:
    #         for key, value in hyperparams.items():
    #             f.write(f"{key}: {value}\n")

    def score(self,points):
        imp=self.importance(points).detach()
        response_mean = imp

        if self.essay_length_mean:
            # imp=imp.mean(dim=2)  # mean per essay - normalise by number of tokens (length)
            # Calculate the sum over the 512-element dimension
            response_sum = imp.sum(dim=2)
            # Calculate DELTA: the number of all-zero rows in the 512-element dimension
            zero_rows = (imp == 0).all(dim=3).sum(dim=2)
            # Calculate the modified length
            modified_length = imp.shape[2] - zero_rows
            # Avoid division by zero by ensuring modified_length is at least 1
            modified_length = torch.clamp(modified_length, min=1)
            # Calculate the mean using the modified length
            response_mean = response_sum / modified_length.unsqueeze(-1)


        score=response_mean.sum(1)
        score+=self.g[None,:]
        if self.cube:
            score = torch.pow(score, 3)

        return score

    def loss(self,points,target):
        response=self.stochastic_importance(points)

        if self.essay_length_mean:
            # response=response.mean(dim=2)  # mean per essay - normalise by number of tokens (length)
            # Calculate the sum over the 512-element dimension
            response_sum = response.sum(dim=2)
            # Calculate DELTA: the number of all-zero rows in the 512-element dimension
            zero_rows = (response == 0).all(dim=3).sum(dim=2)
            # Calculate the modified length
            modified_length = response.shape[2] - zero_rows
            # Avoid division by zero by ensuring modified_length is at least 1
            modified_length = torch.clamp(modified_length, min=1)
            # Calculate the mean using the modified length
            response_mean = response_sum / modified_length.unsqueeze(-1)
            response = response_mean

        response = response.sum(1)
        response = response+self.g[None]
        if self.cube:
            response = torch.pow(response, 3)
        base_loss = self.objective(response, target)
        if self.l1_penalty == 0:
            l1_loss = 0
        else:
            l1_loss = self.l1_penalty * points.shape[0] * self.classes * (self.a.abs().sum() + self.b.abs().sum() + self.g.abs().sum())
        return base_loss + l1_loss

    def importance(self, points):
        # Create a deterministic mask that ensures only those points remain which don't have all features == -1 for a given token
        # imp = self.a[None, :, None, :] * (points[:, None] + self.b[None, :, None, :])
        p = points.reshape(-1,points.shape[2])
        imp = p.mm(self.a.T)+self.b.reshape(-1)[None]
        mask = (points[:, :, 0]==-1)
        imp = imp.reshape([mask.shape[0], mask.shape[1], 2])
        imp[mask] = 0
        return imp
        # return imp


    def stochastic_importance(self, points):
        imp=self.importance(points)
        # importance is 4d (classes is always added as the 1th dimension), because points became 3d
        # points[index, features] -> points[index, token, features]
        token_dim = 1
        mask = (torch.rand(points.shape[0],points.shape[token_dim])>self.dropout_rate)
        imp[mask] = 0
        return imp

    # def fit_project(self,mins,maxs):
    #     mins=mins.min(-1)[0]
    #     mins=mins.min(-1)[0]
    #     maxs=maxs.max(-1)[0]
    #     maxs=maxs.max(-1)[0]
    #     print(mins.shape, maxs.shape,self.b.shape)
    #     assert mins.shape==self.b.shape[1:]
    #     self.mins=mins
    #     self.maxs=maxs

# class RoT_image_mixed(RoT_image):
#     """Compute importance as the product of spatial locations with channels
#     Assumes datapoints are in the form chanel x width x height
#     """
#     def __init__(self,classes,sample_shape,dropout_rate=0.5,use_BCE_loss=False):
#         super().__init__(classes,sample_shape,dropout_rate,use_BCE_loss)
#         self.a_spatial=nn.Parameter(torch.zeros((classes,sample_shape[0]),requires_grad=True))

#     def importance(self,points):
#         return self.a[None,:,None,None]*self.a_spatial[None,None,:,:]*(points[:,None]+self.b[None,:,None,None])


class Linear_regression(RoT):
    def project(self):
        self.b.data[:]=0


class per_point_NAM(torch.nn.Module):
    def __init__(self,classes):
        super().__init__()
        width=128
        self.A=nn.Parameter(torch.randn((width,),requires_grad=True))
        self.A.data/=2
        self.A.data+=3
        self.a=nn.Parameter(torch.randn((width),requires_grad=True))
        self.B=nn.Parameter(torch.zeros((width,classes),requires_grad=True))
        #self.offset=torch.randn(classes,requires_grad=True)
        self.non_lin=torch.nn.SELU()

    def forward(self,x):
        x=self.non_lin(x[:,None]-self.a[None,:])
        x=x*torch.exp(self.A[None,:])
        x=x.mm(self.B)#+self.offset[None,:]
        return x

class per_point_RBF(torch.nn.Module):
    def __init__(self,classes):
        super().__init__()
        width=32
        self.A=nn.Parameter(torch.ones((width,),requires_grad=True))
        self.A.data[:]=0.1
        self.a=nn.Parameter(torch.randn((width),requires_grad=True))
        self.a.data*=5
        self.B=nn.Parameter(torch.zeros((width,classes),requires_grad=True))


    def forward(self,x):
        x=torch.exp(-(x[:,None]-self.a[None,:])**2/self.A[None]**2)
        x=x.mm(self.B)#+self.offset[None,:]
        return x

class per_point_poly(torch.nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.width=6
        self.A=nn.Parameter(torch.zeros((self.width,classes),requires_grad=True))

    def forward(self,x):
        x=x[:,None].pow(torch.arange(self.width)[None,:])
        x=x.mm(self.A)
        return x


class RoT_additive(RoT):
    def __init__(self, classes, sample_shape, dropout_rate=0.5,sub_model=per_point_NAM):
        from itertools import chain
        super().__init__(classes, sample_shape, dropout_rate)
        assert len (sample_shape)==1
        self.fns=[sub_model(classes) for i in range(sample_shape[0])]#Not handling non-vector for now
        for (i,f) in enumerate(self.fns):
            self.add_module('feature '+str(i), f)

    def importance(self, points):
        out=torch.empty(points.shape[0],self.classes,points.shape[1])
        for i in range(points.shape[1]):
            out[:,:,i]=self.fns[i](points[:,i])
        out+=super().importance(points)
        return out


def rand_order(points):
    rand=torch.rand_like(points)
    rand=rand.reshape(points.shape[0],-1)
    order=torch.argsort(rand,1)
    order=order.reshape(points.shape)
    return order

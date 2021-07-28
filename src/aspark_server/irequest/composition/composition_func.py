from composition.basecomp import Compbase

import itertools
import torch

class Comp_factory():
    def get_compfun(compfun):
        funct = ''
        if compfun == 'cls':
            funct = Cls()
        if compfun == 'avg':
            funct = Avg()
        if compfun == 'sum':
            funct = Sum()
        if compfun == 'f_ind':
            funct = F_ind()
        if compfun == 'f_joint':
            funct = F_joint()
        if compfun == 'f_inf':
            funct = F_inf()
        
        return funct

class Cls(Compbase):
    def __init__(self):
        Compbase.__init__(self)
        pass
    def clean_special(self, output, special_mask):
        return output
    def compose(self, output):
        _output = torch.stack([i[:,0] for i in output])
        return torch.mean(_output,0)


class Sum(Compbase):
    def __init__(self):
        Compbase.__init__(self)
    
    def compose(self, output):
        _output = torch.stack([torch.sum(i,1) for i in output])
        return torch.mean(_output, 1)

class Avg(Compbase):
    def __init__(self):
        Compbase.__init__(self)
    def compose(self,output):
        _output = torch.stack([torch.mean(i,1) for i in output])
        return torch.mean(_output,1)

class Fcomp(Compbase):
    def __init__(self, alpha, beta ):
        Compbase.__init__(self)
        self.alpha = alpha
        self.beta = beta  
    
    def _gencompf(self, v1v2):
        # LEFTMOST
        v1v2_sum = torch.sum(v1v2,0)                        # v1 + v2
        v1v2sum_norm = torch.norm(v1v2_sum)                 # norm ( v1 + v2 )

        left_op = torch.divide(v1v2_sum, v1v2sum_norm)      # A / B

        # RIGHTMOST
        v1_norm = torch.norm(v1v2[0])                       # norm ( v1 )
        v2_norm = torch.norm(v1v2[1])                       # norm ( v2 )
        v1_norm_sq = torch.square(v1_norm)                  # Square || v1 ||
        v2_norm_sq = torch.square(v2_norm)                  # Square || v2 ||

        rleft = torch.stack((v1_norm_sq,v2_norm_sq))        # [[ C ], [ D ]]
        rleft = torch.sum(rleft)                            # C + D

        rright = torch.dot(v1v2[0],v1v2[1])                 # <v1,v2>
        right_op =(self.alpha)*rleft - (self.beta)*rright   # alpha E - Beta F

        right_op = torch.sqrt(right_op)                     # Sq root G

        resultado = left_op * right_op                      # H * I

        return resultado
    
    def _sentencefunc(self, sentence):
        _result = self._gencompf(sentence[:2])

        for v_n in sentence[2:]:
            _result = self._gencompf(torch.stack([_result, v_n]))

        return _result
    
    def compose(self, output):
        #try parallelism
        output = torch.stack([torch.stack([self._sentencefunc(layer) for layer in sentence]) for sentence in output])
        output = torch.mean(output,1)
        
        return output


class F_ind(Fcomp):
    def __init__(self):
        Fcomp.__init__(self, alpha=1, beta=0)
        pass

class F_joint(Fcomp):
    def __init__(self):
        Fcomp.__init__(self, alpha=1/4, beta=-1/2)
        pass

class F_inf(Fcomp):
    def __init__(self):
        Fcomp.__init__(self, alpha=1, beta=0)

    def _gencompf(self, v1v2):
        # LEFTMOST
        v1v2_sum = torch.sum(v1v2,0)                        # v1 + v2
        v1v2sum_norm = torch.norm(v1v2_sum)                 # norm ( v1 + v2 )

        left_op = torch.divide(v1v2_sum, v1v2sum_norm)      # A / B

        # RIGHTMOST
        v1_norm = torch.norm(v1v2[0])                       # norm ( v1 )
        v2_norm = torch.norm(v1v2[1])                       # norm ( v2 )
        v1_norm_sq = torch.square(v1_norm)                  # Square || v1 ||
        v2_norm_sq = torch.square(v2_norm)                  # Square || v2 ||

        rleft = torch.stack((v1_norm_sq,v2_norm_sq))        # [[ C ], [ D ]]
        rleft = torch.sum(rleft)                            # C + D

        #Recalculate beta
        _beta = torch.div(torch.min(v1_norm, v2_norm), torch.max(v1_norm, v2_norm))
       
        rright = torch.dot(v1v2[0],v1v2[1])                 # <v1,v2>
        right_op =(self.alpha)*rleft - (_beta)*rright       # alpha E - Beta F

        right_op = torch.sqrt(right_op)                     # Sq root G

        resultado = left_op * right_op                      # H * I

        return resultado
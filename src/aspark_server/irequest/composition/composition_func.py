from composition.basecomp import Compbase

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
    def clean_special(self, output, special_ids, input_ids):
        return output

class Sum(Compbase):
    def __init__(self):
        Compbase.__init__(self)
        pass

class Avg(Compbase):
    def __init__(self):
        Compbase.__init__(self)
        pass

class Fcomp(Compbase):
    def __init__(self, alpha, mu ):
        Compbase.__init__(self)
        self.alpha = alpha
        self.mu = mu
        
    def compose(self):
        pass


class F_ind(Fcomp):
    def __init__(self):
        Fcomp.__init__(self, alpha=1, mu=0)
        pass

class F_joint(Fcomp):
    def __init__(self):
        Fcomp.__init__(self, alpha=1, mu=1)
        pass

class F_inf(Fcomp):
    def __init__(self):
        Fcomp.__init__(self, alpha=1, mu=0)
        pass
    def _recalculate_mu(self):
        self.mu=10
    def compose(self):
        self._recalculate_mu()
        return super(F_inf, self).compose()
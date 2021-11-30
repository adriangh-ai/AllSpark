from abc import abstractmethod
import itertools

import torch

class Compbase():
    def __init__(self):
        pass

    def clean_special(self, output, special_mask):
        """
        Removes special tokens from the word embedding list leaving just token/s representing
        a word from the sentence. It does so by using the special token mask from the tokenizer,
        getting the indices of nonzero values.

        The tensors representing the same sentence on different layers are stacked, as those will
        keep the same dimension through the process.

        Finally, each tensor with all layers of the same sentence is further processes to get only 
        the tensors on selected indices.
        """
        special_mask = ~special_mask.bool()                                   #Invert ids to non-special                                                    
        special_mask = [torch.squeeze                                         #Nonzero indices
                        (torch.stack(i.nonzero(as_tuple=True))) for i in special_mask] 

        output = torch.stack(output,1)                                        #stack same sentence of layers
        #Maps index_select (tensor, dim, tensor) to zipped sentence, 1, mask 
        output = list(itertools.starmap(torch.index_select, zip(output, itertools.repeat(1), special_mask)))
        return output

    @abstractmethod
    def compose(self):
        """
        This is the method specialized in the composition of word embeddings to sentences.
        Sets the contract for the classes to implement it
        """
        pass
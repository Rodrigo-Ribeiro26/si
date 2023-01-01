import sys
sys.path.insert(0,"C:/Users/rodri/Desktop/Mestrado/SIB/si/src")
import itertools

import numpy as np

from si.data.dataset import Dataset


class KMer:
    def __init__(self, k: int = 3, alphabet: str = 'DNA'):
        self.k = k
        self.alphabet = alphabet.upper()

        if self.alphabet == 'DNA':
            self.alphabet = 'ACTG'
        elif self.alphabet == 'PROT':
            self.alphabet = 'FLIMVSPTAY_HQNKDECWRG'
        else:
            self.alphabet = self.alphabet
        
        self.k_mers = None

    
    def fit(self, dataset: Dataset):
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]

        return self

    def _get_kmer(self, sequence: str):
        kmer_count = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence)-self.k +1):
            k_mer = sequence[i:i + self.k]
            kmer_count[k_mer] += 1

        return np.array([kmer_count[k_mer]/len(sequence) for k_mer in self.k_mers])


    def transform(self, dataset: Dataset):
        sequences_kmer = [self._get_kmer(seq) for seq in dataset.X[:, 0]]
        sequences_kmer = np.array(sequences_kmer)

        return Dataset(X=sequences_kmer, y=dataset.y, features=self.k_mers, label=dataset.label)
        

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform(dataset)
"""
Definitions which require Theano to be loaded.

This module is imported within `shim.core.load('theano')`.
"""

from collections.abc import Sized
import logging
import numpy as np
import theano
import theano.sandbox.rng_mrg

logger = logging.getLogger(__file__)

class MRG_RNG(theano.sandbox.rng_mrg.MRG_RandomStream):
    def __init__(self, seed=12345):
        super().__init__(self.normalize_seed(seed))

    @staticmethod
    def normalize_seed(seed):
        # We want to allow seeds
        #   - based on NumPy integers
        #   - Allow seeds bigger than 2147462579 (uint32 goes to about twice this number)
        # Motivation: SeedSequence.generate_state returns an Array[uint32]
        M2 = theano.sandbox.rng_mrg.M2
            # All six seed values must be lower than this number
        if isinstance(seed, Sized) and len(seed) == 1:
            seed = seed[0]
            assert not isinstance(seed, Sized)
        if isinstance(seed, Sized):
            # TODO?: Deal with numbers > M2. I'm unsure of the best way to
            #        deal with dropped bits
            return [int(s) for s in seed]
        else:
            if seed == 0:
                logger.warning("MRG_RNG doesn't allow seeding with 0. Replacing seed with 12345.")
                seed = 12345
            else:
                seed = int(seed)
            # Remark: The rules below are careful not to drop any bits, so any
            #         two different big seed integers will remain different seeds
            if seed <= M2:
                pass
            elif seed.bit_length() < 62:  # Lower bound for seed < M2**2
                seed = [seed % M2 + 1, 0, 0, seed // M2 + 1, 0, 0]  # +1 ensures the value is never 0
            else:
                assert seed <= np.iinfo(np.uint64).max
                seed = [seed % M2 + 1, seed // M2 // M2 + 1, 0, (seed // M2) % M2, 0, 0]
            return seed

# import numpy as np
# seed = np.iinfo(np.uint64).max
# M2 = 2147462579
# (M2**2).bit_length()
# seed = 2**61
# #      443902761
# #
# # seed = M2**2 - 1
# #
# #
# # seed % M2 + 1
# # seed // M2 + 1 - M2
# np.array([seed % M2, (seed // M2) % M2, 0, seed // M2 // M2 + 1, 0, 0]) < M2
# # 1775611043 - M2

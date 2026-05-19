import numpy as np
from JLAB.solver import JLABCode as OriginalJLABCode
from meent.on_numpy.convolution_matrix import to_conv_mat, find_nk_index


class JLABCodeMultiOrder(OriginalJLABCode):
    """
    Extension of original JLABCode that supports arbitrary diffraction order m.
    Does NOT modify the original installed JLAB package.
    """

    def get_diffraction_efficiency(self, n_ridge, m):

        if isinstance(n_ridge, str):
            n_ridge = find_nk_index(n_ridge, self.mat_table, self.wls)

        # Convert ±1 structure → refractive index profile
        self.ucell = (self.ucell + 1) / 2
        self.ucell = self.ucell * (n_ridge**2 - 1.0**2) + 1.0**2

        e_conv_all = to_conv_mat(self.ucell, self.fourier_order)
        o_e_conv_all = to_conv_mat(1 / self.ucell, self.fourier_order)

        de_ri, de_ti = self.solve(self.wls, e_conv_all, o_e_conv_all)

        center = de_ti.shape[0] // 2
        target_index = center + m

        if target_index < 0 or target_index >= len(de_ti):
            raise ValueError(
                f"Requested diffraction order m={m} exceeds Fourier order range."
            )

        return de_ti[target_index]
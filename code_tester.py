import utils.auxiliary as aux
import utils.config as cfg
import utils.case_known_solns_man as ks

current_integral, error, k = aux.unified_adaptive_quadrature(
    f=lambda x: ks.f1(x, t=0.25) * aux.phi_m(2, cfg.ell, x),
    ell=cfg.ell, tol=500
    )
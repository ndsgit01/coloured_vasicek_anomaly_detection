import statsmodels.api as sm
import numpy as np


class VasicekModel:
    """
    Vasicek Model: Generate time series with coloured noise
    """
    def __init__(self, kappa, theta, sigma, horizon, delta_t, r_0, noise):
        """
        kappa -> speed of reversion
        theta -> long term mean level
        sigma -> instantaneous volatility
        horizon -> forecast horizon
        delta_t -> time increment
        r_0 -> starting interest rate
        noise -> function to generate noise series, default: white noise
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.horizon = horizon
        self.delta_t = delta_t
        self.n_intervals = int(horizon / delta_t)
        self.r_0 = r_0
        self.noise_series = noise(self.n_intervals) # noise remains constant once object initialized

    def generate_rate_time_series(self):
        """
        returns -> (np. array of time instances, np. array of rates)
        """
        r = [self.r_0]
        t = [0]
        for i in range(1, self.n_intervals + 1):
            delta_w = self.delta_t**0.5 * self.noise_series[i-1]
            delta_r = self.kappa * (self.theta - r[-1]) * self.delta_t + self.sigma * delta_w
            r_next = r[-1] + delta_r
            r.append(r_next)
            t_next = t[-1] + self.delta_t
            t.append(t_next)
        return np.array(t), np.array(r)
    
class CalibrateVasicek:
    @staticmethod
    def vasicek_em_gls_calibration(r, sigma):
        # GLS fit
        y = r[1:]
        x = r[:-1]
        model = sm.GLS(y, sm.add_constant(x), sigma)
        results = model.fit()
        intercept, slope = results.params
        sd = results.scale**0.5

        # Vasicek params
        dt = 1/(len(r)-1)
        kappa = (1-slope)*(len(r)-1)
        theta = intercept/(1-slope)
        sigma = sd*np.sqrt((len(r)-1))

        return kappa, theta, sigma, results
    
    @staticmethod
    def vasicek_gls_calibration(r, sigma):
        # GLS fit
        y = r[1:]
        x = r[:-1]
        model = sm.GLS(y, sm.add_constant(x), sigma)
        results = model.fit()
        intercept, slope = results.params
        sd = results.scale**0.5

        # Vasicek params
        dt = 1/(len(r)-1)
        kappa = -np.log(slope)*(len(r)-1)
        theta = intercept/(1-slope)
        sigma = sd*np.sqrt((-2*np.log(slope))/(dt*(1-slope**2)))

        return kappa, theta, sigma, results
        
    
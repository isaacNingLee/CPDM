from typing import Tuple, Union
import torch
from torch.nn import functional as F
from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

class MyDDPMScheduler(DDPMScheduler):

    def _my_get_variance(self, t, predicted_variance=None, variance_type=None):
        prev_t = self.previous_timestep(t)

        alphas_cumprod = self.alphas_cumprod.to(device=t.device)
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = torch.where(prev_t >= 0, alphas_cumprod[prev_t], 1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = variance
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance
    
    def my_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        assert model_output.shape[0] == timestep.shape[0] == sample.shape[0], \
            '`timestep` must have the same size in the 0th dimension as both `model_output` and `sample`'
        
        t_adaptive_shape = [timestep.shape[0]] + [1] * (len(sample.shape) - 1)

        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alphas_cumprod = self.alphas_cumprod.to(device=model_output.device, dtype=model_output.dtype)
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = torch.where(prev_t >= 0, alphas_cumprod[prev_t], 1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.view(t_adaptive_shape) ** (0.5) * model_output) / alpha_prod_t.view(t_adaptive_shape) ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t.view(t_adaptive_shape)**0.5) * sample - (beta_prod_t.view(t_adaptive_shape)**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff.view(t_adaptive_shape) * pred_original_sample + current_sample_coeff.view(t_adaptive_shape) * sample

        # 6. Add noise
        device = model_output.device
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
        )
        if self.variance_type == "fixed_small_log":
            variance = self._my_get_variance(t, predicted_variance=predicted_variance).view(t_adaptive_shape) * variance_noise
        elif self.variance_type == "learned_range":
            variance = self._my_get_variance(t, predicted_variance=predicted_variance).view(t_adaptive_shape)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (self._my_get_variance(t, predicted_variance=predicted_variance).view(t_adaptive_shape) ** 0.5) * variance_noise

        variance = torch.where(torch.repeat_interleave(t, torch.numel(variance[0]), dim=0).reshape(variance.shape) > 0, variance, 0.0)
        
        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
    
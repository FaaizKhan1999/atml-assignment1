import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        """
        Wraps a base optimizer (like AdamW) to perform Sharpness-Aware Minimization.
        """
        assert rho >= 0.0, f"Invalid rho, must be non-negative: {rho}"
        
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # Initialize the underlying optimizer with the extracted params
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Calculates the ascent perturbation and adds it to the weights.
        Must be called after the first loss.backward().
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            # scale = rho / ||g||_2
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: 
                    continue
                    
                # Calculate perturbation epsilon
                e_w = p.grad * scale
                # Perturb the weights: theta = theta + epsilon
                p.add_(e_w)
                # Store epsilon for the restoration step
                self.state[p]["e_w"] = e_w

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Restores the original weights and performs the actual optimizer step.
        Must be called after the second loss.backward() at the perturbed weights.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                # Restore original weights: theta = (theta + epsilon) - epsilon
                p.sub_(self.state[p]["e_w"])

        # Base optimizer updates weights using the gradients from the perturbed pass
        self.base_optimizer.step()

        if zero_grad: 
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        """
        Computes the global L2 norm of the gradients.
        """
        shared_device = self.param_groups[0]["params"][0].device
        
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

import torch
import torch.nn as nn

class BlackScholesPINN:
    def __init__(self, Smax=20, Smin=0, E=10, r=0.05, Sigma=0.02, 
                 N_inner=1000, N_boundary=100, device=None, seed=123,dtype = torch.float64):
        self.Smax = Smax
        self.Smin = Smin
        self.E = E
        self.r = r
        self.Sigma = Sigma
        self.N_inner = N_inner
        self.N_boundary = N_boundary
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.dtype = dtype

        self._set_seed()
        self._prepare_data()

    def _set_seed(self):
        torch.manual_seed(self.seed)

    def _prepare_data(self):
        torch.manual_seed(self.seed)

        self.t_inner = torch.rand(self.N_inner, requires_grad=True, device=self.device,dtype=self.dtype)
        self.S_inner = (self.Smax - self.Smin) * torch.rand(self.N_inner, requires_grad=True, device=self.device,dtype=self.dtype) + self.Smin

        self.t_zero = torch.zeros(self.N_boundary, device=self.device, requires_grad=True,dtype=self.dtype)
        self.t_array = torch.rand(self.N_boundary, requires_grad=True, device=self.device,dtype=self.dtype)
        self.S_array = (self.Smax - self.Smin) * torch.rand(self.N_boundary, requires_grad=True, device=self.device,dtype=self.dtype) + self.Smin
        self.S_min_array = torch.zeros(self.N_boundary, device=self.device, requires_grad=True,dtype=self.dtype)
        self.S_max_array = torch.multiply(torch.ones(self.N_boundary, device=self.device, requires_grad=True), self.Smax)

        self.Inner_domain = torch.column_stack((self.t_inner, self.S_inner))
        self.Initial_boundary = torch.column_stack((self.t_zero, self.S_array))
        self.Start_boundary = torch.column_stack((self.t_array, self.S_min_array))
        self.End_boundary = torch.column_stack((self.t_array, self.S_max_array))
        self.End_boundary2 = torch.column_stack((self.t_array, self.S_max_array))

    class PINNForBs(nn.Module):
        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
            super().__init__()
            activation = nn.Softplus
            self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())
            self.fch = nn.Sequential(*[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS - 1)])
            self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        def forward(self, x):
            x = self.fcs(x)
            x = self.fch(x)
            x = self.fce(x)
            return x

    def train_model(self, epochs=2000, lr=0.01, N_INPUT=2, N_OUTPUT=1, N_HIDDEN=10, N_LAYERS=3):
        self._set_seed()
        self.model = self.PINNForBs(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS).to(self.device,dtype=self.dtype)

        mseloss = nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

        loss_hist = []
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Start Boundary
            Boundary_0 = self.model(self.Start_boundary)
            loss0 = torch.mean((Boundary_0.squeeze() - torch.zeros_like(Boundary_0.squeeze())) ** 2)

            # End Boundary
            Boundary_1 = self.model(self.End_boundary)
            C_boundary_end = (self.End_boundary2[:, 1] - self.E * torch.exp(-self.r * self.End_boundary2[:, 0]))
            loss1 = torch.mean((Boundary_1.squeeze() - C_boundary_end.squeeze()) ** 2)

            # Initial Condition
            Boundary_t_0 = self.model(self.Initial_boundary)
            C_boundary_t_0 = torch.maximum(self.Initial_boundary[:, 1] - self.E, torch.zeros_like(self.Initial_boundary[:, 1]))
            loss_t_0 = mseloss(Boundary_t_0.squeeze(), C_boundary_t_0)

            # Forward Pass for internal domain
            C_inner = self.model(self.Inner_domain)

            dC = torch.autograd.grad(C_inner, self.Inner_domain, torch.ones_like(C_inner), create_graph=True, retain_graph=True)[0]
            dC_dt = dC[:, 0].view(-1, 1)
            dC_ds = dC[:, 1].view(-1, 1)

            d2C = torch.autograd.grad(dC_ds, self.Inner_domain, torch.ones_like(dC_ds), create_graph=True, retain_graph=True)[0]
            d2C_d2s = d2C[:, 0].view(-1, 1)
            
            m = self.Inner_domain[:, 1].view(-1, 1)
            loss2 = torch.mean((dC_dt - (0.5 * (self.Sigma ** 2) * (m ** 2) * d2C_d2s) - self.r * m * dC_ds + self.r * C_inner[:, 0]).squeeze())

            loss = loss0 + torch.abs(loss1) + torch.abs(loss2) + loss_t_0
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_hist.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss0: {loss0}, Loss1: {loss1}, Loss2: {loss2}, Loss_t_0: {loss_t_0}")

        return loss_hist

# Example usage
# bs_pinn = BlackScholesPINN(seed=123)
# loss_history = bs_pinn.train_model(epochs=5000, lr=0.001, N_INPUT=2, N_OUTPUT=1, N_HIDDEN=8, N_LAYERS=3)


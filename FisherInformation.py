import torch
import torch.nn as nn

class FisherInformationLoss(nn.Module):
    """
    Computes the Fisher information matrix and returns its determinant as the loss.

    The purpose of this function is to calculate the Fisher information matrix, which is a measure of the amount of information that a set of random variables contains about another random variable. The determinant of the Fisher information matrix is then returned as the loss, which can be used in optimization algorithms to improve the model's performance.

    Args:
        theta_fiducial (torch.Tensor): The input tensor containing the fiducial parameters. These are the parameters that are considered as the true values in the model.
        delta_theta (torch.Tensor): The input tensor containing the variation in the fiducial parameters. This tensor represents the change in the fiducial parameters.
        data (torch.Tensor): The simulated data with shape (num_batches, num_param+1,2,dim_data). The first dim is the number of samples in the generated data, the second dim is the parameter being modified, with the 0th index being no parameter modification. The third dim is 0 for the -delta_param and 1 for +delta_param.
        coupling_cov (float): A hyperparameter that controls the coupling between the Fisher information loss and the norm of the difference between the covariance matrix and the identity matrix.

    Returns:
        torch.Tensor: The determinant of the Fisher information matrix is returned as the loss. This value represents the amount of information that the model contains about the data.
    """
    def __init__(self):
        super(FisherInformationLoss, self).__init__()
    def forward(self, theta_fiducial, delta_theta, data, coupling_cov=10.0):
        """
        Computes the Fisher information matrix and returns its determinant as the loss.

        Args:
            theta_fiducial (torch.Tensor): The input tensor containing the fiducial parameters
            delta_theta (torch.Tensor): The input tensor containing the variation in the fiducial parameters.
            data (torch.Tensor): The simulated data with shape (num_batches, num_param+1,2,dim_data). The first dim is the number of samples in the generated data, the second dim is the parameter being modified, with the 0th index being no parameter modification. The third dim is 0 for the -delta_param and 1 for +delta_param.
        """
        
        fiducial_data=data[:,0,0,:]
        perturbed_data=data[:,1:,:,:]
        
        num_param = len(theta_fiducial)
        
        data_dim = fiducial_data.shape[-1]
        num_data = fiducial_data.shape[0]

        # Numerical gradient calculation
        num_der_mean = []  

        for param_index in range(num_param):
            num_der_mean.append(
                torch.mean(
                (perturbed_data[:,param_index,0,:] - fiducial_data[:,:]) / (delta_theta[param_index]), dim=0)
                )
            
        # Compute covariance of fiducial data
        C = torch.cov(fiducial_data.T)
        if num_param == 1:
            C_inv=torch.reshape(1/C, (1,1))
            C=torch.reshape(C, (1,1))
        else:
            C_inv = torch.linalg.inv(C)

        # Compute Fisher Information Matrix
        F = torch.zeros((num_param, num_param))
        for i in range(num_param):
            for j in range(num_param):
                if num_param == 1:
                    
                
                    F[i, j] = C_inv * num_der_mean[i] * num_der_mean[j]
                    
                else:
                    F[i, j] = torch.matmul(num_der_mean[i] , torch.matmul(C_inv , num_der_mean[j]))

        
        norm_C_ID=torch.norm(C-torch.ones_like(C)) + torch.norm(C_inv-torch.ones_like(C_inv))
        
        return - torch.slogdet(F)[1]*torch.slogdet(F)[0] + coupling_cov * norm_C_ID / (norm_C_ID + torch.exp(-norm_C_ID)) * norm_C_ID  , torch.det(C), torch.det(C_inv), torch.det(F), norm_C_ID, 



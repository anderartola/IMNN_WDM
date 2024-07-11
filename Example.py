import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from FisherInformation import FisherInformationLoss
from MLP import MLP
import torch.optim as optim

if __name__ == "__main__":
    
    
    #Load fluxes
    flux_cdm=torch.from_numpy(np.exp(-np.load("Sherwood/planck1_20_1024_cdm/los_4.4/tau.npy")[0].T))
    flux_wdm12=torch.from_numpy(np.exp(-np.load("Sherwood/planck1_20_1024_wdm12/los_4.4/tau.npy")[0].T))
    flux_wdm8=torch.from_numpy(np.exp(-np.load("Sherwood/planck1_20_1024_wdm8/los_4.4/tau.npy")[0].T))
    flux_wdm4=torch.from_numpy(np.exp(-np.load("Sherwood/planck1_20_1024_wdm4/los_4.4/tau.npy")[0].T))
    flux_wdm3=torch.from_numpy(np.exp(-np.load("Sherwood/planck1_20_1024_wdm3/los_4.4/tau.npy")[0].T))
    flux_wdm2=torch.from_numpy(np.exp(-np.load("Sherwood/planck1_20_1024_wdm2/los_4.4/tau.npy")[0].T))
    flux_wdm1=torch.from_numpy(np.exp(-np.load("Sherwood/planck1_20_1024_wdm1/los_4.4/tau.npy")[0].T))

    flux_cdm=flux_cdm/torch.mean(flux_cdm) - 1
    flux_wdm1=flux_wdm1/torch.mean(flux_wdm1) - 1
    flux_wdm2=flux_wdm2/torch.mean(flux_wdm2) - 1
    flux_wdm3=flux_wdm3/torch.mean(flux_wdm3) - 1
    flux_wdm4=flux_wdm4/torch.mean(flux_wdm4) - 1
    flux_wdm8=flux_wdm8/torch.mean(flux_wdm8) - 1
    flux_wdm12=flux_wdm12/torch.mean(flux_wdm12) - 1

    freq=torch.fft.rfftfreq(n=2048)

    flux_cdm=torch.sqrt(freq)*torch.abs(torch.fft.rfft(flux_cdm, dim=1))
    flux_wdm1=torch.sqrt(freq)*torch.abs(torch.fft.rfft(flux_wdm1, dim=1))
    flux_wdm2=torch.sqrt(freq)*torch.abs(torch.fft.rfft(flux_wdm2, dim=1))
    flux_wdm3=torch.sqrt(freq)*torch.abs(torch.fft.rfft(flux_wdm3, dim=1))
    flux_wdm4=torch.sqrt(freq)*torch.abs(torch.fft.rfft(flux_wdm4, dim=1))
    flux_wdm8=torch.sqrt(freq)*torch.abs(torch.fft.rfft(flux_wdm8, dim=1))
    flux_wdm12=torch.sqrt(freq)*torch.abs(torch.fft.rfft(flux_wdm12, dim=1))
    
    
    
    #Reshuffle
    num_data = 4500
    num_data_val= 500    

    np.random.seed(0)
    index_reshuffle=np.random.choice(range(5000), replace=False, size=5000)

    flux_wdm1=flux_wdm1[index_reshuffle]
    flux_wdm2=flux_wdm2[index_reshuffle]
    flux_wdm3=flux_wdm3[index_reshuffle]
    flux_wdm4=flux_wdm4[index_reshuffle]
    flux_wdm8=flux_wdm8[index_reshuffle]
    flux_wdm12=flux_wdm12[index_reshuffle]
    flux_cdm=flux_cdm[index_reshuffle]

    flux_wdm1_train=flux_wdm1[0:num_data]
    flux_wdm1_val=flux_wdm1[num_data:]

    flux_wdm2_train=flux_wdm2[0:num_data]
    flux_wdm2_val=flux_wdm2[num_data:]

    flux_wdm3_train=flux_wdm3[0:num_data]
    flux_wdm3_val=flux_wdm3[num_data:]

    flux_wdm4_train=flux_wdm4[0:num_data]
    flux_wdm4_val=flux_wdm4[num_data:]

    flux_wdm8_train=flux_wdm8[0:num_data]
    flux_wdm8_val=flux_wdm8[num_data:]

    flux_wdm12_train=flux_wdm12[0:num_data]
    flux_wdm12_val=flux_wdm12[num_data:]

    flux_cdm_train=flux_cdm[0:num_data]
    flux_cdm_val=flux_cdm[num_data:]

    data_dim=flux_cdm.shape[-1]
    np.random.seed(0)

    torch.manual_seed(0)

    model = MLP()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

        
    # Generate the dataset



    batch_size = 500
    batch_size_val = batch_size

    #True parameters
    theta_fiducial=torch.tensor([0.0])
    delta_theta=torch.tensor([1/3])


    class PitchDataset(Dataset):
        def __init__(self, data_tensor):
            self.data = data_tensor
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]


    # Create fiducial dataset

    row0=torch.stack((flux_cdm_train,flux_cdm_train))


    row1=torch.stack((flux_wdm3_train, flux_wdm3_train))



    fiducial_data=torch.swapaxes(torch.swapaxes(torch.stack([row0,row1]),0,2),1,2 )

    print("Fiducial data shape",fiducial_data.shape)

    fiducial_dataset=PitchDataset(fiducial_data)

    #Create data loader.
    fiducial_data_loader = DataLoader(fiducial_dataset, batch_size=batch_size, shuffle=True)



    #Same for a validation dataset
    row0_val=torch.stack((flux_cdm_val,flux_cdm_val))


    row1_val=torch.stack((flux_wdm3_val, flux_wdm3_val))

    fiducial_data_val=torch.swapaxes(torch.swapaxes(torch.stack([row0_val,row1_val]),0,2),1,2 )

    print("Fiducial data val shape",fiducial_data_val.shape)


    fiducial_dataset_val=PitchDataset(fiducial_data_val)

    #Create data loader.
    fiducial_data_loader_val = DataLoader(fiducial_dataset_val, batch_size=batch_size_val, shuffle=True)


    
    
    num_epochs = 5000  # Set the number of epochs

    loss_function=FisherInformationLoss()

    training_loss=[]
    training_norm=[]
    val_loss=[]
    all_F=[]
    val_F=[]
    training_cov=[]
    val_cov=[]
    val_norm= []

    training_mean=[]
    val_mean=[]
    val_mean_der=[]
    training_mean_der=[]


    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_det_F= 0.0
        running_cov= 0.0
        running_norm = 0.0
        
        running_det_F_val = 0.0
        running_cov_val = 0.0
        running_loss_val= 0.0
        running_norm_val= 0.0
        
        for data_batch in fiducial_data_loader:   
            
            data_batch=data_batch.float()


            
            loss_list=loss_function( theta_fiducial, delta_theta, model(data_batch)  )
            loss = loss_list[0]  # Compute loss
            Det_F=loss_list[3]
            Cov=loss_list[1]
            norm=loss_list[4]
            
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model parameters
            optimizer.zero_grad()  # Zero the parameter gradients

            running_cov +=  Cov.item() * data_batch.size(0)
            running_loss += loss.item() * data_batch.size(0)  # Accumulate loss
            running_det_F += Det_F.item()* data_batch.size(0)
            running_norm += norm.item() * data_batch.size(0)
            
        #print(len(fiducial_dataset))    
        epoch_loss = running_loss / num_data  # Calculate average loss for the epoch
        epoch_F= running_det_F / num_data
        epoch_cov= running_cov / num_data
        epoch_norm= running_norm / num_data
        
        training_loss.append(epoch_loss)
        all_F.append(epoch_F)
        training_cov.append(epoch_cov )
        training_norm.append(epoch_norm)
        
        print(f'Epoch {epoch+1}/{num_epochs}, F: {epoch_F:.4f}')
        
        
        
        #Validation set
        model.eval()
        with torch.no_grad():
            for data_batch in fiducial_data_loader_val:
                data_batch=data_batch.float()
                loss_val=loss_function( theta_fiducial, delta_theta, model(data_batch))

                running_loss_val += loss_val[0].item() * data_batch.size(0)
                running_det_F_val += loss_val[3].item() * data_batch.size(0)
                running_cov_val += loss_val[1].item() * data_batch.size(0)
                running_norm_val += loss_val[4].item() * data_batch.size(0)
                
            
            val_F.append( running_det_F_val / num_data_val )
            val_cov.append( running_cov_val / num_data_val )
            val_loss.append(running_loss_val / num_data_val)
            val_norm.append( running_norm_val / num_data_val)


    plt.style.use("my_style")
    fig, (ax1,ax2,ax3, ax4)=plt.subplots(1, 4, sharex=True, figsize=(9,5))
    ax1.plot(all_F,color="blue",label="Training |F|")
    ax1.plot(val_F,color="red", label="Validation |F| ")

    ax2.plot(training_cov, color="blue",label="|C| training")
    ax2.plot(val_cov, color="red",label="|C| validation")

    ax3.plot(training_loss,color="blue",label="Training loss")
    ax3.plot(val_loss,color="red",label="Validation loss")

    ax4.plot(training_norm,color="blue",label="Training norm C-I")
    ax4.plot(val_norm,color="red",label="Validation norm C-I")


    ax1.set_xlabel("Epoch")
    ax2.set_xlabel("Epoch")

    #ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()
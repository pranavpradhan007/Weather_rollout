import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def construct_training_data(trajectories, method='all_samples'):
    """
    Construct training data from multiple KS trajectories.
    
    Parameters:
    -----------
    trajectories : array of shape (n_trajectories, n_timesteps, n_spatial)
        Multiple KS trajectories
    method : str
        'all_samples': Create X,y pairs from all trajectories combined
        'split_trajectories': Split by entire trajectories
    
    Returns:
    --------
    X_train, y_train : arrays
        Training input-output pairs
    X_val, y_val : arrays (if method='all_samples')
        Validation pairs
    X_test, y_test : arrays (if method='all_samples')
        Test pairs
    """
    n_trajectories, n_timesteps, n_spatial = trajectories.shape
    
    print(f"Input trajectories shape: {trajectories.shape}")
    print(f"  {n_trajectories} trajectories")
    print(f"  {n_timesteps} time steps per trajectory")
    print(f"  {n_spatial} spatial points per time step")
    print()
    
    # Construct input-output pairs
    X = []
    y = []
    
    for traj_idx in range(n_trajectories):
        trajectory = trajectories[traj_idx]
        
        for t in range(n_timesteps - 1):
            # Input: spatial field at time t
            X.append(trajectory[t, :])
            
            # Output: spatial field at time t+1
            y.append(trajectory[t+1, :])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Total samples created: {len(X)}")
    print(f"X shape: {X.shape} (n_samples, n_spatial)")
    print(f"y shape: {y.shape} (n_samples, n_spatial)")
    print()
    
    if method == 'all_samples':
        # Random split across all samples
        print("Splitting method: Random split across all samples")
        print("  80% train, 10% validation, 10% test")
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
        )
        
        print(f"\nTraining set:   {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set:       {X_test.shape[0]} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    elif method == 'split_trajectories':
        # Split by entire trajectories
        print("Splitting method: Hold out entire trajectories")
        
        n_train = int(0.7 * n_trajectories)
        n_val = int(0.15 * n_trajectories)
        
        # Ensure at least 1 trajectory for val and test
        if n_val == 0:
            n_val = 1
        n_test = n_trajectories - n_train - n_val
        if n_test == 0:
            n_test = 1
            n_train = n_trajectories - n_val - n_test
        
        print(f"  Training trajectories: {n_train}")
        print(f"  Validation trajectories: {n_val}")
        print(f"  Test trajectories: {n_test}")
        
        train_trajs = trajectories[:n_train]
        val_trajs = trajectories[n_train:n_train+n_val]
        test_trajs = trajectories[n_train+n_val:]
        
        # Construct X, y for each split
        X_train, y_train = [], []
        for traj in train_trajs:
            for t in range(len(traj) - 1):
                X_train.append(traj[t, :])
                y_train.append(traj[t+1, :])
        
        X_val, y_val = [], []
        for traj in val_trajs:
            for t in range(len(traj) - 1):
                X_val.append(traj[t, :])
                y_val.append(traj[t+1, :])
        
        X_test, y_test = [], []
        for traj in test_trajs:
            for t in range(len(traj) - 1):
                X_test.append(traj[t, :])
                y_test.append(traj[t+1, :])
        
        X_train, y_train = np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
        X_val, y_val = np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.float32)
        X_test, y_test = np.array(X_test, dtype=np.float32), np.array(y_test, dtype=np.float32)
        
        print(f"\nTraining set:   {X_train.shape[0]} samples from {n_train} trajectories")
        print(f"Validation set: {X_val.shape[0]} samples from {n_val} trajectories")
        print(f"Test set:       {X_test.shape[0]} samples from {n_test} trajectories")
        
        return X_train, y_train, X_val, y_val, X_test, y_test


def visualize_training_samples(X_train, y_train, x_grid, n_samples=6):
    """
    Visualize some training examples
    """
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].plot(x_grid, X_train[idx], 'b-', linewidth=1.5, label='Input: u(x,t)')
        axes[i].plot(x_grid, y_train[idx], 'r--', linewidth=1.5, label='Target: u(x,t+dt)')
        axes[i].set_xlabel('x', fontsize=10)
        axes[i].set_ylabel('u', fontsize=10)
        axes[i].set_title(f'Training Sample {idx}', fontsize=11)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_samples.png', dpi=150, bbox_inches='tight')
    print("Saved: training_samples.png")
    plt.show()


def analyze_training_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Analyze the training data statistics
    """
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    
    datasets = [
        ('Training', X_train, y_train),
        ('Validation', X_val, y_val),
        ('Test', X_test, y_test)
    ]
    
    for name, X, y in datasets:
        print(f"\n{name} Set:")
        print(f"  Samples: {len(X)}")
        print(f"  Input (X) - mean: {X.mean():.6f}, std: {X.std():.6f}")
        print(f"  Input (X) - min: {X.min():.6f}, max: {X.max():.6f}")
        print(f"  Output (y) - mean: {y.mean():.6f}, std: {y.std():.6f}")
        print(f"  Output (y) - min: {y.min():.6f}, max: {y.max():.6f}")
    
    # Compute temporal difference statistics
    print(f"\nTemporal Differences (y - X):")
    diff_train = y_train - X_train
    diff_val = y_val - X_val
    diff_test = y_test - X_test
    
    print(f"  Training - mean: {diff_train.mean():.6f}, std: {diff_train.std():.6f}")
    print(f"  Validation - mean: {diff_val.mean():.6f}, std: {diff_val.std():.6f}")
    print(f"  Test - mean: {diff_test.mean():.6f}, std: {diff_test.std():.6f}")
    
    # Plot difference distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(diff_train.flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('y - X', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Training Set: Temporal Differences', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(diff_val.flatten(), bins=100, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('y - X', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Validation Set: Temporal Differences', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(diff_test.flatten(), bins=100, alpha=0.7, color='green', edgecolor='black')
    axes[2].set_xlabel('y - X', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Test Set: Temporal Differences', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_differences.png', dpi=150, bbox_inches='tight')
    print("\nSaved: temporal_differences.png")
    plt.show()


def ks_integrate_naive(u, Lx, dt, Nt, nplot, Nspin_up=500):
    """
    Integrate Kuramoto-Sivashinsky equation using CNAB2 (Crank-Nicolson Adam-Bashforth)
    
    Parameters:
    -----------
    u : array
        Initial condition
    Lx : float
        Domain length
    dt : float
        Time step
    Nt : int
        Total number of time steps
    nplot : int
        Save every nplot time steps
        
    Returns:
    --------
    U : array of shape (Nplot, Nx)
        Saved time series data
    x : array
        Spatial grid
    t : array
        Time points where data was saved
    """
    Nx = len(u)  # number of gridpoints
    
    # Integer wavenumbers: exp(2*pi*i*kx*x/L)
    kx = np.concatenate([np.arange(0, Nx//2), 
                         np.array([0]), 
                         np.arange(-Nx//2+1, 0)])
    
    # Real wavenumbers: exp(i*alpha*x)
    alpha = 2*np.pi*kx/Lx
    
    # Operators in Fourier space
    D = 1j*alpha                    # D = d/dx operator
    L = alpha**2 - alpha**4         # linear operator -D^2 - D^4
    G = -0.5*D                      # -1/2 D operator
    
    Nplot = int(np.round(Nt/nplot)) + 1  # total number of saved time steps
    
    # Spatial and temporal grids
    x = np.arange(Nx)*Lx/Nx
    t = np.arange(Nplot)*dt*nplot
    U = np.zeros((Nplot, Nx))
    
    # Convenience variables for CNAB2
    dt2 = dt/2
    dt32 = 3*dt/2
    A = np.ones(Nx) + dt2*L
    B = (np.ones(Nx) - dt2*L)**(-1)
    
    # -1/2 d/dx(u^2) = -u u_x, collocation calculation
    Nn = G * fft(u*u)
    Nn1 = Nn.copy()
    
    U[0, :] = u  # save initial value u to matrix U
    np_counter = 1  # counter for saved data
    
    # Transform data to spectral coefficients
    u = fft(u)
     # number of initial steps to skip for spin-up
    # Timestepping loop
    for n in range(Nt+Nspin_up):
        Nn1 = Nn.copy()                           # shift N^{n-1} <- N^n
        Nn = G * fft(np.real(ifft(u))**2)        # compute N^n = -u u_x
        
        u = B * (A * u + dt32*Nn - dt2*Nn1)      # CNAB2 formula
        
        if (n+1) % nplot == 0 and n > Nspin_up:
            U[np_counter, :] = np.real(ifft(u))
            np_counter += 1
    
    return U, x, t

def ks_integrate_naive(u, Lx, dt, Nt, nplot, Nspin_up=500):
    """
    Integrate Kuramoto-Sivashinsky equation using CNAB2 (Crank-Nicolson Adam-Bashforth)
    
    Parameters:
    -----------
    u : array
        Initial condition
    Lx : float
        Domain length
    dt : float
        Time step
    Nt : int
        Total number of time steps
    nplot : int
        Save every nplot time steps
        
    Returns:
    --------
    U : array of shape (Nplot, Nx)
        Saved time series data
    x : array
        Spatial grid
    t : array
        Time points where data was saved
    """
    Nx = len(u)  # number of gridpoints
    
    # Integer wavenumbers: exp(2*pi*i*kx*x/L)
    kx = np.concatenate([np.arange(0, Nx//2), 
                         np.array([0]), 
                         np.arange(-Nx//2+1, 0)])
    
    # Real wavenumbers: exp(i*alpha*x)
    alpha = 2*np.pi*kx/Lx
    
    # Operators in Fourier space
    D = 1j*alpha                    # D = d/dx operator
    L = alpha**2 - alpha**4         # linear operator -D^2 - D^4
    G = -0.5*D                      # -1/2 D operator
    
    Nplot = int(np.round(Nt/nplot)) + 1  # total number of saved time steps
    
    # Spatial and temporal grids
    x = np.arange(Nx)*Lx/Nx
    t = np.arange(Nplot)*dt*nplot
    U = np.zeros((Nplot, Nx))
    
    # Convenience variables for CNAB2
    dt2 = dt/2
    dt32 = 3*dt/2
    A = np.ones(Nx) + dt2*L
    B = (np.ones(Nx) - dt2*L)**(-1)
    
    # -1/2 d/dx(u^2) = -u u_x, collocation calculation
    Nn = G * fft(u*u)
    Nn1 = Nn.copy()
    
    U[0, :] = u  # save initial value u to matrix U
    np_counter = 1  # counter for saved data
    
    # Transform data to spectral coefficients
    u = fft(u)
     # number of initial steps to skip for spin-up
    # Timestepping loop
    for n in range(Nt+Nspin_up):
        Nn1 = Nn.copy()                           # shift N^{n-1} <- N^n
        Nn = G * fft(np.real(ifft(u))**2)        # compute N^n = -u u_x
        
        u = B * (A * u + dt32*Nn - dt2*Nn1)      # CNAB2 formula
        
        if (n+1) % nplot == 0 and n > Nspin_up:
            U[np_counter, :] = np.real(ifft(u))
            np_counter += 1
    
    return U, x, t


# Function to generate multiple trajectories for ML training
def generate_multiple_trajectories(n_trajectories=5, **kwargs):
    """
    Generate multiple KS trajectories with different initial conditions
    
    Parameters:
    -----------
    n_trajectories : int
        Number of different trajectories to generate
    **kwargs : dict
        Parameters to pass to ks_integrate_naive (Lx, dt, Nt, nplot)
        
    Returns:
    --------
    trajectories : list of arrays
        List of trajectory arrays, each of shape (Nplot, Nx)
    """
    Lx = kwargs.get('Lx', 128)
    Nx = kwargs.get('Nx', 1024)
    dt = kwargs.get('dt', 1/16)
    Nt = kwargs.get('Nt', 1600)
    nplot = kwargs.get('nplot', 8)
    Nspin_up = kwargs.get('Nspin_up', 300)
    
    trajectories = []
    x = Lx * np.arange(Nx) / Nx
    
    print(f"Generating {n_trajectories} trajectories...")
    for i in range(n_trajectories):
        # Generate random initial condition
        np.random.seed(i)
        n_modes = 10
        u0 = np.zeros(Nx)
        for k in range(1, n_modes+1):
            phase1 = np.random.rand() * 2 * np.pi
            phase2 = np.random.rand() * 2 * np.pi
            amp = 1.0 / k
            u0 += amp * np.cos(2*np.pi*k*x/Lx + phase1)
            u0 += amp * np.sin(2*np.pi*k*x/Lx + phase2)
        
        U, _, _ = ks_integrate_naive(u0, Lx, dt, Nt, nplot, Nspin_up)
        trajectories.append(U)
        print(f"  Trajectory {i+1}/{n_trajectories} complete, shape: {U.shape}")
    
    return trajectories, x,  np.arange(nplot) * dt * nplot
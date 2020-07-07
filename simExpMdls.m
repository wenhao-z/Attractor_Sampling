% The main code performing all kinds of simulations on a varieties of
% tasks and models in demonstrating the Langevin sampling in attractor networks

% Wen-Hao Zhang, April 13, 2020


setWorkPath;
addpath(fullfile(Path_RootDir, 'simExp'));

%%
flagTask = 6;
% 1. Demo of the Langevin sampling (OU process) of 1d stimulus feature
% 2. Demo of the Langevin sampling of 2d stimulus features
% 3. Demo of the Langevin sampling in a single CANN
% 4. Scan parameters of Langevin sampling in a single CANN
% 5. Demo of the 2D Langevin sampling in coupled CANNs
% 6. Scan parameters of Langevin sampling in two coupled CANNs
% 7. Scan parameters of Langevin sampling in multiple coupled CANNs
% 8. Demo of Langevin sampling in a chaotic CANN

switch flagTask
    case 1
        simOUProcess;
    case 2
        demoLangevinSampling2D;
    case 3
        demoCANNSampling;
    case 4
        scanCANNSampling;
    case 5
        demoCoupledCANNSampling;
    case 6
        scanCoupledCANNSampling;
    case 7
        scanMultiCoupledCANNSampling;
    case 8
        demoChaoticCANNSampling;
end
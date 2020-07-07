% Default parameters of network
NetPars.N        = 180;  % The number of neurons
NetPars.numNets  = 1;  % number of networks
NetPars.Width    = 180; % the range of parameter space from (-Width, Width), unit: deg

% Preferred stimulus of neurons (location on feature space)
PrefStim         = linspace(-NetPars.Width,NetPars.Width, NetPars.N+1)'; 
PrefStim(1)      = [];
NetPars.PrefStim = PrefStim;
clear PrefStim

%% Temporal parameters
NetPars.tau  = 1; % Time constant of neuron activity
NetPars.tLen = 600 * NetPars.tau; % whole length of simulation
NetPars.dt   = NetPars.tau/100; % the iterative step

%% Connection
% NetPars.connFunc = 'vonMises'; % or Gaussian
NetPars.connFunc = 'Gaussian'; % or vonMises
switch NetPars.connFunc
    case 'Gaussian'
        NetPars.k         = 5e-4; % global inhibition strength
        NetPars.TunWidth  = 40; % Tuning width, the std. of tuning function. Unit: deg.
    case 'vonMises'
        NetPars.k           = 3e-4; % global inhibition strength
        NetPars.TunKappa    = 3; % Tuning width, concentration of von-Mises function, about 40 deg.
end
NetPars.JrcRatio = 0.5; % Recurrent connection strength within the same network, relative to Jc (the minimal recurrent 
%                         connection strength for the network to hold a persistent activity without feedforward inputs).
NetPars.JrpRatio = 0.4; % Reciprocal connection strength between networks; same parameter for J12 and J21, relative to Jrc

%% Network input 
% -----------------------------
% Input intensity and location
% -----------------------------
% Peak intensity of feedforward inputs, [numNets, 1]
NetPars.AmplRatio = 0.6; % Relative to Uc, which is the persistent bump height without stimulus when Jrc = Jc 
NetPars.AmplRatio = repmat(NetPars.AmplRatio, [NetPars.numNets, 1]);

% Intensity of background input
NetPars.AmplBkg = 0; 

% Input location, [numNets, 1]
NetPars.Posi = 0;
clear Posi

% ------------------
% Noise structure
% ------------------
NetPars.bAddNoise = 1; % 1: add noise; 0: noise free;
% NetPars.PosiNoise = 0; % bool variable, 1: noise on the position of external input; 0: full noise

% Internal noise inside network
% The noise strength of all networks are the same for simplicity
NetPars.fanoFactor = 1; % fano factor of internal noise

% External noise associated with feedforward connections
% NetPars.typeExtNois = 'Poisson'; % or 'Gaussian'

% ------------------
% Cueing conditions
% ------------------
NetPars.cueCond = 0; % Cueing condition. 
%                         0: both cue; 
%                         1: only cue 1; 
%                         2: only cue 2.

% Random seed
NetPars.seedNois = 0;
% NetPars.seedIntNois = 0;
% NetPars.seedExtNois = sum(clock)*100;

NetPars.flagSeed = 1;
switch NetPars.flagSeed
    case 1
        NetPars.flagSeed = 'sameSeed';
        % use the same random seed for all parameters 
    case 2
        NetPars.flagSeed = 'SameSeedCueCond';
        % different random seed under different parameter settings, but for
        % each parameter set, the seeds under three cue conditions are
        % exactly the same
    case 3
        NetPars.flagSeed = 'diffSeed';
end

NetPars = orderfields(NetPars);
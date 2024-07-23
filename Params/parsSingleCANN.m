% Parameters for coupled two CANNs
% Wen-Hao Zhang, oct-06, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

%% Load default network parameters
defaultNetPars;

%% Specialized parameters of CANN

NetPars.numNets = 1;
NetPars = getDependentPars(NetPars);

%% Caution: don't move below lines before parseNetPars
% NetPars.JrcRatio      = 0.5;
% NetPars.JrcRatio      = 1./(NetPars.rho * sqrt(2))/NetPars.Jc;

% NetPars.JrcRatio      = 1./sqrt(2)/NetPars.Jc; % Old version before Apr 14, 2020
NetPars.JrcRatio      = 0.5;
NetPars.JRand         = 0;

% -----------------
% Input parameters
% -----------------
NetPars.AmplRatio = 0.8; % Relative to Uc, which is the persistent bump height without stimulus when Jrc = Jc 
NetPars.Posi      = 0; % the last element is used for different position

NetPars.stdIntNois = sqrt(NetPars.AmplBkg * NetPars.fanoFactor); % internal noise

NetPars.bAddNoise = 1;
NetPars.AmplBkg = 0;

% Parameters of multiple trials
NetPars.tTrial     = 500 * NetPars.tau;
NetPars.nTrials    = 1; % number of trials
NetPars.tLen       = NetPars.nTrials * NetPars.tTrial;
NetPars.tStat      = 50 * NetPars.tau; % The starting time to make statistics

%%
% Parse network parameters
parseNetPars;

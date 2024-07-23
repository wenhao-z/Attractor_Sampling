% Parameters for coupled two CANNs
% Wen-Hao Zhang, Oct-06, 2016
% wenhaoz1@andrew.cmu.edu
% @Carnegie Mellon University

%% Load default network parameters
defaultNetPars;

%% Specialized parameters of CANN
NetPars.numNets = 2;
NetPars = getDependentPars(NetPars);

%% Caution: don't move below lines before parseNetPars
% NetPars.Jrc      = 0.5*NetPars.Jc;
NetPars.JrcRatio = 1./sqrt(2)/NetPars.Jc;
NetPars.JrpRatio = 0.7; % same parameter for J12 and J21, relative to Jrc

% -----------------
% Input parameters
% -----------------
NetPars.AmplRatio = 0.7*NetPars.Uc;
NetPars.AmplRatio = repmat(NetPars.AmplRatio, [NetPars.numNets, 1]);

NetPars.Posi = ...
    [NetPars.PrefStim(end/2-1), ...
    NetPars.PrefStim(end/2+1)]'; % the last element is used for different position

NetPars.stdIntNois = sqrt(NetPars.AmplBkg * NetPars.fanoFactor); % internal noise
NetPars.stdIntNois = repmat(NetPars.stdIntNois, [NetPars.numNets, 1]);

NetPars.bAddNoise = 1;
NetPars.AmplBkg = 0;

% Internal variability
NetPars.fanoFactor = 0.5;

% Parameters of multiple trials
NetPars.tTrial  = 500 * NetPars.tau;
NetPars.nTrials = 1; % number of trials
NetPars.tLen    = NetPars.nTrials * NetPars.tTrial;
NetPars.tStat   = 50 * NetPars.tau; % The starting time to make statistics
%%
% Parse network parameters
parseNetPars;
